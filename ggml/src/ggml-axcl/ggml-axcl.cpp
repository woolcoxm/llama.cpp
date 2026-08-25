// ggml-axcl: Axera AXCL NPU backend for ggml
//
// Targets the M.2 accelerator cards (AX8850/AX650) on hosts like the
// Raspberry Pi 5. The AXCL runtime is graph-only: the NPU executes
// precompiled .axmodel graphs, not individual ops. The long-term plan
// (see DESIGN-AXCL.md) is to dispatch MUL_MAT to precompiled matmul
// graphs with weights staged in on-card CMM.
//
// Milestone 1 scope: register the device, expose CMM buffers, report
// properties. supports_op() returns false for everything so the
// scheduler never routes work here until a real op lands.

#include "ggml-axcl.h"

#include "axcl.h"
#include "axcl_rt_device.h"
#include "axcl_rt_memory.h"
#include "axcl_rt_type.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_AXCL_NAME "AXCL"
#define AXCL_MAX_BUFFER_SIZE 7ull * 1024ull * 1024ull * 1024ull // TODO: query freeCmmSize at alloc time

static ggml_guid_t ggml_backend_axcl_guid() {
    // fixed random guid
    static ggml_guid guid = {0x8a, 0x41, 0x2e, 0xc7, 0x5b, 0x63, 0x4d, 0xf1,
                             0x9a, 0x30, 0x6e, 0x2b, 0xd4, 0x77, 0xc3, 0x21};
    return &guid;
}

static size_t axcl_alloc_alignment() {
    return 4096; // page-align CMM allocations; conservative for DMA
}

static inline size_t axcl_aligned_size(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

static ggml_backend_dev_t ggml_backend_axcl_reg_dev(int32_t device);

//
// global init
//

static std::mutex axcl_global_mutex;

static bool axcl_global_init() {
    static bool initialized = false;
    static bool available   = false;
    std::lock_guard<std::mutex> lock(axcl_global_mutex);
    if (!initialized) {
        axclError err = axclInit(nullptr);
        if (err != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: axclInit failed with %d\n", (int) err);
            available = false;
        } else {
            available = true;
        }
        initialized = true;
    }
    return available;
}

static int32_t axcl_get_device_count() {
    if (!axcl_global_init()) {
        return 0;
    }
    uint32_t count = 0;
    if (axclrtGetDeviceCount(&count) != AXCL_SUCC) {
        return 0;
    }
    return (int32_t) count;
}

// the card's slot index is NOT 0 on switch-topology HATs (observed: 3).
// axclrtGetDeviceList() is also the mandatory activation probe: without it
// the device manager reports every device as "not connected"
static int32_t axcl_get_device_index(int32_t ordinal) {
    axclrtDeviceList dl;
    memset(&dl, 0, sizeof(dl));
    if (axclrtGetDeviceList(&dl) != AXCL_SUCC || dl.num == 0) {
        return 0;
    }
    return dl.devices[ordinal < (int32_t) dl.num ? ordinal : 0];
}

static std::string axcl_get_device_description(int32_t device) {
    axclrtDeviceProperties props;
    memset(&props, 0, sizeof(props));
    if (axclrtGetDeviceProperties(device, &props) == AXCL_SUCC) {
        return std::string("Axera ") + props.swVersion + " (" + std::to_string(props.totalCmmSize / 1024) +
               " MiB CMM)";
    }
    return "Axera NPU";
}

//
// buffer (device memory / CMM)
//

struct ggml_backend_axcl_buffer_context {
    int32_t device;
    void *  ptr;   // plain host memory: CMM pointers are NOT cpu-dereferenceable,
    size_t  size;  // so ggml-visible buffers stay in host RAM; the matmul
};                  // engines stage through CMM internally via axclrtMemcpy

static void ggml_backend_axcl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    free(ctx->ptr);
    delete ctx;
}

static void * ggml_backend_axcl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    return ctx->ptr;
}

static void ggml_backend_axcl_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                   uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void ggml_backend_axcl_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_axcl_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static bool ggml_backend_axcl_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                                struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_axcl_buffer_set_tensor(buffer, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_axcl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    memset(ctx->ptr, value, ctx->size);
}

static const struct ggml_backend_buffer_i ggml_backend_axcl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_axcl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_axcl_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_axcl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_axcl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_axcl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_axcl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_axcl_buffer_clear,
    /* .reset           = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_axcl_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "AXCL";
}

static ggml_backend_buffer_t ggml_backend_axcl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                        size_t size) {
    size = axcl_aligned_size(size, axcl_alloc_alignment());

    // fusion routes small norm/gate weights into our buffer beyond llama.cpp's
    // size estimate; add slack so the ctx allocator never runs dry
    const size_t slack = 64ull * 1024 * 1024;
    fprintf(stderr, "[axcl-buf] alloc_buffer(%zu MB + %zu slack)\n", size / 1024 / 1024, slack / 1024 / 1024);
    void * ptr = malloc(size + slack);
    if (ptr == nullptr) {
        fprintf(stderr, "[axcl-buf] HOST ALLOC FAILED for %zu bytes\n", size + slack);
        GGML_LOG_ERROR("ggml-axcl: host alloc failed for %zu bytes\n", size + slack);
        return ggml_backend_buffer_init(buft, ggml_backend_axcl_buffer_interface, nullptr, 0);
    }

    auto * ctx = new ggml_backend_axcl_buffer_context{0, ptr, size + slack};

    return ggml_backend_buffer_init(buft, ggml_backend_axcl_buffer_interface, ctx, size + slack);
}

static size_t ggml_backend_axcl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return axcl_alloc_alignment();
}

static size_t ggml_backend_axcl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return AXCL_MAX_BUFFER_SIZE;
}

static const struct ggml_backend_buffer_type_i axcl_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_axcl_buffer_type_name,
    /* .alloc_buffer   = */ ggml_backend_axcl_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_axcl_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_axcl_buffer_type_get_max_size,
    /* .get_alloc_size = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_axcl_buffer_type(int32_t device) {
    static std::vector<ggml_backend_buffer_type_t> bufts;
    static bool                                     initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int32_t i = 0; i < axcl_get_device_count(); i++) {
                bufts.push_back(new ggml_backend_buffer_type{
                    /* .iface    = */ axcl_buffer_type_interface,
                    /* .device   = */ ggml_backend_axcl_reg_dev(i),
                    /* .context  = */ nullptr,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device >= 0 && device < (int32_t) bufts.size());

    return bufts[device];
}

//
// matmul engines (milestone 2)
//
// A GEMM .axmodel compiled by Pulsar2 with runtime inputs:
//   Y[M,N] = X[M,K] (f16) @ W[K,N] (f16)
// Shapes are static per engine; we look them up by (K, N) from
// $AXCL_MATMUL_DIR (default /usr/local/share/ggml-axcl/matmul) with file
// name pattern: matmul_m1_k{K}_n{N}.axmodel
//
// First light: X/W are staged in host buffers per call (the runtime DMAs
// them to the card). Weight staging in CMM with upload-once semantics is
// milestone 3.
//

#include "axcl_rt_engine.h"
#include "axcl_rt_engine_type.h"

struct axcl_matmul {
    uint64_t model_id   = 0;
    uint64_t context_id = 0;
    axclrtEngineIOInfo io_info = nullptr;
    axclrtEngineIO     io      = nullptr;

    int64_t m = 0, k = 0, n = 0;

    std::string x_name, w_name, y_name;
    int x_idx = -1, w_idx = -1, y_idx = -1;
    axclrtEngineDataType x_type = AXCL_DATA_TYPE_NONE;
    axclrtEngineDataType w_type = AXCL_DATA_TYPE_NONE;
    axclrtEngineDataType y_type = AXCL_DATA_TYPE_NONE;

    // canonical axcl-samples pattern: DEVICE buffers bound to the engine IO
    // (host pointers are rejected card-side at execute time), plus host
    // staging used through axclrtMemcpy
    void * dx = nullptr;                 // f32 [K]
    void * dw = nullptr;                 // f32 [K, N] (transposed weights)
    void * dy = nullptr;                 // f32 [N]
    std::vector<float> x_h, y_h;        // host staging (per-call)
    // upload-once weight staging: many weight tensors share one engine shape
    // (every layer's k_proj is the same (k,n)) so device buffers are keyed
    // by the weight tensor's data pointer, not the engine
    std::unordered_map<const void *, void *> dev_w;
    std::vector<float>                    w_h; // scratch for the one upload
};

static void axcl_preload_all_engines();

static axclrtContext g_axcl_ctx = 0; // thread-local bind target for worker threads

// stage profiling: micros accumulated per stage, reported every REPORT computes
#include <cstdint>
static uint64_t prof_stage_wstage = 0, prof_stage_xh2d = 0, prof_stage_bind = 0,
                prof_stage_exec = 0, prof_stage_yd2h = 0, prof_stage_total = 0,
                prof_wstaged = 0, prof_count = 0;
#define PROF_REPORT_EVERY 120
static void prof_report() {
    if (prof_count % PROF_REPORT_EVERY != 0 || prof_count == 0) return;
    fprintf(stderr,
        "[axcl-prof] computes=%llu wstaged=%llu | avg micros/compute: total=%llu wstage=%llu xh2d=%llu bind=%llu exec=%llu yd2h=%llu other=%llu\n",
        (unsigned long long) prof_count, (unsigned long long) prof_wstaged,
        (unsigned long long) (prof_stage_total / prof_count),
        (unsigned long long) (prof_stage_wstage / prof_count),
        (unsigned long long) (prof_stage_xh2d / prof_count),
        (unsigned long long) (prof_stage_bind / prof_count),
        (unsigned long long) (prof_stage_exec / prof_count),
        (unsigned long long) (prof_stage_yd2h / prof_count),
        (unsigned long long) ((prof_stage_total - prof_stage_wstage - prof_stage_xh2d -
                               prof_stage_bind - prof_stage_exec - prof_stage_yd2h) / prof_count));
    fflush(stderr);
}

// card-resident weight pool: ONE CMM allocation carved bump-style so weights
// upload exactly once (verified tolerable by test_pool: single malloc +
// carved slots + interleaved executes). Falls back to per-call upload when
// exhausted. Size via GGML_AXCL_WPOOL_MB (default 2560).
struct axcl_weight_pool {
    char *  base = nullptr;
    char *  bump = nullptr;
    size_t  remain = 0;
    void * carve(size_t bytes) {
        const size_t align = 4096;
        size_t need = (bytes + align - 1) & ~(align - 1);
        if (base == nullptr || need > remain) {
            return nullptr;
        }
        void * p = bump;
        bump += need;
        remain -= need;
        return p;
    }
};
static axcl_weight_pool g_axcl_pool;

static void axcl_weight_pool_init() {
    fprintf(stderr, "[axcl-pool] pool_init called\n");
    if (g_axcl_pool.base != nullptr) {
        return;
    }
    const char * env = getenv("GGML_AXCL_WPOOL_MB");
    size_t mb = env ? strtoull(env, nullptr, 10) : 2560;
    if (mb == 0) {
        return; // pool disabled: always per-call upload
    }
    void * p = nullptr;
    if (axclrtMalloc(&p, mb * 1024 * 1024, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC || !p) {
        GGML_LOG_WARN("ggml-axcl: weight pool alloc (%zu MB) failed; using per-call uploads\n", mb);
        return;
    }
    g_axcl_pool.base = g_axcl_pool.bump = (char *) p;
    fprintf(stderr, "[axcl-pool] pool alloc OK\n");
    g_axcl_pool.remain = mb * 1024 * 1024;
    GGML_LOG_INFO("ggml-axcl: weight pool ready (%zu MB)\n", mb);
}

static bool axcl_engine_global_init() {
    static bool initialized = false;
    static bool available   = false;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {
        // activation sequence discovered the hard way: GetDeviceList probes
        // and activates the device (also yields the real slot index);
        // CreateContext/SetCurrentContext must precede EngineInit
        int32_t dev = axcl_get_device_index(0);
        axclrtSetDevice(dev);
        axclrtContext ctx = 0;
        axclError  err  = axclrtCreateContext(&ctx, dev);
        if (err == AXCL_SUCC) {
            g_axcl_ctx = ctx;
            axclrtSetCurrentContext(ctx);
            err = axclrtEngineInit(AXCL_VNPU_DISABLE);
        }
        available = (err == AXCL_SUCC);
        if (!available) {
            GGML_LOG_ERROR("ggml-axcl: engine activation failed with %d\n", (int) err);
        }
        initialized = true;
    }
    return available;
}

static std::string axcl_matmul_model_path(int64_t k, int64_t n) {
    const char * dir = getenv("AXCL_MATMUL_DIR");
    char         path[512];
    snprintf(path, sizeof(path), "%s/matmul_m1_k%lld_n%lld.axmodel",
             dir ? dir : "/usr/local/share/ggml-axcl/matmul", (long long) k, (long long) n);
    return path;
}

// the AXCL engine IO is not thread-safe: serialize loads and executes
static std::mutex axcl_exec_mutex;

static axcl_matmul * axcl_matmul_load(int64_t k, int64_t n) {
    if (!axcl_engine_global_init()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> exec_lock(axcl_exec_mutex);

    axcl_matmul * mm = new axcl_matmul();
    mm->m = 1;
    mm->k = k;
    mm->n = n;

    std::string path = axcl_matmul_model_path(k, n);
    fprintf(stderr, "[axcl-dbg] loading %s\n", path.c_str());
    if (axclrtEngineLoadFromFile(path.c_str(), &mm->model_id) != AXCL_SUCC) {
        GGML_LOG_WARN("ggml-axcl: no matmul engine at %s\n", path.c_str());
        delete mm;
        return nullptr;
    }

    if (axclrtEngineCreateContext(mm->model_id, &mm->context_id) != AXCL_SUCC ||
        axclrtEngineGetIOInfo(mm->model_id, &mm->io_info) != AXCL_SUCC ||
        axclrtEngineCreateIO(mm->io_info, &mm->io) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: failed to create context/IO for %s\n", path.c_str());
        delete mm;
        return nullptr;
    }

    uint32_t n_in = axclrtEngineGetNumInputs(mm->io_info);
    uint32_t n_out = axclrtEngineGetNumOutputs(mm->io_info);

    // expected: 2 inputs (X, W) + 1 output (Y); the compiled graph keeps
    // f32 boundaries (quantization is internal)
    for (uint32_t i = 0; i < n_in && i < 2; i++) {
        const char * name = axclrtEngineGetInputNameByIndex(mm->io_info, i);
        if (name != nullptr) {
            (i == 0 ? mm->x_name : mm->w_name) = name;
            axclrtEngineDataType t = AXCL_DATA_TYPE_NONE;
            if (axclrtEngineGetInputDataType(mm->io_info, i, &t) == AXCL_SUCC) {
                (i == 0 ? mm->x_type : mm->w_type) = t;
            }
        }
    }
    for (uint32_t i = 0; i < n_out && i < 1; i++) {
        const char * name = axclrtEngineGetOutputNameByIndex(mm->io_info, i);
        if (name != nullptr) {
            mm->y_name = name;
            axclrtEngineDataType t = AXCL_DATA_TYPE_NONE;
            if (axclrtEngineGetOutputDataType(mm->io_info, i, &t) == AXCL_SUCC) {
                mm->y_type = t;
            }
        }
    }
    fprintf(stderr, "[axcl-dbg] io names ok\n");
    mm->x_idx = axclrtEngineGetInputIndexByName(mm->io_info, mm->x_name.c_str());
    mm->w_idx = axclrtEngineGetInputIndexByName(mm->io_info, mm->w_name.c_str());
    mm->y_idx = axclrtEngineGetOutputIndexByName(mm->io_info, mm->y_name.c_str());

    mm->x_h.resize(k);
    mm->w_h.resize(k * n);
    mm->y_h.resize(n);
    if (axclrtMalloc(&mm->dx, (size_t) k * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMalloc(&mm->dw, (size_t) k * n * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMalloc(&mm->dy, (size_t) n * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: CMM alloc failed for %s\n", path.c_str());
        delete mm;
        return nullptr;
    }

    fprintf(stderr, "[axcl-dbg] engine ready\n");
    GGML_LOG_INFO("ggml-axcl: matmul engine loaded %s (inputs '%s','%s' -> '%s', idx %d,%d -> %d)\n",
                  path.c_str(), mm->x_name.c_str(), mm->w_name.c_str(), mm->y_name.c_str(), mm->x_idx,
                  mm->w_idx, mm->y_idx);

    return mm;
}

// shape-keyed engine cache; after preload it is final (no IO on hot paths)
struct axcl_matmul_cache_t {
    std::mutex mutex;
    std::unordered_map<uint64_t, axcl_matmul *> engines; // key: (k << 32) | n
    std::unordered_map<uint64_t, bool>        missing;
    bool                                       preloaded = false;
};

static axcl_matmul_cache_t * axcl_matmul_cache() {
    static axcl_matmul_cache_t cache;
    return &cache;
}

static axcl_matmul * axcl_matmul_get(int64_t k, int64_t n) {
    axcl_matmul_cache_t * cache = axcl_matmul_cache();

    const uint64_t key = ((uint64_t) k << 32) | (uint32_t) n;

    std::lock_guard<std::mutex> lock(cache->mutex);
    auto it = cache->engines.find(key);
    if (it != cache->engines.end()) {
        return it->second;
    }
    if (cache->missing.count(key) || cache->preloaded) {
        return nullptr; // after preload the cache is final: no IO on hot paths
    }
    axcl_matmul * mm = axcl_matmul_load(k, n);
    if (mm) {
        cache->engines[key] = mm;
    } else {
        cache->missing[key] = true;
    }
    return mm;
}

// load every matmul_m1_k*_n*.axmodel in AXCL_MATMUL_DIR once, at backend
// startup. Engine loads touch the card (LoadFromFile/CreateIO/CMM) and
// must never race a concurrent Execute - the scheduler probes supports_op
// from its own thread mid-graph, which deadlocked the runtime
static void axcl_preload_all_engines() {
    axcl_matmul_cache_t * cache = axcl_matmul_cache();
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (cache->preloaded) {
        return;
    }
    cache->preloaded = true;

    const char * dir = getenv("AXCL_MATMUL_DIR");
    std::string d = dir ? dir : "/usr/local/share/ggml-axcl/matmul";
    DIR * dp = opendir(d.c_str());
    if (!dp) {
        return;
    }
    struct dirent * de;
    int loaded = 0;
    while ((de = readdir(dp)) != nullptr) {
        int64_t k, n;
        if (sscanf(de->d_name, "matmul_m1_k%lld_n%lld.axmodel", (long long *) &k, (long long *) &n) == 2) {
            const uint64_t key = ((uint64_t) k << 32) | (uint32_t) n;
            if (cache->engines.count(key) || cache->missing.count(key)) {
                continue;
            }
            axcl_matmul * mm = axcl_matmul_load(k, n);
            if (mm) {
                cache->engines[key] = mm;
                loaded++;
            } else {
                cache->missing[key] = true;
            }
        }
    }
    closedir(dp);
    GGML_LOG_INFO("ggml-axcl: preloaded %d matmul engines from %s\n", loaded, d.c_str());
}

// dequantize src0 (ggml weight, [K x N] row-major, ne0 = K) into a
// transposed f32 [K, N] row-major host buffer: w[k*N + n] = src0[n][k].
// works for every ggml type via the public type-trait dequantizers
static void axcl_dequant_any_to_f32_transposed(const struct ggml_tensor * t, float * w32) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    if (t->type == GGML_TYPE_F32) {
        const float * w = (const float *) t->data;
        for (int64_t nn = 0; nn < n; nn++) {
            for (int64_t kk = 0; kk < k; kk++) {
                w32[kk * n + nn] = w[nn * k + kk];
            }
        }
        return;
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
    GGML_ASSERT(traits && traits->to_float);
    std::vector<float> row(k);
    for (int64_t nn = 0; nn < n; nn++) {
        traits->to_float((const void *) ((char *) t->data + nn * t->nb[1]), row.data(), k);
        for (int64_t kk = 0; kk < k; kk++) {
            w32[kk * n + nn] = row[kk];
        }
    }
}

static bool ggml_axcl_compute_mul_mat(axcl_matmul * mm, const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    std::lock_guard<std::mutex> exec_lock(axcl_exec_mutex);
    auto t0 = std::chrono::steady_clock::now();
    const int64_t k = mm->k;
    const int64_t n = mm->n;

    // X: activation, src1 is [K, 1] f32 contiguous
    memcpy(mm->x_h.data(), src1->data, (size_t) k * 4);

    // W: card-resident when staged, else one-time staging into the pool
    void * dw = nullptr;
    auto   wit = mm->dev_w.find(src0->data);
    if (wit != mm->dev_w.end()) {
        dw = wit->second;
    }
    bool w_uploaded = (dw != nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (!w_uploaded) {
        if (mm->w_h.empty()) {
            mm->w_h.resize(k * n);
        }
        axcl_dequant_any_to_f32_transposed(src0, mm->w_h.data());
        void * slot = g_axcl_pool.carve((size_t) k * n * 4);
        if (slot != nullptr &&
            axclrtMemcpy(slot, mm->w_h.data(), (size_t) k * n * 4, AXCL_MEMCPY_HOST_TO_DEVICE) == AXCL_SUCC) {
            mm->dev_w[src0->data] = slot;
            dw = slot; // staged: future calls take the w_uploaded path
        }
        // pool miss (or H2D fail): dw stays null -> per-call upload below
    }

    auto t2 = std::chrono::steady_clock::now();
    if (axclrtMemcpy(mm->dx, mm->x_h.data(), (size_t) k * 4, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: X H2D failed\n");
        return false;
    }
    if (!w_uploaded) {
        if (axclrtMemcpy(mm->dw, mm->w_h.data(), (size_t) k * n * 4, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: W H2D failed\n");
            return false;
        }
    }
    auto t3 = std::chrono::steady_clock::now();

    void * wbuf = w_uploaded ? dw : mm->dw;
    if (axclrtEngineSetInputBufferByIndex(mm->io, mm->x_idx, mm->dx, (size_t) k * 4) != AXCL_SUCC ||
        axclrtEngineSetInputBufferByIndex(mm->io, mm->w_idx, wbuf, (size_t) k * n * 4) != AXCL_SUCC ||
        axclrtEngineSetOutputBufferByIndex(mm->io, mm->y_idx, mm->dy, (size_t) n * 4) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: bind device buffers failed\n");
        return false;
    }
    auto t4 = std::chrono::steady_clock::now();

    axclError ex = axclrtEngineExecute(mm->model_id, mm->context_id, 0, mm->io);
    auto t5 = std::chrono::steady_clock::now();
    if (ex != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: engine execute failed\n");
        return false;
    }

    if (axclrtMemcpy(mm->y_h.data(), mm->dy, (size_t) n * 4, AXCL_MEMCPY_DEVICE_TO_HOST) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: D2H copy failed\n");
        return false;
    }
    memcpy(dst->data, mm->y_h.data(), (size_t) n * 4);
    auto t6 = std::chrono::steady_clock::now();

    auto us = [](auto a, auto b) { return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
    prof_stage_wstage += us(t0, t1) + us(t1, t2); // staging check + one-time upload
    prof_stage_xh2d   += us(t2, t3);
    prof_stage_bind   += us(t3, t4);
    prof_stage_exec   += us(t4, t5);
    prof_stage_yd2h   += us(t5, t6);
    prof_stage_total  += us(t0, t6);
    if (w_uploaded) prof_wstaged++;
    prof_count++;
    prof_report();
    return true;
}



//
// backend
//

struct ggml_backend_axcl_context {
    int32_t device;
};

static const char * ggml_backend_axcl_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_AXCL_NAME;
}

static void ggml_backend_axcl_free(ggml_backend_t backend) {
    ggml_backend_axcl_context * ctx = (ggml_backend_axcl_context *) backend->context;
    delete ctx;
    delete backend;
}

// stride-aware element accessor: flat index -> byte offset
static size_t axcl_off(const struct ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

static uint64_t prof_hostops = 0;

// total byte offset of outer row r (r enumerates ne1*ne2*ne3 combos) - the
// naive r*nb[1] indexing is wrong for 3D/4D tensors
static size_t axcl_row_off(const struct ggml_tensor * t, int64_t r) {
    const int64_t n1 = t->ne[1], n2 = t->ne[2];
    int64_t i1 = r % n1;
    int64_t i2 = (r / n1) % n2;
    int64_t i3 = r / (n1 * n2);
    return (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3];
}

// host-side compute for the fusion ops; tensors live in our host buffers.
// returns false when the op is not ours
static bool ggml_axcl_host_op(struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    float *       dst  = (float *) ((char *) node->data);
    const int64_t ne0  = node->ne[0];
    const int64_t nr   = ggml_nrows(node); // rows of ne0

    fprintf(stderr, "[axcl-hop] %s ne=(%lld,%lld,%lld,%lld) src0=(%lld,%lld,%lld,%lld)\n",
            ggml_op_name(node->op), (long long) node->ne[0], (long long) node->ne[1],
            (long long) node->ne[2], (long long) node->ne[3],
            (long long) src0->ne[0], (long long) src0->ne[1],
            (long long) src0->ne[2], (long long) src0->ne[3]);
    switch (node->op) {
        case GGML_OP_RMS_NORM: {
            float eps;
            memcpy(&eps, node->op_params, sizeof(eps));
            // src1 (weight) is optional in this ggml version: the weight
            // multiply may arrive as a separate MUL op
            for (int64_t r = 0; r < nr; r++) {
                const float * xr = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                double ss = 0.0;
                for (int64_t i = 0; i < ne0; i++) ss += (double) xr[i] * xr[i];
                float rms = (float) (1.0 / sqrt(ss / ne0 + eps));
                if (src1 != nullptr) {
                    const float * w = (const float *) ((const char *) src1->data + axcl_row_off(src1, r));
                    for (int64_t i = 0; i < ne0; i++) dr[i] = xr[i] * rms * w[i];
                } else {
                    for (int64_t i = 0; i < ne0; i++) dr[i] = xr[i] * rms;
                }
            }
            (void) dst;
            break;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const bool mul = (node->op == GGML_OP_MUL);
            // scalar broadcast or same-shape (the only shapes in our splits)
            const int64_t ne = ggml_nbytes(node) / 4; // f32 contiguous fast path
            const bool same = src0->nb[0] == 4 && src1->nb[0] == 4 &&
                              (src1->ne[0] * ((src1->ne[1] > 1 ? src1->ne[1] : 1)) == 1 || ne0 == src1->ne[0]);
            if (same && src0->nb[1] == (size_t) ne0 * 4 && node->nb[1] == (size_t) ne0 * 4 &&
                (src1->ne[0] * src1->ne[1] == 1 || src1->nb[1] == (size_t) src1->ne[0] * 4)) {
                const float * a = (const float *) src0->data;
                const float * b = (const float *) src1->data;
                const bool scalar = (src1->ne[0] * src1->ne[1] == 1);
                for (int64_t i = 0; i < ne; i++) {
                    dst[i] = mul ? a[i] * (scalar ? b[0] : b[i]) : a[i] + (scalar ? b[0] : b[i]);
                }
            } else {
                for (int64_t r = 0; r < nr; r++) {
                    const float * a = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                    const float * b = (const float *) ((char *) src1->data + axcl_row_off(src1, r % (src1->ne[1] * src1->ne[2] * src1->ne[3])));
                    float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                    for (int64_t i = 0; i < ne0; i++) {
                        float bv = (src1->ne[0] == 1) ? b[0] : b[i];
                        dr[i] = mul ? a[i] * bv : a[i] + bv;
                    }
                }
            }
            break;
        }
        case GGML_OP_GLU: {
            // fused activation gate: out = act(x[:h]) * x[h:], h = ne0/2
            int glu;
            memcpy(&glu, node->op_params, sizeof(glu));
            const int64_t h = ne0 / 2;
            for (int64_t r = 0; r < nr; r++) {
                const float * x  = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                for (int64_t i = 0; i < h; i++) {
                    float a = x[i], b = x[h + i], act;
                    switch ((enum ggml_glu_op) glu) {
                        case GGML_GLU_OP_REGLU:     act = a > 0 ? a : 0; break;
                        case GGML_GLU_OP_GEGLU:     act = a * (1.0f / (1.0f + expf(-a))); break;
                        case GGML_GLU_OP_SWIGLU_OAI: { float s = 1.0f + expf(-a); act = (a / s) * (a / s); } break;
                        default:                    act = a / (1.0f + expf(-a)); break;
                    }
                    dr[i] = act * b;
                }
            }
            break;
        }
        case GGML_OP_SOFT_MAX: {
            float params[2] = {1.0f, 0.0f};
            memcpy(params, node->op_params, sizeof(params));
            const float scale = params[0];
            for (int64_t r = 0; r < nr; r++) {
                const float * x = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                float mx = -INFINITY;
                for (int64_t i = 0; i < ne0; i++) mx = fmaxf(mx, x[i] * scale);
                float sum = 0.0f;
                for (int64_t i = 0; i < ne0; i++) { dr[i] = expf(x[i] * scale - mx); sum += dr[i]; }
                float inv = 1.0f / sum;
                for (int64_t i = 0; i < ne0; i++) dr[i] *= inv;
            }
            break;
        }
        case GGML_OP_SCALE: {
            float v;
            memcpy(&v, node->op_params, sizeof(v));
            for (int64_t r = 0; r < nr; r++) {
                const float * x = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                for (int64_t i = 0; i < ne0; i++) dr[i] = x[i] * v;
            }
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_DUP: {
            // same-layout f32 copy (fast path contiguous)
            const int64_t ne = ggml_nbytes(node) / 4;
            if (src0->nb[0] == 4 && node->nb[0] == 4 &&
                (nr <= 1 || (src0->nb[1] == (size_t) ne0 * 4 && node->nb[1] == (size_t) ne0 * 4))) {
                memcpy(node->data, src0->data, (size_t) ne * 4);
            } else {
                for (int64_t r = 0; r < nr; r++) {
                    const float * x = (const float *) ((char *) src0->data + (size_t) r * src0->nb[1]);
                    float *       dr = (float *) ((char *) node->data + (size_t) r * node->nb[1]);
                    for (int64_t i = 0; i < ne0; i++) dr[i] = x[i];
                }
            }
            break;
        }
        case GGML_OP_GET_ROWS: {
            // out[r, :] = rows(src0)[ src1[r] ] (dequant if needed)
            const int64_t nrq = node->ne[1];
            const int64_t nc  = src0->ne[0];
            const struct ggml_type_traits * tr = ggml_get_type_traits(src0->type);
            std::vector<float> rowbuf(nc);
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
                src0->type != GGML_TYPE_BF16) {
                GGML_ASSERT(tr && tr->to_float);
            }
            for (int64_t r = 0; r < nrq; r++) {
                int64_t id;
                if (src1->type == GGML_TYPE_I32) {
                    id = (int64_t) ((const int32_t *) ((const char *) src1->data + (size_t) r * src1->nb[1]))[0];
                } else {
                    id = ((const int64_t *) ((const char *) src1->data + (size_t) r * src1->nb[1]))[0];
                }
                // views index beyond the view's row count into the parent
                // (KV cache): address via nb[1] like the CPU kernel, no
                // bounds check on ne[1]
                const void * srcrow = (const char *) src0->data + (size_t) id * src0->nb[1];
                float * drow = (float *) ((char *) node->data + (size_t) r * node->nb[1]);
                if (src0->type == GGML_TYPE_F32) {
                    memcpy(drow, srcrow, (size_t) nc * 4);
                } else if (src0->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * h = (const ggml_fp16_t *) srcrow;
                    for (int64_t i = 0; i < nc; i++) drow[i] = GGML_COMPUTE_FP16_TO_FP32(h[i]);
                } else if (src0->type == GGML_TYPE_BF16) {
                    const uint16_t * h = (const uint16_t *) srcrow;
                    for (int64_t i = 0; i < nc; i++) {
                        uint32_t u = (uint32_t) h[i] << 16;
                        memcpy(&drow[i], &u, 4);
                    }
                } else {
                    tr->to_float(srcrow, rowbuf.data(), nc);
                    memcpy(drow, rowbuf.data(), (size_t) nc * 4);
                }
            }
            break;
        }
        default:
            return false;
    }
    prof_hostops++;
    return true;
}

static enum ggml_status ggml_backend_axcl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_axcl_context * ctx = (ggml_backend_axcl_context *) backend->context;
    axclrtSetDevice(axcl_get_device_index(ctx->device));
    if (g_axcl_ctx) {
        axclrtSetCurrentContext(g_axcl_ctx); // contexts are thread-local: bind worker threads
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue; // metadata-only: data pointer already correct
        }
        if (ggml_axcl_host_op(node)) {
            continue; // fused host-side: no backend boundary
        }
        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * src0 = node->src[0];
            axcl_matmul *        mm   = axcl_matmul_get(src0->ne[0], src0->ne[1]);
            if (mm == nullptr) {
                GGML_LOG_ERROR("ggml-axcl: node %d MUL_MAT k=%lld n=%lld has no engine\n", i,
                               (long long) src0->ne[0], (long long) src0->ne[1]);
                return GGML_STATUS_ABORTED;
            }
            if (!ggml_axcl_compute_mul_mat(mm, src0, node->src[1], node)) {
                GGML_LOG_ERROR("ggml-axcl: node %d MUL_MAT failed\n", i);
                return GGML_STATUS_ABORTED;
            }
        } else {
            fprintf(stderr, "[axcl-dbg] node %d op %s NOT SUPPORTED\n", i, ggml_op_name(node->op));
            GGML_LOG_ERROR("ggml-axcl: node %d op %s not supported\n", i, ggml_op_name(node->op));
            return GGML_STATUS_ABORTED;
        }
    }
    fprintf(stderr, "[axcl-dbg] graph done: %d nodes ok\n", cgraph->n_nodes);
    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_axcl_interface = {
    /* .get_name           = */ ggml_backend_axcl_name,
    /* .free               = */ ggml_backend_axcl_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .set_tensor_2d_async= */ NULL,
    /* .get_tensor_2d_async= */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_axcl_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
    /* .graph_optimize     = */ NULL,
};

ggml_backend_t ggml_backend_axcl_init(int32_t device) {
    fprintf(stderr, "[axcl-pool] backend_init entry\n");
    if (device < 0 || device >= axcl_get_device_count()) {
        GGML_LOG_ERROR("ggml-axcl: invalid device %d\n", device);
        return nullptr;
    }
    if (axcl_engine_global_init()) {
        axcl_preload_all_engines(); // outside the activation mutex
        axcl_weight_pool_init();
    }
    // translate ordinal to the real slot index via the activation probe
    int32_t slot = axcl_get_device_index(device);
    if (axclrtSetDevice(slot) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: axclrtSetDevice(%d) failed\n", slot);
        return nullptr;
    }

    ggml_backend_axcl_context * ctx = new ggml_backend_axcl_context{device};

    return new ggml_backend{
        /* .guid      = */ ggml_backend_axcl_guid(),
        /* .iface     = */ ggml_backend_axcl_interface,
        /* .device    = */ ggml_backend_axcl_reg_dev(device),
        /* .context   = */ ctx,
    };
}

bool ggml_backend_is_axcl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_axcl_guid());
}

//
// device
//

struct ggml_backend_axcl_device_context {
    int32_t     device;
    std::string name;
    std::string description;
    std::string device_id; // pci bus id, e.g. "0001:04:00.0"
};

static const char * ggml_backend_axcl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_axcl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_axcl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    axclrtDeviceProperties props;
    memset(&props, 0, sizeof(props));
    if (axclrtGetDeviceProperties(axcl_get_device_index(ctx->device), &props) == AXCL_SUCC) {
        *total = (size_t) props.totalCmmSize * 1024;
        *free  = (size_t) props.freeCmmSize * 1024;
    } else {
        *total = *free = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_axcl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_axcl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_axcl_device_get_name(dev);
    props->description = ggml_backend_axcl_device_get_description(dev);
    props->type        = ggml_backend_axcl_device_get_type(dev);
    ggml_backend_axcl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = nullptr;

    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    axclrtDeviceProperties             axcl_props;
    memset(&axcl_props, 0, sizeof(axcl_props));
    if (axclrtGetDeviceProperties(axcl_get_device_index(ctx->device), &axcl_props) == AXCL_SUCC) {
        props->device_id = ctx->device_id.c_str();
    }

    props->caps.async                = false;
    props->caps.host_buffer          = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = false;
    props->caps.mmap_support         = false;
}

static ggml_backend_t ggml_backend_axcl_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    GGML_UNUSED(params);
    return ggml_backend_axcl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_axcl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_axcl_device_context * ctx = (ggml_backend_axcl_device_context *) dev->context;
    return ggml_backend_axcl_buffer_type(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_axcl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr,
                                                                           size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_axcl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // view-class ops are metadata-only (no data movement): the scheduler
    // places them in our splits, so accept and skip them at compute time
    switch (op->op) {
        case GGML_OP_NONE: {
            // accept weight leaves ONLY when they already live in our buffer
            // (accepting globally re-routes model weight placement and
            // blows up the ctx-tensor allocator)
            return op->buffer != nullptr &&
                   strcmp(ggml_backend_buffer_name(op->buffer), "AXCL") == 0;
        }
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }
    // cheap host-side ops we compute ourselves: fusing these into our
    // splits collapses the per-layer CPU<->NPU boundary count
    switch (op->op) {
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_GLU: // silu expressed as GLU in this ggml
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SCALE:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            // embedding lookup from weights living in our buffer; without
            // accepting it the scheduler aborts (no backend pairs our buft
            // with this op)
            return true;
        default:
            break;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    // one-line rejection telemetry per unique (k,n,m) triple
    {
        static std::mutex m;
        static std::unordered_map<uint64_t, bool> seen;
        uint64_t key = ((uint64_t) src0->ne[0] << 44) ^ ((uint64_t) src0->ne[1] << 12) ^ src1->ne[1];
        std::lock_guard<std::mutex> l(m);
        if (!seen.count(key)) {
            seen[key] = true;
            bool engine = axcl_matmul_get(src0->ne[0], src0->ne[1]) != nullptr;
            fprintf(stderr, "[axcl-sched] MUL_MAT k=%lld n=%lld m=%lld type=%s engine=%d -> %s\n",
                    (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[1],
                    ggml_type_name(src0->type), (int) engine,
                    (engine && src1->ne[1] == 1) ? "ACCEPT" : "REJECT");
        }
    }
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    // any weight type is fine - the dequant path uses ggml type traits
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        (ggml_get_type_traits(src0->type) == nullptr || ggml_get_type_traits(src0->type)->to_float == nullptr)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // src0 must be a true 2D weight; src1 [K,1] counts as 1-D to ggml_n_dims,
    // so check the higher dims explicitly instead
    if (ggml_n_dims(src0) != 2 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->ne[1] != 1) {
        return false; // decode only (M == 1); prefill stays on CPU
    }
    // engine must exist for this (K, N)
    return axcl_matmul_get(src0->ne[0], src0->ne[1]) != nullptr;
}

static bool ggml_backend_axcl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    GGML_UNUSED(buft);
    // our compute path reads host memory directly (memcpy from t->data), so
    // CPU-resident weights are fine - without this the scheduler never
    // routes MUL_MAT to us when the model lives in host RAM
    return true;
}

static bool ggml_backend_axcl_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // run MUL_MAT on the NPU even though the weights live in CPU buffers:
    // the compute path stages X/W from host memory per call
    return ggml_backend_axcl_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_axcl_device_interface = {
    /* .get_name             = */ ggml_backend_axcl_device_get_name,
    /* .get_description      = */ ggml_backend_axcl_device_get_description,
    /* .get_memory           = */ ggml_backend_axcl_device_get_memory,
    /* .get_type             = */ ggml_backend_axcl_device_get_type,
    /* .get_props            = */ ggml_backend_axcl_device_get_props,
    /* .init_backend         = */ ggml_backend_axcl_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_axcl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_axcl_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_axcl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_axcl_device_supports_buft,
    /* .offload_op           = */ ggml_backend_axcl_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// registry
//

struct ggml_backend_axcl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_axcl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_AXCL_NAME;
}

static size_t ggml_backend_axcl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_axcl_reg_context * ctx = (ggml_backend_axcl_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_axcl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_axcl_reg_context * ctx = (ggml_backend_axcl_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_axcl_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_axcl_reg_interface = {
    /* .get_name          = */ ggml_backend_axcl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_axcl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_axcl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_axcl_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_axcl_reg(void) {
    static ggml_backend_reg reg;
    static bool             initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_axcl_reg_context * ctx = new ggml_backend_axcl_reg_context;
            for (int32_t i = 0; i < axcl_get_device_count(); i++) {
                auto * dev_ctx = new ggml_backend_axcl_device_context();
                dev_ctx->device = i;
                dev_ctx->name   = GGML_AXCL_NAME + std::to_string(i);
                dev_ctx->description = axcl_get_device_description(i);

                axclrtDeviceProperties props;
                memset(&props, 0, sizeof(props));
                if (axclrtGetDeviceProperties(axcl_get_device_index(i), &props) == AXCL_SUCC) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%04x:%02x:%02x.0", props.pciDomain, props.pciBusID,
                             props.pciDeviceID);
                    dev_ctx->device_id = buf;
                }

                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_axcl_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx,
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_axcl_reg_interface,
                /* .context     = */ ctx,
            };
        }

        initialized = true;
    }

    return &reg;
}

static ggml_backend_dev_t ggml_backend_axcl_reg_dev(int32_t device) {
    return ggml_backend_reg_dev_get(ggml_backend_axcl_reg(), device);
}
