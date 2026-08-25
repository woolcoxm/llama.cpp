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
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <unordered_set>
#include <chrono>
#include <mutex>
#include <atomic>
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
    void *  ptr;
    size_t  size;
    bool    is_cmm = false;  // true when ptr is card memory (axclrtMalloc)
};

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
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    char * dst = (char *) tensor->data + offset;
    if (ctx && ctx->is_cmm) {
        axclrtMemcpy(dst, data, size, AXCL_MEMCPY_HOST_TO_DEVICE);
    } else {
        memcpy(dst, data, size);
    }
}

static void ggml_backend_axcl_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    const char * src = (const char *) tensor->data + offset;
    if (ctx && ctx->is_cmm) {
        axclrtMemcpy(data, src, size, AXCL_MEMCPY_DEVICE_TO_HOST);
    } else {
        memcpy(data, src, size);
    }
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

static bool ggml_backend_axcl_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true; // host RAM: CPU and our backend both access tensors directly
}

static const struct ggml_backend_buffer_type_i axcl_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_axcl_buffer_type_name,
    /* .alloc_buffer   = */ ggml_backend_axcl_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_axcl_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_axcl_buffer_type_get_max_size,
    /* .get_alloc_size = */ NULL,
    /* .is_host        = */ ggml_backend_axcl_buffer_type_is_host,
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
    // staging used through axclrtMemcpy. Staging lives in PINNED host memory
    // (axclrtMallocHost): regular malloc costs ~1ms per small H2D (page
    // pinning per transfer); pinned drops it to tens of microseconds.
    void * dx = nullptr;                 // f32 [K]
    void * dw = nullptr;                 // f32 [K, N] (transposed weights)
    void * dy = nullptr;                 // f32 [N]
    float * x_h = nullptr, * y_h = nullptr; // pinned host staging
    // upload-once weight staging: many weight tensors share one engine shape
    // (every layer's k_proj is the same (k,n)) so device buffers are keyed
    // by the weight tensor's data pointer, not the engine
    std::unordered_map<const void *, void *> dev_w;
    std::vector<float>                    w_h; // scratch for the one upload
    // pre-bound IO per weight set (W + Y bound once; X rebound per call)
    std::unordered_map<const void *, axclrtEngineIO> io_by_w;
};

static void axcl_preload_all_engines();

static axclrtContext g_axcl_ctx = 0; // thread-local bind target for worker threads

// stage profiling: micros accumulated per stage, reported every REPORT computes
#include <cstdint>
static uint64_t prof_stage_wstage = 0, prof_stage_xh2d = 0, prof_stage_bind = 0,
                prof_stage_exec = 0, prof_stage_yd2h = 0, prof_stage_total = 0,
                prof_wstaged = 0, prof_count = 0;
#define PROF_REPORT_EVERY 120
static uint64_t prof_t_attn = 0, prof_t_qkv = 0, prof_t_gateup = 0, prof_t_host = 0, prof_t_small = 0;
static uint64_t prof_n_attn = 0, prof_n_qkv = 0, prof_n_gateup = 0, prof_n_host = 0, prof_n_small = 0;
static inline uint64_t axcl_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void prof_report() {
    if (prof_count % PROF_REPORT_EVERY != 0 || prof_count == 0) return;
    fprintf(stderr,
        "[axcl-prof] computes=%llu wstaged=%llu | avg micros/compute: total=%llu wstage=%llu xh2d=%llu bind=%llu exec=%llu yd2h=%llu other=%llu | attn=%llu/%llu qkv=%llu/%llu gateup=%llu/%llu host=%llu/%llu small=%llu/%llu\n",
        (unsigned long long) prof_count, (unsigned long long) prof_wstaged,
        (unsigned long long) (prof_stage_total / prof_count),
        (unsigned long long) (prof_stage_wstage / prof_count),
        (unsigned long long) (prof_stage_xh2d / prof_count),
        (unsigned long long) (prof_stage_bind / prof_count),
        (unsigned long long) (prof_stage_exec / prof_count),
        (unsigned long long) (prof_stage_yd2h / prof_count),
        (unsigned long long) ((prof_stage_total - prof_stage_wstage - prof_stage_xh2d -
                               prof_stage_bind - prof_stage_exec - prof_stage_yd2h) / prof_count),
        (unsigned long long) prof_t_attn, (unsigned long long) prof_n_attn,
        (unsigned long long) prof_t_qkv, (unsigned long long) prof_n_qkv,
        (unsigned long long) prof_t_gateup, (unsigned long long) prof_n_gateup,
        (unsigned long long) prof_t_host, (unsigned long long) prof_n_host,
        (unsigned long long) prof_t_small, (unsigned long long) prof_n_small);
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

// generic multi-input/multi-output fused engine with weight staging
struct axcl_fused_engine {
    uint64_t model = 0, ectx = 0;
    axclrtEngineIOInfo info = nullptr;
    axclrtEngineIO     io   = nullptr;
    std::vector<void *> dev_in;    // device buffers for each input
    std::vector<void *> dev_out;   // device buffers for each output
    std::vector<int>    in_idx;    // input indices by name order
    std::vector<int>    out_idx;
    // weight staging: maps ggml weight tensor ptr -> device buffer
    std::unordered_map<const void *, void *> staged_w;
    std::vector<float> scratch;     // dequant scratch
    // Pre-bound IO handles: rebinding weight addresses on every execute
    // costs ~3ms/call in descriptor re-setup (isolated exec with fixed
    // bindings runs 4x faster). One IO per weight set, created on first
    // use, rotated thereafter — only the activation input is rebound.
    std::unordered_map<const void *, axclrtEngineIO> io_by_w0;
};

static bool axcl_fused_load(axcl_fused_engine * fe, const char * path,
                            const std::vector<const char *> & in_names,
                            const std::vector<const char *> & out_names) {
    FILE * f = fopen(path, "r");
    if (!f) return false;
    fclose(f);
    if (axclrtEngineLoadFromFile(path, &fe->model) != AXCL_SUCC) return false;
    axclrtEngineGetIOInfo(fe->model, &fe->info);
    axclrtEngineCreateIO(fe->info, &fe->io);
    axclrtEngineCreateContext(fe->model, &fe->ectx);
    fe->in_idx.clear(); fe->out_idx.clear();
    for (auto n : in_names)  fe->in_idx.push_back(axclrtEngineGetInputIndexByName(fe->info, n));
    for (auto n : out_names) fe->out_idx.push_back(axclrtEngineGetOutputIndexByName(fe->info, n));
    return true;
}

// allocate device buffers for all IO (sizes from tensor dims via a size query)
static void axcl_fused_alloc(axcl_fused_engine * fe,
                             const std::vector<size_t> & in_sizes,
                             const std::vector<size_t> & out_sizes) {
    fe->dev_in.resize(in_sizes.size());
    fe->dev_out.resize(out_sizes.size());
    for (size_t i = 0; i < in_sizes.size(); i++)
        axclrtMalloc(&fe->dev_in[i], in_sizes[i], AXCL_MEM_MALLOC_HUGE_FIRST);
    for (size_t i = 0; i < out_sizes.size(); i++)
        axclrtMalloc(&fe->dev_out[i], out_sizes[i], AXCL_MEM_MALLOC_HUGE_FIRST);
}

// stage a weight tensor once: dequant to f32 AND transpose to the engine's
// [k, n] layout (the raw ggml layout is row-per-output [n, k] — uploading it
// untransposed silently scrambles the projection)
static void axcl_dequant_any_to_f32_transposed(const struct ggml_tensor * t, float * w32);
static void * axcl_fused_stage_w(axcl_fused_engine * fe, const struct ggml_tensor * w, size_t bytes) {
    auto it = fe->staged_w.find(w->data);
    if (it != fe->staged_w.end()) return it->second;
    void * dev = nullptr;
    axclrtMalloc(&dev, bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    if (dev == nullptr) return nullptr;
    fe->scratch.resize(bytes / 4);
    axcl_dequant_any_to_f32_transposed(w, fe->scratch.data());
    axclrtMemcpy(dev, fe->scratch.data(), bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    fe->staged_w[w->data] = dev;
    return dev;
}

// fused attention engine: softmax(q*K^T*scale + mask)*V for all heads
// in one NPU execute (mixed precision, GQA via host-side head repeat)
struct axcl_attn_engine {
    uint64_t model = 0, ectx = 0;
    axclrtEngineIOInfo info = nullptr;
    axclrtEngineIO     io   = nullptr;
    void * dq = nullptr, * dm = nullptr, * dout = nullptr;
    int iq = -1, ik = -1, iv = -1, im = -1, iout = -1;
    int h_q = 16, h_kv = 8, d = 128, t = 32;
    std::vector<float> q_buf, k_buf, v_buf, m_buf, out_buf; // host staging
    // device-resident KV cache: one buffer pair per layer, keyed by the host
    // cache pointers. wm = tokens already on the device; decode is append-only
    // per layer, so each step ships only the new token's slice instead of the
    // whole zero-padded [HQ, T, D] tensor (~8 MB/layer at T=512).
    struct kv_slot {
        const void * kptr = nullptr, * vptr = nullptr;
        void * dk = nullptr, * dv = nullptr;
        int wm = 0;
    };
    kv_slot kv[128];
    int kv_n = 0;
};
static axcl_attn_engine g_attn;
static axcl_fused_engine g_qkv;   // rms_norm + q/k/v projections
static axcl_fused_engine g_gate_up; // gate + up projections

static void axcl_attn_load() {
    if (g_attn.model != 0) return;
    const char * env = getenv("AXCL_ATTN_MODEL");
    const char * path = env ? env : "/usr/local/share/ggml-axcl/attn_h16_d128_t32.axmodel";
    FILE * f = fopen(path, "r");
    if (!f) return;
    fclose(f);
    // engine context length from the filename (..._t512.axmodel); default 32
    for (const char * p = strstr(path, "_t"); p != nullptr; p = strstr(p + 1, "_t")) {
        if (p[2] >= '0' && p[2] <= '9') {
            const int tv = atoi(p + 2);
            if (tv > 0) g_attn.t = tv;
            break;
        }
    }
    if (axclrtEngineLoadFromFile(path, &g_attn.model) != AXCL_SUCC) {
        g_attn.model = 0;
        return;
    }
    fprintf(stderr, "ATTN_ENGINE_LOADED model=%llx t=%d\n", (unsigned long long)g_attn.model, g_attn.t);
    axclrtEngineGetIOInfo(g_attn.model, &g_attn.info);
    axclrtEngineCreateIO(g_attn.info, &g_attn.io);
    axclrtEngineCreateContext(g_attn.model, &g_attn.ectx);
    g_attn.iq   = axclrtEngineGetInputIndexByName(g_attn.info, "Q");
    g_attn.ik   = axclrtEngineGetInputIndexByName(g_attn.info, "K");
    g_attn.iv   = axclrtEngineGetInputIndexByName(g_attn.info, "V");
    g_attn.im   = axclrtEngineGetInputIndexByName(g_attn.info, "mask");
    g_attn.iout = axclrtEngineGetOutputIndexByName(g_attn.info, "out");

    const int HQ = g_attn.h_q, D = g_attn.d, T = g_attn.t;
    axclrtMalloc(&g_attn.dq,   HQ * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_attn.dm,   T * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_attn.dout, HQ * D * 4, AXCL_MEM_MALLOC_HUGE_FIRST);
    g_attn.q_buf.resize(HQ * D);
    g_attn.k_buf.resize((size_t) HQ * T * D);
    g_attn.v_buf.resize((size_t) HQ * T * D);
    g_attn.m_buf.resize(T);
    g_attn.out_buf.resize(HQ * D);
    GGML_LOG_INFO("ggml-axcl: attention engine loaded (%s)\n", path);
}

// fetch D consecutive elements of a cache row in the cache's element type
static inline void axcl_kv_fetch(const void * row, ggml_type ty, float * dst, int d_elems) {
    if (ty == GGML_TYPE_F32) {
        memcpy(dst, row, (size_t) d_elems * 4);
    } else if (ty == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) row;
        for (int i = 0; i < d_elems; i++) dst[i] = GGML_COMPUTE_FP16_TO_FP32(h[i]);
    } else if (ty == GGML_TYPE_BF16) {
        const uint16_t * h = (const uint16_t *) row;
        for (int i = 0; i < d_elems; i++) {
            uint32_t u = (uint32_t) h[i] << 16;
            memcpy(&dst[i], &u, 4);
        }
    }
}

// incremental KV residency: append-only upload per layer, returns the layer's
// device K/V buffers. A rewind (seq < wm, e.g. context shift) forces a full
// re-upload. Assumes single-sequence decode; multi-slot serving would need
// per-sequence watermarks.
static bool axcl_attn_sync_kv(const void * k_base, const void * v_base,
                              size_t k_nb1, size_t v_nb1, ggml_type k_ty, ggml_type v_ty,
                              int seq, int n_kv_heads,
                              void ** dk, void ** dv) {
    const int HQ = g_attn.h_q, D = g_attn.d, T = g_attn.t;
    const int G = HQ / n_kv_heads;
    const size_t k_es = ggml_type_size(k_ty), v_es = ggml_type_size(v_ty);
    int s = -1;
    for (int i = 0; i < g_attn.kv_n; i++) {
        if (g_attn.kv[i].kptr == k_base && g_attn.kv[i].vptr == v_base) { s = i; break; }
    }
    if (s < 0) {
        if (g_attn.kv_n >= (int) (sizeof(g_attn.kv) / sizeof(g_attn.kv[0]))) return false;
        s = g_attn.kv_n++;
        g_attn.kv[s].kptr = k_base;
        g_attn.kv[s].vptr = v_base;
        const size_t bytes = (size_t) HQ * T * D * 4;
        if (axclrtMalloc(&g_attn.kv[s].dk, bytes, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&g_attn.kv[s].dv, bytes, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: KV residency alloc failed (%zu MB/side)\n", bytes >> 20);
            g_attn.kv_n--;
            return false;
        }
        // zero-fill once so the padded tail [seq, T) never contains stale data
        memset(g_attn.k_buf.data(), 0, g_attn.k_buf.size() * 4);
        memset(g_attn.v_buf.data(), 0, g_attn.v_buf.size() * 4);
        axclrtMemcpy(g_attn.kv[s].dk, g_attn.k_buf.data(), bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(g_attn.kv[s].dv, g_attn.v_buf.data(), bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    }
    int wm = g_attn.kv[s].wm;
    if (seq < wm) wm = 0;
    if (seq > wm) {
        for (int h = 0; h < HQ; h++) {
            const int hk = h / G;
            float * kd = &g_attn.k_buf[((size_t) h * T + wm) * D];
            float * vd = &g_attn.v_buf[((size_t) h * T + wm) * D];
            for (int t = wm; t < seq; t++) {
                axcl_kv_fetch((const char *) k_base + (size_t)(hk * D) * k_es + (size_t) t * k_nb1, k_ty, kd, D);
                axcl_kv_fetch((const char *) v_base + (size_t)(hk * D) * v_es + (size_t) t * v_nb1, v_ty, vd, D);
                kd += D; vd += D;
            }
        }
        const size_t bytes = (size_t)(seq - wm) * D * 4;
        for (int h = 0; h < HQ; h++) {
            axclrtMemcpy((char *) g_attn.kv[s].dk + ((size_t) h * T + wm) * D * 4,
                         &g_attn.k_buf[((size_t) h * T + wm) * D], bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtMemcpy((char *) g_attn.kv[s].dv + ((size_t) h * T + wm) * D * 4,
                         &g_attn.v_buf[((size_t) h * T + wm) * D], bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
        }
        g_attn.kv[s].wm = seq;
    }
    *dk = g_attn.kv[s].dk;
    *dv = g_attn.kv[s].dv;
    return true;
}

// valid token count for the layer owning this K cache view
static int axcl_attn_kv_wm(const void * k_base) {
    for (int i = 0; i < g_attn.kv_n; i++) {
        if (g_attn.kv[i].kptr == k_base) return g_attn.kv[i].wm;
    }
    return -1;
}

// run one fused attention call: incremental KV upload + execute + download
static bool axcl_attn_run(const float * q, const void * k_cache, const void * v_cache,
                          size_t k_nb1, size_t v_nb1, ggml_type k_ty, ggml_type v_ty,
                          int seq, int n_kv_heads, int head_dim,
                          float * out) {
    if (g_attn.model == 0) return false;
    const int HQ = g_attn.h_q, D = g_attn.d, T = g_attn.t;
    if ((int) head_dim != D || seq > T) return false;

    void * dk = nullptr, * dv = nullptr;
    if (!axcl_attn_sync_kv(k_cache, v_cache, k_nb1, v_nb1, k_ty, v_ty, seq, n_kv_heads, &dk, &dv)) return false;

    // repack Q: [n_embd] -> [HQ, D] (identity reshape)
    memcpy(g_attn.q_buf.data(), q, (size_t) HQ * D * 4);

    // mask: 0 for valid, -1e9 beyond
    for (int t = 0; t < T; t++) g_attn.m_buf[t] = (t < seq) ? 0.0f : -1e9f;

    axclrtMemcpy(g_attn.dq, g_attn.q_buf.data(), (size_t) HQ * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(g_attn.dm, g_attn.m_buf.data(), T * 4, AXCL_MEMCPY_HOST_TO_DEVICE);

    const size_t kv_bytes = (size_t) HQ * T * D * 4;
    axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.iq, g_attn.dq, (size_t) HQ * D * 4);
    axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.ik, dk, kv_bytes);
    axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.iv, dv, kv_bytes);
    axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.im, g_attn.dm, T * 4);
    axclrtEngineSetOutputBufferByIndex(g_attn.io, g_attn.iout, g_attn.dout, (size_t) HQ * D * 4);

    if (axclrtEngineExecute(g_attn.model, g_attn.ectx, 0, g_attn.io) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: attention engine execute failed\n");
        return false;
    }

    axclrtMemcpy(g_attn.out_buf.data(), g_attn.dout, (size_t) HQ * D * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
    memcpy(out, g_attn.out_buf.data(), (size_t) HQ * D * 4);
    return true;
}

static void axcl_weight_pool_init() {
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
        if (available) {
            // the runtime logs every context/device bind at INFO — silence it
            // (it fires per graph compute and costs real time)
            axclSetLogLevel(3); // warnings and above
            // axcl_rt keeps worker threads alive past main(); without an
            // explicit finalize they throw during exit handling. atexit runs
            // before library teardown, so the runtime is still fully up here.
            atexit([]() {
                axclrtSetCurrentContext(g_axcl_ctx);
                axclrtEngineFinalize();
                axclFinalize();
            });
        }
    }
    return available;
}

static std::string axcl_matmul_model_path(int64_t m, int64_t k, int64_t n) {
    const char * dir = getenv("AXCL_MATMUL_DIR");
    char path[512];
    if (m > 1) {
        // batch engines live in their own directory
        snprintf(path, sizeof(path), "/usr/local/share/ggml-axcl/matmul_m%lld/matmul_m%lld_k%lld_n%lld.axmodel",
                 (long long) m, (long long) m, (long long) k, (long long) n);
        if (access(path, R_OK) == 0) return path;
        return ""; // no batch engine — caller falls back
    }
    snprintf(path, sizeof(path), "%s/matmul_m%lld_k%lld_n%lld.axmodel",
             dir ? dir : "/usr/local/share/ggml-axcl/matmul", (long long) m, (long long) k, (long long) n);
    return path;
}

// the AXCL engine IO is not thread-safe: serialize loads and executes
static std::mutex axcl_exec_mutex;

// device-resident chain: activations flow engine -> engine without host
// round-trips. g_chain_x_override lets compute_mul_mat bind a device buffer
// as X directly (set by the chain runner before dispatching a MUL_MAT).
static void * g_chain_x_override = nullptr;

// tensor data ptr -> device buffer holding its current value. Entries are
// valid only in topological order (engine output buffers get reused); the
// map is cleared whenever a new forward pass starts (GET_ROWS node).
struct axcl_chain {
    axcl_fused_engine norm, add, glu, qkvn, addnorm, gludown;
    bool engines_ok = false;
    std::unordered_map<const void *, void *> dev;
};
static axcl_chain g_chain;
static int g_dbg_qkvn, g_dbg_addnorm, g_dbg_gludown, g_dbg_norm, g_dbg_add, g_dbg_glu;

// load the chain engines (norm/add/glu); returns true when all available
static void axcl_chain_load() {
    if (g_chain.norm.model != 0 || g_chain.engines_ok) return;
    const char * dir = "/usr/local/share/ggml-axcl/chain";
    char p[512];
    snprintf(p, sizeof(p), "%s/rmsnorm_h1024.axmodel", dir);
    bool ok = axcl_fused_load(&g_chain.norm, p, {"x", "w"}, {"y"});
    if (!ok) fprintf(stderr, "[chain] FAIL norm\n");
    snprintf(p, sizeof(p), "%s/add_h1024.axmodel", dir);
    ok = ok && axcl_fused_load(&g_chain.add, p, {"a", "b"}, {"y"});
    snprintf(p, sizeof(p), "%s/glu2_h3072.axmodel", dir);
    ok = ok && axcl_fused_load(&g_chain.glu, p, {"g", "u"}, {"y"});
    if (ok) {
        axcl_fused_alloc(&g_chain.norm, {1024*4, 1024*4}, {1024*4});
        axcl_fused_alloc(&g_chain.add,  {1024*4, 1024*4}, {1024*4});
        axcl_fused_alloc(&g_chain.glu,  {3072*4, 3072*4}, {3072*4});
        g_chain.engines_ok = true;
        GGML_LOG_INFO("ggml-axcl: chain engines loaded (norm/add/glu)\n");
    }
}

// stage a 1-D weight (norm gains etc.): dequant if needed, upload once
static void * axcl_chain_stage_w(axcl_fused_engine * fe, const struct ggml_tensor * w, size_t bytes) {
    auto it = fe->staged_w.find(w->data);
    if (it != fe->staged_w.end()) return it->second;
    void * dev = nullptr;
    axclrtMalloc(&dev, bytes, AXCL_MEM_MALLOC_HUGE_FIRST);
    if (dev == nullptr) return nullptr;
    if (w->type == GGML_TYPE_F32) {
        axclrtMemcpy(dev, w->data, bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    } else {
        fe->scratch.resize(bytes / 4);
        const auto * tr = ggml_get_type_traits(w->type);
        if (tr && tr->to_float) tr->to_float(w->data, fe->scratch.data(), (int64_t)(bytes / 4));
        axclrtMemcpy(dev, fe->scratch.data(), bytes, AXCL_MEMCPY_HOST_TO_DEVICE);
    }
    fe->staged_w[w->data] = dev;
    return dev;
}

// fetch a device buffer for a tensor's current value; nullptr = host only.
// POP semantics: the entry is consumed on read. ggml reuses tensor host
// addresses within a graph, so a lingering entry would hand a later tensor
// another tensor's device buffer — the single-consumer pop keeps the map
// exact, and any second consumer falls back to the host write-back.
static void * axcl_chain_get(const struct ggml_tensor * t) {
    auto it = g_chain.dev.find(t->data);
    if (it == g_chain.dev.end()) return nullptr;
    void * p = it->second;
    g_chain.dev.erase(it);
    return p;
}

// record an engine output for tensor t + write it back to host memory
static void axcl_chain_put(struct ggml_tensor * t, void * dev_out, size_t bytes) {
    g_chain.dev[t->data] = dev_out;
    axclrtMemcpy(t->data, dev_out, bytes, AXCL_MEMCPY_DEVICE_TO_HOST);
}

// run a small chain engine: inputs resolved from the device map when present
// (H2D otherwise), output recorded + written back
static bool axcl_chain_run(axcl_fused_engine * fe, struct ggml_tensor * node,
                           const size_t * in_bytes, size_t n_in, size_t out_bytes,
                           bool record = true) {
    std::lock_guard<std::mutex> lock(axcl_exec_mutex);
    for (size_t j = 0; j < n_in; j++) {
        struct ggml_tensor * src = node->src[j];
        void * devv = axcl_chain_get(src);
        if (devv != nullptr) {
            if (devv == fe->dev_out[0]) {
                // alias hazard: the input IS this engine's previous output
                // buffer (e.g. residual chains) — copy it aside first
                axclrtMemcpy(fe->dev_in[j], devv, in_bytes[j], AXCL_MEMCPY_DEVICE_TO_DEVICE);
                devv = fe->dev_in[j];
            }
            axclrtEngineSetInputBufferByIndex(fe->io, fe->in_idx[j], devv, in_bytes[j]);
        } else {
            axclrtMemcpy(fe->dev_in[j], src->data, in_bytes[j], AXCL_MEMCPY_HOST_TO_DEVICE);
            axclrtEngineSetInputBufferByIndex(fe->io, fe->in_idx[j], fe->dev_in[j], in_bytes[j]);
        }
    }
    axclrtEngineSetOutputBufferByIndex(fe->io, fe->out_idx[0], fe->dev_out[0], out_bytes);
    if (axclrtEngineExecute(fe->model, fe->ectx, 0, fe->io) != AXCL_SUCC) return false;
    axclrtMemcpy(node->data, fe->dev_out[0], out_bytes, AXCL_MEMCPY_DEVICE_TO_HOST);
    if (record) g_chain.dev[node->data] = fe->dev_out[0];
    return true;
}

static axcl_matmul * axcl_matmul_load(int64_t m, int64_t k, int64_t n) {
    if (!axcl_engine_global_init()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> exec_lock(axcl_exec_mutex);

    axcl_matmul * mm = new axcl_matmul();
    mm->m = m;
    mm->k = k;
    mm->n = n;

    std::string path = axcl_matmul_model_path(m, k, n);
    if (path.empty()) { delete mm; return nullptr; }
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

    mm->w_h.resize(k * n);
    if (axclrtMallocHost((void **) &mm->x_h, (size_t) m * k * 4) != AXCL_SUCC ||
        axclrtMallocHost((void **) &mm->y_h, (size_t) m * n * 4) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: pinned host alloc failed for %s\n", path.c_str());
        delete mm;
        return nullptr;
    }
    if (axclrtMalloc(&mm->dx, (size_t) m * k * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMalloc(&mm->dw, (size_t) k * n * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMalloc(&mm->dy, (size_t) m * n * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
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
    // key: (m << 44) | (k << 22) | n  (m small: 1,4,8; k,n < 4M)
    std::unordered_map<uint64_t, axcl_matmul *> engines;
    std::unordered_map<uint64_t, bool>        missing;
    bool                                       preloaded = false;
    static uint64_t make_key(int64_t m, int64_t k, int64_t n) {
        return ((uint64_t) m << 44) | ((uint64_t) k << 22) | (uint32_t) n;
    }
};

static axcl_matmul_cache_t * axcl_matmul_cache() {
    static axcl_matmul_cache_t cache;
    return &cache;
}

static axcl_matmul * axcl_matmul_get(int64_t m, int64_t k, int64_t n) {
    axcl_matmul_cache_t * cache = axcl_matmul_cache();

    const uint64_t key = axcl_matmul_cache_t::make_key(m, k, n);

    std::lock_guard<std::mutex> lock(cache->mutex);
    auto it = cache->engines.find(key);
    if (it != cache->engines.end()) {
        return it->second;
    }
    if (cache->missing.count(key) || (cache->preloaded && m == 1)) {
        return nullptr; // after preload the m=1 set is final; batch engines
                        // load lazily (first prefill/speculative batch)
    }
    axcl_matmul * mm = axcl_matmul_load(m, k, n);
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
    // batch engines (m>1) preload from their own directories
    for (int64_t bm : {4}) {
        char bdir[512];
        snprintf(bdir, sizeof(bdir), "/usr/local/share/ggml-axcl/matmul_m%lld", (long long) bm);
        DIR * bdp = opendir(bdir);
        if (!bdp) continue;
        struct dirent * bde;
        while ((bde = readdir(bdp)) != nullptr) {
            int64_t m, k, n;
            if (sscanf(bde->d_name, "matmul_m%lld_k%lld_n%lld.axmodel", (long long *) &m, (long long *) &k, (long long *) &n) == 3) {
                const uint64_t key = axcl_matmul_cache_t::make_key(m, k, n);
                if (!cache->engines.count(key) && !cache->missing.count(key)) {
                    axcl_matmul * mm = axcl_matmul_load(m, k, n);
                    if (mm) cache->engines[key] = mm; else cache->missing[key] = true;
                }
            }
        }
        closedir(bdp);
    }
    struct dirent * de;
    int loaded = 0;
    while ((de = readdir(dp)) != nullptr) {
        int64_t m, k, n;
        if (sscanf(de->d_name, "matmul_m%lld_k%lld_n%lld.axmodel", (long long *) &m, (long long *) &k, (long long *) &n) == 3) {
            const uint64_t key = axcl_matmul_cache_t::make_key(m, k, n);
            if (cache->engines.count(key) || cache->missing.count(key)) {
                continue;
            }
            axcl_matmul * mm = axcl_matmul_load(m, k, n);
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

    // X: activation, src1 is [K, 1] f32 — may be a strided view (permuted
    // attention tensors), so read element-wise unless contiguous.
    // In chain mode the activation may already be device-resident: bind it
    // directly and skip the host staging + H2D entirely.
    void * dev_x_override = g_chain_x_override;
    g_chain_x_override = nullptr;
    if (dev_x_override != nullptr) {
        // device-resident X: bind directly below, no staging
    } else if (src1->nb[0] == 4 && src1->nb[1] == (size_t) k * 4) {
        memcpy(mm->x_h, src1->data, (size_t) mm->m * k * 4);
    } else {
        // strided X: copy row by row
        for (int64_t row = 0; row < mm->m; row++) {
            const char * xr = (const char *) src1->data + (size_t) row * src1->nb[1];
            for (int64_t kk = 0; kk < k; kk++)
                mm->x_h[row * k + kk] = *(const float *) (xr + (size_t) kk * src1->nb[0]);
        }
    }

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
    if (dev_x_override == nullptr &&
        axclrtMemcpy(mm->dx, mm->x_h, (size_t) mm->m * k * 4, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
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
    void * xbuf = dev_x_override != nullptr ? dev_x_override : mm->dx;
    // pre-bound IO per weight set when staged (W + Y fixed; X rebound)
    axclrtEngineIO io = mm->io;
    if (w_uploaded) {
        axclrtEngineIO & pb = mm->io_by_w[src0->data];
        if (pb == nullptr) {
            axclrtEngineCreateIO(mm->io_info, &pb);
            axclrtEngineSetInputBufferByIndex(pb, mm->w_idx, wbuf, (size_t) k * n * 4);
            axclrtEngineSetOutputBufferByIndex(pb, mm->y_idx, mm->dy, (size_t) n * 4);
        }
        io = pb;
    } else {
        // per-call weight upload path (prefill / pool miss): bind W + Y too
        if (axclrtEngineSetInputBufferByIndex(io, mm->w_idx, wbuf, (size_t) k * n * 4) != AXCL_SUCC ||
            axclrtEngineSetOutputBufferByIndex(io, mm->y_idx, mm->dy, (size_t) n * 4) != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: bind device buffers failed\n");
            return false;
        }
    }
    if (axclrtEngineSetInputBufferByIndex(io, mm->x_idx, xbuf, (size_t) mm->m * k * 4) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: bind device buffers failed\n");
        return false;
    }
    auto t4 = std::chrono::steady_clock::now();

    axclError ex;
    static axclrtStream async_stream = nullptr;
    static const bool use_async = getenv("GGML_AXCL_ASYNC") != nullptr;
    if (use_async) {
        if (async_stream == nullptr) axclrtCreateStream(&async_stream);
        ex = axclrtEngineExecuteAsync(mm->model_id, mm->context_id, 0, io, async_stream);
        if (ex == AXCL_SUCC) axclrtSynchronizeStream(async_stream);
    } else {
        ex = axclrtEngineExecute(mm->model_id, mm->context_id, 0, io);
    }
    auto t5 = std::chrono::steady_clock::now();
    if (ex != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: engine execute failed\n");
        return false;
    }

    if (axclrtMemcpy(mm->y_h, mm->dy, (size_t) mm->m * n * 4, AXCL_MEMCPY_DEVICE_TO_HOST) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: D2H copy failed\n");
        return false;
    }
    if (dst->nb[0] == 4 && dst->nb[1] == (size_t) n * 4) {
        memcpy(dst->data, mm->y_h, (size_t) mm->m * n * 4);
    } else {
        for (int64_t row = 0; row < mm->m; row++) {
            float * dr = (float *) ((char *) dst->data + (size_t) row * dst->nb[1]);
            for (int64_t nn = 0; nn < n; nn++)
                dr[nn] = mm->y_h[row * n + nn];
        }
    }
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
            // exact CPU semantics: per-dim broadcast (i_k % src->ne[k]), f32
            const bool mul = (node->op == GGML_OP_MUL);
            const int64_t n1 = node->ne[1], n2 = node->ne[2] ? node->ne[2] : 1, n3 = node->ne[3] ? node->ne[3] : 1;
            const int64_t a1 = src0->ne[1] ? src0->ne[1] : 1, a2 = src0->ne[2] ? src0->ne[2] : 1, a3 = src0->ne[3] ? src0->ne[3] : 1;
            const int64_t b1 = src1->ne[1] ? src1->ne[1] : 1, b2 = src1->ne[2] ? src1->ne[2] : 1, b3 = src1->ne[3] ? src1->ne[3] : 1;
            const bool bscalar = (src1->ne[0] == 1 && b1 * b2 * b3 == 1);
            for (int64_t j3 = 0; j3 < n3; j3++) {
                for (int64_t j2 = 0; j2 < n2; j2++) {
                    for (int64_t j1 = 0; j1 < n1; j1++) {
                        const float * a = (const float *) ((char *) src0->data +
                            (size_t)(j1 % a1) * src0->nb[1] + (size_t)(j2 % a2) * src0->nb[2] + (size_t)(j3 % a3) * src0->nb[3]);
                        const float * b = (const float *) ((char *) src1->data +
                            (size_t)(j1 % b1) * src1->nb[1] + (size_t)(j2 % b2) * src1->nb[2] + (size_t)(j3 % b3) * src1->nb[3]);
                        float * dr = (float *) ((char *) node->data +
                            (size_t)j1 * node->nb[1] + (size_t)j2 * node->nb[2] + (size_t)j3 * node->nb[3]);
                        if (bscalar) {
                            const float bv = b[0];
                            if (mul) { for (int64_t i = 0; i < ne0; i++) dr[i] = a[i] * bv; }
                            else     { for (int64_t i = 0; i < ne0; i++) dr[i] = a[i] + bv; }
                        } else if (src1->ne[0] == 1) {
                            for (int64_t i = 0; i < ne0; i++) dr[i] = mul ? a[i] * b[0] : a[i] + b[0];
                        } else {
                            if (mul) { for (int64_t i = 0; i < ne0; i++) dr[i] = a[i] * b[i]; }
                            else     { for (int64_t i = 0; i < ne0; i++) dr[i] = a[i] + b[i]; }
                        }
                    }
                }
            }
            break;
        }
        case GGML_OP_GLU: {
            // CPU semantics: two-input form gate=src0, up=src1; single-input
            // form keeps both halves in src0 with the `swapped` op-param
            // deciding which half is the gate
            int32_t gp[2] = {0, 0};
            memcpy(gp, node->op_params, sizeof(gp));
            const int glu = gp[0];
            const bool swapped = gp[1] != 0;
            const bool two_in = src1 != nullptr;
            const int64_t nc = two_in ? src0->ne[0] : src0->ne[0] / 2;
            for (int64_t r = 0; r < nr; r++) {
                const float * a = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                const float * b = nullptr;
                if (two_in) {
                    b = (const float *) ((char *) src1->data + axcl_row_off(src1, r));
                } else {
                    b = a + (swapped ? 0 : nc);
                    a = a + (swapped ? nc : 0);
                }
                float * dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                for (int64_t i = 0; i < nc; i++) {
                    float x = a[i], y = b[i], act;
                    switch ((enum ggml_glu_op) glu) {
                        case GGML_GLU_OP_REGLU:     act = x > 0 ? x : 0; break;
                        case GGML_GLU_OP_GEGLU:     act = x * (1.0f / (1.0f + expf(-x))); break;
                        case GGML_GLU_OP_SWIGLU_OAI: { float s = 1.0f + expf(-x); act = (x / s) * (x / s); } break;
                        default:                    act = x / (1.0f + expf(-x)); break;
                    }
                    dr[i] = act * y;
                }
            }
            break;
        }
        case GGML_OP_SOFT_MAX: {
            float params[2] = {1.0f, 0.0f};
            memcpy(params, node->op_params, sizeof(params));
            const float scale = params[0];
            // optional additive mask (src1): f16/f32, broadcast over rows or
            // one row per outer row — required for causal manual attention
            const struct ggml_tensor * ms = src1;
            const int64_t mrows = ms ? (ms->ne[1] * ms->ne[2] * ms->ne[3]) : 0;
            for (int64_t r = 0; r < nr; r++) {
                const float * x = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                const char * mrow = (ms && mrows > 0)
                    ? (const char *) ms->data + axcl_row_off(ms, mrows > 1 ? (r % mrows) : 0) : nullptr;
                float mx = -INFINITY;
                for (int64_t i = 0; i < ne0; i++) {
                    float v = x[i] * scale;
                    if (mrow) {
                        v += (ms->type == GGML_TYPE_F16)
                            ? GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) mrow)[i])
                            : ((const float *) mrow)[i];
                    }
                    mx = fmaxf(mx, v);
                }
                float sum = 0.0f;
                for (int64_t i = 0; i < ne0; i++) {
                    float v = x[i] * scale;
                    if (mrow) {
                        v += (ms->type == GGML_TYPE_F16)
                            ? GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) mrow)[i])
                            : ((const float *) mrow)[i];
                    }
                    dr[i] = expf(v - mx);
                    sum += dr[i];
                }
                float inv = 1.0f / sum;
                for (int64_t i = 0; i < ne0; i++) dr[i] *= inv;
            }
            break;
        }
        case GGML_OP_SCALE: {
            // op_params: [0]=scale, [1]=bias; dst = src*scale + bias
            float sb[2] = {1.0f, 0.0f};
            memcpy(sb, node->op_params, sizeof(sb));
            for (int64_t r = 0; r < nr; r++) {
                const float * x = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float *       dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                for (int64_t i = 0; i < ne0; i++) dr[i] = x[i] * sb[0] + sb[1];
            }
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            // stride-aware copy with type conversion (CPU dup semantics):
            // rows may be views with arbitrary strides, element sizes differ
            if (!src0 || !src0->data || !node->data) break;
            const ggml_type sty = src0->type, dty = node->type;
            const size_t des = ggml_type_size(dty);
            const bool f32tof32 = (sty == GGML_TYPE_F32 && dty == GGML_TYPE_F32);
            for (int64_t j3 = 0; j3 < node->ne[3]; j3++) {
                for (int64_t j2 = 0; j2 < node->ne[2]; j2++) {
                    for (int64_t j1 = 0; j1 < node->ne[1]; j1++) {
                        const char * s = (const char *) src0->data +
                            (size_t)j1 * src0->nb[1] + (size_t)j2 * src0->nb[2] + (size_t)j3 * src0->nb[3];
                        char * d = (char *) node->data +
                            (size_t)j1 * node->nb[1] + (size_t)j2 * node->nb[2] + (size_t)j3 * node->nb[3];
                        if (sty == dty && src0->nb[0] == des && node->nb[0] == des) {
                            memcpy(d, s, (size_t) ne0 * des);
                        } else if (f32tof32) {
                            for (int64_t i = 0; i < ne0; i++)
                                *(float *)(d + i * 4) = *(const float *)(s + i * src0->nb[0]);
                        } else {
                            float buf[2048];
                            for (int64_t i0 = 0; i0 < ne0; i0 += 2048) {
                                const int64_t n = (ne0 - i0 < 2048) ? (ne0 - i0) : 2048;
                                axcl_kv_fetch(s + (size_t)i0 * src0->nb[0], sty, buf, (int) n);
                                if (dty == GGML_TYPE_F32) {
                                    memcpy(d + (size_t)i0 * 4, buf, (size_t)n * 4);
                                } else if (dty == GGML_TYPE_F16) {
                                    ggml_fp16_t * h = (ggml_fp16_t *)(d + (size_t)i0 * 2);
                                    for (int64_t i = 0; i < n; i++) h[i] = GGML_COMPUTE_FP32_TO_FP16(buf[i]);
                                } else if (dty == GGML_TYPE_BF16) {
                                    uint16_t * h = (uint16_t *)(d + (size_t)i0 * 2);
                                    for (int64_t i = 0; i < n; i++) {
                                        uint32_t u; memcpy(&u, &buf[i], 4);
                                        h[i] = (uint16_t)(u >> 16);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        case GGML_OP_GET_ROWS: {
            // CPU semantics: ids decompose as (i10,i11,i12); the id's outer
            // dims map into BOTH the destination and the source rows
            const int64_t nc  = src0->ne[0];
            const int64_t nrid = ggml_nelements(src1);
            const int64_t ne10 = src1->ne[0] ? src1->ne[0] : 1;
            const int64_t ne11 = src1->ne[1];
            const int64_t ne12 = src1->ne[2];
            const struct ggml_type_traits * tr = ggml_get_type_traits(src0->type);
            std::vector<float> rowbuf(nc > 0 ? nc : 1);
            for (int64_t i = 0; i < nrid; i++) {
                const int64_t i12 = i / (ne11 * ne10);
                const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
                const int64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;
                const char * idp = (const char *) src1->data +
                    (size_t)i10 * src1->nb[0] + (size_t)i11 * src1->nb[1] + (size_t)i12 * src1->nb[2];
                int64_t id = (src1->type == GGML_TYPE_I64)
                    ? *(const int64_t *) idp : (int64_t) *(const int32_t *) idp;
                if (id < 0) id = 0;
                if (id >= src0->ne[1]) id = src0->ne[1] - 1; // stay alive; CPU asserts
                const char * srow = (const char *) src0->data +
                    (size_t)id * src0->nb[1] + (size_t)i11 * src0->nb[2] + (size_t)i12 * src0->nb[3];
                char * drow = (char *) node->data +
                    (size_t)i10 * node->nb[1] + (size_t)i11 * node->nb[2] + (size_t)i12 * node->nb[3];
                if (src0->type == GGML_TYPE_F32) {
                    memcpy(drow, srow, (size_t)nc * 4);
                } else if (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) {
                    axcl_kv_fetch(srow, src0->type, rowbuf.data(), (int)nc);
                    memcpy(drow, rowbuf.data(), (size_t)nc * 4);
                } else {
                    GGML_ASSERT(tr && tr->to_float);
                    tr->to_float(srow, rowbuf.data(), nc);
                    memcpy(drow, rowbuf.data(), (size_t)nc * 4);
                }
            }
            break;
        }
        case GGML_OP_ROPE: {
            // op_params: [0]=n_past, [1]=n_dims, [2]=mode, [3]=n_ctx,
            // [4]=n_ctx_orig, floats from byte 20: freq_base, freq_scale...
            // Layout follows the CPU kernel: ne[1]=heads, ne[2]=tokens, and
            // positions are indexed by the TOKEN dim (pos[i2]), not the flat
            // row. MROPE (mode 8/40) is not supported.
            const int32_t * ip = (const int32_t *) node->op_params;
            const int n_dims = ip[1] >= 2 ? ip[1] : (int) src0->ne[0];
            const int mode   = ip[2];
            float freq_base = 10000.0f, freq_scale = 1.0f;
            memcpy(&freq_base,  (char *) node->op_params + 20, sizeof(float));
            memcpy(&freq_scale, (char *) node->op_params + 24, sizeof(float));
            const bool neo = (mode & 2) != 0; // GGML_ROPE_TYPE_NEOX pairing
            const int64_t hd = src0->ne[0];
            const int64_t n1 = src0->ne[1];
            const int64_t half = n_dims / 2;
            const int64_t nrows = ggml_nrows(node);
            const int32_t * pos = (src1 != nullptr && src1->type == GGML_TYPE_I32)
                                ? (const int32_t *) src1->data : nullptr;
            const float * ffac = (node->src[2] != nullptr && node->src[2]->type == GGML_TYPE_F32)
                                ? (const float *) node->src[2]->data : nullptr;
            for (int64_t r = 0; r < nrows; r++) {
                const float * x = (const float *) ((char *) src0->data + axcl_row_off(src0, r));
                float * dr = (float *) ((char *) node->data + axcl_row_off(node, r));
                const int64_t i2 = (r / n1) % (src0->ne[2] ? src0->ne[2] : 1); // token index
                const float p = pos ? (float) pos[i2] : 0.0f;
                if (neo) {
                    for (int64_t i = 0; i < half; i++) {
                        const float ffs = ffac ? ffac[i] : 1.0f;
                        const float theta = p * freq_scale / ffs * powf(freq_base, (float)(-2.0 * i) / n_dims);
                        const float cv = cosf(theta), sv = sinf(theta);
                        const float x0 = x[i], x1 = x[i + half];
                        dr[i] = x0 * cv - x1 * sv;
                        dr[i + half] = x0 * sv + x1 * cv;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float ffs = ffac ? ffac[i0 / 2] : 1.0f;
                        const float theta = p * freq_scale / ffs * powf(freq_base, (float)(-1.0 * i0) / n_dims);
                        const float cv = cosf(theta), sv = sinf(theta);
                        const float x0 = x[i0], x1 = x[i0 + 1];
                        dr[i0] = x0 * cv - x1 * sv;
                        dr[i0 + 1] = x0 * sv + x1 * cv;
                    }
                }
                for (int64_t i = n_dims; i < hd; i++) dr[i] = x[i]; // partial-dim tail
            }
            break;
        }
        case GGML_OP_SET_ROWS: {
            // CPU semantics: loops (i03, i02, i); ids broadcast with modulo on
            // their outer dims; both source and destination offset by i02/i03
            if (!src0 || !src0->data || !node || !node->data || !src1 || !src1->data) break;
            const int64_t nc = src0->ne[0];
            const ggml_type dt = node->type;
            const size_t des = ggml_type_size(dt);
            const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
            const int64_t ne11 = src1->ne[1] ? src1->ne[1] : 1, ne12 = src1->ne[2] ? src1->ne[2] : 1;
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i = 0; i < src0->ne[1]; i++) {
                        const int64_t i12 = i03 % ne12;
                        const int64_t i11 = i02 % ne11;
                        const char * idp = (const char *) src1->data +
                            (size_t)i * src1->nb[0] + (size_t)i11 * src1->nb[1] + (size_t)i12 * src1->nb[2];
                        int64_t id = (src1->type == GGML_TYPE_I64)
                            ? *(const int64_t *) idp : (int64_t) *(const int32_t *) idp;
                        if (id < 0 || (size_t)((size_t)id * node->nb[1] + (size_t)nc * des) > ggml_nbytes(node)) continue;
                        const char * srow = (const char *) src0->data +
                            (size_t)i * src0->nb[1] + (size_t)i02 * src0->nb[2] + (size_t)i03 * src0->nb[3];
                        char * drow = (char *) node->data +
                            (size_t)id * node->nb[1] + (size_t)i02 * node->nb[2] + (size_t)i03 * node->nb[3];
                        float buf[4096];
                        const float * sr;
                        if (src0->type == GGML_TYPE_F32) {
                            sr = (const float *) srow;
                        } else if (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) {
                            axcl_kv_fetch(srow, src0->type, buf, (int)nc);
                            sr = buf;
                        } else {
                            const auto * trq = ggml_get_type_traits(src0->type);
                            if (!trq || !trq->to_float) continue;
                            trq->to_float(srow, buf, nc);
                            sr = buf;
                        }
                        if (dt == GGML_TYPE_F32) {
                            memcpy(drow, sr, (size_t)nc * 4);
                        } else if (dt == GGML_TYPE_F16) {
                            ggml_fp16_t * h = (ggml_fp16_t *) drow;
                            for (int64_t q = 0; q < nc; q++) h[q] = GGML_COMPUTE_FP32_TO_FP16(sr[q]);
                        } else if (dt == GGML_TYPE_BF16) {
                            uint16_t * h = (uint16_t *) drow;
                            for (int64_t q = 0; q < nc; q++) {
                                uint32_t u; memcpy(&u, &sr[q], 4);
                                h[q] = (uint16_t)(u >> 16);
                            }
                        }
                    }
                }
            }
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // flash attention actual shapes (measured):
            //   Q [D, nq, HQ] token stride nb[1], head stride nb[2]
            //   K [D, seq, HKV] token stride nb[1], head slice = hk*D elems
            //   V [D, seq, HKV] same as K
            // out has Q's shape. The engine computes single-query attention;
            // batched prefill runs one call per token with a causal mask.
            if (g_attn.model == 0 || !src0 || !src1 || !node->src[2]) return false;
            const struct ggml_tensor * qt = src0;
            const struct ggml_tensor * kt = src1;
            const struct ggml_tensor * vt = node->src[2];
            const int HQ = 16, D = 128, HKV = 8;
            if (qt->ne[0] != D || kt->ne[2] != HKV) return false; // wrong head config
            const int nq = (int) qt->ne[1];
            // K view ne[1] is the PADDED cache width, not the valid count.
            // The FA tensor has no direct validity info — but llama.cpp
            // passes the causal mask as src[3]; read its last valid slot.
            // Fallback: sync first, then the watermark tells the live count.
            int seq_total = -1;
            if (node->src[3] != nullptr) {
                const struct ggml_tensor * msk = node->src[3];
                // mask [n_kv] f16: 0 = valid, -inf = masked
                const int mlen = (int) (msk->ne[0] * msk->ne[1] * msk->ne[2] * msk->ne[3]);
                for (int t = mlen - 1; t >= 0; t--) {
                    float v = (msk->type == GGML_TYPE_F16)
                        ? GGML_COMPUTE_FP16_TO_FP32(((const ggml_fp16_t *) msk->data)[t])
                        : ((const float *) msk->data)[t];
                    if (v > -1e8f) { seq_total = t + 1; break; }
                }
            }
            if (seq_total < 0) return false; // no mask — cannot determine validity
            if (seq_total > g_attn.t) return false; // context too long for engine

            void * dk = nullptr, * dv = nullptr;
            if (!axcl_attn_sync_kv(kt->data, vt->data, kt->nb[1], vt->nb[1],
                                   kt->type, vt->type, seq_total, HKV, &dk, &dv)) return false;

            const int base = seq_total - nq; // first cache slot of this ubatch
            static float eq[16 * 128];
            const size_t kv_bytes = (size_t)HQ * g_attn.t * D * 4;
            for (int tq_i = 0; tq_i < nq; tq_i++) {
                const int seq_t = base + tq_i + 1; // causal: token sees [0, base+tq_i]
                for (int h = 0; h < HQ; h++)
                    for (int d = 0; d < D; d++)
                        eq[h * D + d] = *(const float *)((const char *)qt->data +
                            (size_t)tq_i * qt->nb[1] + (size_t)d * qt->nb[0] + (size_t)h * qt->nb[2]);
                for (int t = 0; t < g_attn.t; t++) g_attn.m_buf[t] = (t < seq_t) ? 0.0f : -1e9f;
                axclrtMemcpy(g_attn.dq, eq, (size_t)HQ * D * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtMemcpy(g_attn.dm, g_attn.m_buf.data(), g_attn.t * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.iq, g_attn.dq, (size_t)HQ * D * 4);
                axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.ik, dk, kv_bytes);
                axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.iv, dv, kv_bytes);
                axclrtEngineSetInputBufferByIndex(g_attn.io, g_attn.im, g_attn.dm, g_attn.t * 4);
                axclrtEngineSetOutputBufferByIndex(g_attn.io, g_attn.iout, g_attn.dout, (size_t)HQ * D * 4);
                if (axclrtEngineExecute(g_attn.model, g_attn.ectx, 0, g_attn.io) != AXCL_SUCC) {
                    GGML_LOG_ERROR("ggml-axcl: attention engine execute failed\n");
                    return false;
                }
                axclrtMemcpy(g_attn.out_buf.data(), g_attn.dout, (size_t)HQ * D * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
                for (int h = 0; h < HQ; h++)
                    for (int d = 0; d < D; d++)
                        *(float *)((char *)node->data + (size_t)tq_i * node->nb[1] +
                            (size_t)d * node->nb[0] + (size_t)h * node->nb[2]) =
                            g_attn.out_buf[h * D + d];
            }
            break;
        }

        default:
            return false;
    }
    prof_hostops++;
    return true;
}

// execute the QKV fused engine: hidden + norm_w + weights -> q, k, v
static bool axcl_qkv_run(struct ggml_tensor * hidden, struct ggml_tensor * norm_w,
                         struct ggml_tensor * q_w, struct ggml_tensor * k_w, struct ggml_tensor * v_w,
                         float * q_out, float * k_out, float * v_out) {
    if (g_qkv.model == 0) return false;
    const size_t h_sz = 1024 * 4, n_sz = 1024 * 4;
    const size_t qw_sz = (size_t)1024 * 2048 * 4, kw_sz = (size_t)1024 * 1024 * 4;
    const size_t q_sz = 2048 * 4, k_sz = 1024 * 4, v_sz = 1024 * 4;

    // upload activation (hidden); stage weights once
    float h_buf[1024];
    if (hidden->type == GGML_TYPE_F32) memcpy(h_buf, hidden->data, h_sz);
    else {
        const auto * tr = ggml_get_type_traits(hidden->type);
        if (!tr || !tr->to_float) return false;
        tr->to_float(hidden->data, h_buf, 1024);
    }
    axclrtMemcpy(g_qkv.dev_in[0], h_buf, h_sz, AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtMemcpy(g_qkv.dev_in[1], norm_w->data, n_sz, AXCL_MEMCPY_HOST_TO_DEVICE);

    void * dqw = axcl_fused_stage_w(&g_qkv, q_w, qw_sz);
    void * dkw = axcl_fused_stage_w(&g_qkv, k_w, kw_sz);
    void * dvw = axcl_fused_stage_w(&g_qkv, v_w, kw_sz);
    if (!dqw || !dkw || !dvw) return false;
    // NOTE: staged weights are already in their own CMM buffers; we bind them
    // as inputs via SetInputBufferByIndex below

    // bind inputs (activation in pre-allocated, weights in staged buffers)
    axclrtEngineSetInputBufferByIndex(g_qkv.io, 0, g_qkv.dev_in[0], h_sz);
    axclrtEngineSetInputBufferByIndex(g_qkv.io, 1, g_qkv.dev_in[1], n_sz);
    axclrtEngineSetInputBufferByIndex(g_qkv.io, 2, dqw, qw_sz);
    axclrtEngineSetInputBufferByIndex(g_qkv.io, 3, dkw, kw_sz);
    axclrtEngineSetInputBufferByIndex(g_qkv.io, 4, dvw, kw_sz);
    axclrtEngineSetOutputBufferByIndex(g_qkv.io, 0, g_qkv.dev_out[0], q_sz);
    axclrtEngineSetOutputBufferByIndex(g_qkv.io, 1, g_qkv.dev_out[1], k_sz);
    axclrtEngineSetOutputBufferByIndex(g_qkv.io, 2, g_qkv.dev_out[2], v_sz);

    if (axclrtEngineExecute(g_qkv.model, g_qkv.ectx, 0, g_qkv.io) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: QKV engine execute failed\n");
        return false;
    }

    axclrtMemcpy(q_out, g_qkv.dev_out[0], q_sz, AXCL_MEMCPY_DEVICE_TO_HOST);
    axclrtMemcpy(k_out, g_qkv.dev_out[1], k_sz, AXCL_MEMCPY_DEVICE_TO_HOST);
    axclrtMemcpy(v_out, g_qkv.dev_out[2], v_sz, AXCL_MEMCPY_DEVICE_TO_HOST);
    return true;
}

// execute the gate+up fused engine
static bool axcl_gate_up_run(struct ggml_tensor * h, struct ggml_tensor * gate_w, struct ggml_tensor * up_w,
                             float * gate_out, float * up_out) {
    if (g_gate_up.model == 0) return false;
    const size_t h_sz = 1024 * 4;
    const size_t gw_sz = (size_t)1024 * 3072 * 4;
    const size_t o_sz = 3072 * 4;

    float h_buf[1024];
    if (h->type == GGML_TYPE_F32) memcpy(h_buf, h->data, h_sz);
    else {
        const auto * tr = ggml_get_type_traits(h->type);
        if (!tr || !tr->to_float) return false;
        tr->to_float(h->data, h_buf, 1024);
    }
    axclrtMemcpy(g_gate_up.dev_in[0], h_buf, h_sz, AXCL_MEMCPY_HOST_TO_DEVICE);

    void * dgw = axcl_fused_stage_w(&g_gate_up, gate_w, gw_sz);
    void * duw = axcl_fused_stage_w(&g_gate_up, up_w, gw_sz);
    if (!dgw || !duw) return false;

    // pre-bound IO per layer: weights + outputs bound once; only h rebound
    axclrtEngineIO & guio = g_gate_up.io_by_w0[gate_w->data];
    if (guio == nullptr) {
        axclrtEngineCreateIO(g_gate_up.info, &guio);
        axclrtEngineSetInputBufferByIndex(guio, 1, dgw, gw_sz);
        axclrtEngineSetInputBufferByIndex(guio, 2, duw, gw_sz);
        axclrtEngineSetOutputBufferByIndex(guio, 0, g_gate_up.dev_out[0], o_sz);
        axclrtEngineSetOutputBufferByIndex(guio, 1, g_gate_up.dev_out[1], o_sz);
    }
    axclrtEngineSetInputBufferByIndex(guio, 0, g_gate_up.dev_in[0], h_sz);

    if (axclrtEngineExecute(g_gate_up.model, g_gate_up.ectx, 0, guio) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: gate+up engine execute failed\n");
        return false;
    }

    axclrtMemcpy(gate_out, g_gate_up.dev_out[0], o_sz, AXCL_MEMCPY_DEVICE_TO_HOST);
    axclrtMemcpy(up_out, g_gate_up.dev_out[1], o_sz, AXCL_MEMCPY_DEVICE_TO_HOST);
    return true;
}

// attention engine state: q@k buffers Q and K; @v triggers the engine call.
// Intermediates (scale/mask/softmax) compute on garbage from un-computed q@k
// — harmless because the engine's @v output replaces everything downstream
static struct ggml_tensor * attn_q_buf = nullptr;
static struct ggml_tensor * attn_k_buf = nullptr;

// cross-split QKV: detect 3 MUL_MATs sharing src1 (shape-based, no norm)
static struct ggml_tensor * xqkv_q[3] = {nullptr, nullptr, nullptr};
static const void * xqkv_src1 = nullptr; // shared src1 of the projections
static int xqkv_count = 0;

static bool axcl_qkv_try_flush() {
    bool ok = false;
    if (xqkv_count == 3 && g_qkv.model != 0) {
        // h = the shared src1 (already normed by the host-side RMS_NORM)
        struct ggml_tensor * h = xqkv_q[0]->src[1];
        float h_buf[1024];
        if (h->type == GGML_TYPE_F32) memcpy(h_buf, h->data, 1024*4);
        else {
            const auto * tr = ggml_get_type_traits(h->type);
            if (tr && tr->to_float) tr->to_float(h->data, h_buf, 1024);
            else { xqkv_count = 0; xqkv_src1 = nullptr; xqkv_q[0] = xqkv_q[1] = xqkv_q[2] = nullptr; return false; }
        }
        axclrtMemcpy(g_qkv.dev_in[0], h_buf, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
        void * dqw = axcl_fused_stage_w(&g_qkv, xqkv_q[0]->src[0], (size_t)1024*2048*4);
        void * dkw = axcl_fused_stage_w(&g_qkv, xqkv_q[1]->src[0], (size_t)1024*1024*4);
        void * dvw = axcl_fused_stage_w(&g_qkv, xqkv_q[2]->src[0], (size_t)1024*1024*4);
        if (dqw && dkw && dvw) {
            // pre-bound IO per layer: weights + outputs bound once at
            // creation; only the activation input is rebound per call
            axclrtEngineIO & io = g_qkv.io_by_w0[xqkv_q[0]->src[0]->data];
            if (io == nullptr) {
                axclrtEngineCreateIO(g_qkv.info, &io);
                axclrtEngineSetInputBufferByIndex(io, 1, dqw, (size_t)1024*2048*4);
                axclrtEngineSetInputBufferByIndex(io, 2, dkw, (size_t)1024*1024*4);
                axclrtEngineSetInputBufferByIndex(io, 3, dvw, (size_t)1024*1024*4);
                axclrtEngineSetOutputBufferByIndex(io, 0, g_qkv.dev_out[0], 2048*4);
                axclrtEngineSetOutputBufferByIndex(io, 1, g_qkv.dev_out[1], 1024*4);
                axclrtEngineSetOutputBufferByIndex(io, 2, g_qkv.dev_out[2], 1024*4);
            }
            axclrtEngineSetInputBufferByIndex(io, 0, g_qkv.dev_in[0], 1024*4);
            if (axclrtEngineExecute(g_qkv.model, g_qkv.ectx, 0, io) == AXCL_SUCC) {
                axclrtMemcpy(xqkv_q[0]->data, g_qkv.dev_out[0], 2048*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                axclrtMemcpy(xqkv_q[1]->data, g_qkv.dev_out[1], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                axclrtMemcpy(xqkv_q[2]->data, g_qkv.dev_out[2], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                ok = true;
            }
        }
    }
    xqkv_count = 0; xqkv_src1 = nullptr;
    xqkv_q[0] = xqkv_q[1] = xqkv_q[2] = nullptr;
    return ok;
}

static enum ggml_status ggml_backend_axcl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_axcl_context * ctx = (ggml_backend_axcl_context *) backend->context;
    axclrtSetDevice(axcl_get_device_index(ctx->device));
    if (g_axcl_ctx) {
        axclrtSetCurrentContext(g_axcl_ctx); // contexts are thread-local: bind worker threads
    }

    // PRE-PASS: none. The gate+up fusion USED to execute here, but the
    // pre-pass runs before any node is computed — it read the un-normalized
    // hidden from an uninitialized buffer. Both fusions now run inside the
    // main loop where their inputs are already computed.
    std::unordered_set<int> done;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (done.count(i)) continue;
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue; // metadata-only: data pointer already correct
        }
        // cross-split QKV fusion: q/k/v are three consecutive MUL_MATs
        // sharing src1. Executed at the FIRST node — computing late (at the
        // v node) violates the allocator's lifetime assumptions and corrupts
        // downstream reads via reused chunks.
        // fused rmsnorm+qkv: intercept the q_proj whose src1 is a norm node —
        // one engine call replaces the norm engine + qkv fusion
        if (node->op == GGML_OP_MUL_MAT && g_chain.qkvn.model != 0 &&
            node->src[1]->ne[1] == 1 && getenv("GGML_AXCL_NO_FUSION") == nullptr &&
            node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 2048 &&
            node->src[1]->op == GGML_OP_MUL &&
            node->src[1]->src[0] != nullptr && node->src[1]->src[0]->op == GGML_OP_RMS_NORM) {
            // src1 = MUL(norm_out, w); h = the pre-norm hidden, w = gain
            struct ggml_tensor * muln = node->src[1];
            struct ggml_tensor * nrm = muln->src[0];
            struct ggml_tensor * nrm_w = muln->src[1];
            struct ggml_tensor * h = nrm->src[0];
            // find k/v projections in the window (same src1 = the norm node)
            struct ggml_tensor * n1 = nullptr, * n2 = nullptr;
            int i1 = -1, i2 = -1;
            for (int w = i + 1; w < cgraph->n_nodes && w < i + 8; w++) {
                struct ggml_tensor * cand = cgraph->nodes[w];
                if (cand->op == GGML_OP_MUL_MAT && cand->src[1] == muln &&
                    cand->src[0]->ne[0] == 1024 && cand->src[0]->ne[1] == 1024 &&
                    !done.count(w)) {
                    if (n1 == nullptr) { n1 = cand; i1 = w; }
                    else               { n2 = cand; i2 = w; break; }
                }
            }
            if (n2 != nullptr) {
                // run fused norm+qkv now (h is ready: topo order)
                float h_buf[1024];
                if (h->type == GGML_TYPE_F32) memcpy(h_buf, h->data, 1024*4);
                else {
                    const auto * tr = ggml_get_type_traits(h->type);
                    if (!tr || !tr->to_float) { /* fall through to old path */ }
                    else tr->to_float(h->data, h_buf, 1024);
                }
                void * dnw = nrm_w ? axcl_chain_stage_w(&g_chain.qkvn, nrm_w, 1024*4) : nullptr;
                void * dqw = axcl_fused_stage_w(&g_chain.qkvn, node->src[0], (size_t)1024*2048*4);
                void * dkw = axcl_fused_stage_w(&g_chain.qkvn, n1->src[0], (size_t)1024*1024*4);
                void * dvw = axcl_fused_stage_w(&g_chain.qkvn, n2->src[0], (size_t)1024*1024*4);
                if (dnw && dqw && dkw && dvw) {
                    std::lock_guard<std::mutex> lock(axcl_exec_mutex);
                    axclrtMemcpy(g_chain.qkvn.dev_in[0], h_buf, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                    axclrtEngineSetInputBufferByIndex(g_chain.qkvn.io, 0, g_chain.qkvn.dev_in[0], 1024*4);
                    axclrtEngineSetInputBufferByIndex(g_chain.qkvn.io, 1, dnw, 1024*4);
                    axclrtEngineSetInputBufferByIndex(g_chain.qkvn.io, 2, dqw, (size_t)1024*2048*4);
                    axclrtEngineSetInputBufferByIndex(g_chain.qkvn.io, 3, dkw, (size_t)1024*1024*4);
                    axclrtEngineSetInputBufferByIndex(g_chain.qkvn.io, 4, dvw, (size_t)1024*1024*4);
                    axclrtEngineSetOutputBufferByIndex(g_chain.qkvn.io, 0, g_chain.qkvn.dev_out[0], 2048*4);
                    axclrtEngineSetOutputBufferByIndex(g_chain.qkvn.io, 1, g_chain.qkvn.dev_out[1], 1024*4);
                    axclrtEngineSetOutputBufferByIndex(g_chain.qkvn.io, 2, g_chain.qkvn.dev_out[2], 1024*4);
                    if (axclrtEngineExecute(g_chain.qkvn.model, g_chain.qkvn.ectx, 0, g_chain.qkvn.io) == AXCL_SUCC) {
                        g_dbg_qkvn++;
                        axclrtMemcpy(node->data, g_chain.qkvn.dev_out[0], 2048*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                        axclrtMemcpy(n1->data, g_chain.qkvn.dev_out[1], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                        axclrtMemcpy(n2->data, g_chain.qkvn.dev_out[2], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                        // mark the norm node done if it is immediately before
                        done.insert(i1); done.insert(i2);
                        for (int w = i - 1; w >= 0 && i - w < 4; w--) {
                            if (cgraph->nodes[w] == nrm || cgraph->nodes[w] == muln) { done.insert(w); }
                        }
                        continue;
                    }
                }
            }
        }
        if (node->op == GGML_OP_MUL_MAT && g_qkv.model != 0 &&
            node->src[1]->ne[1] == 1 && getenv("GGML_AXCL_NO_FUSION") == nullptr &&
            node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 2048) {
            // window scan: the graph interleaves q's norm/rope between the
            // projections; executing the engine at the q node stays in
            // topological order for all consumers
            struct ggml_tensor * n1 = nullptr, * n2 = nullptr;
            int i1 = -1, i2 = -1;
            for (int w = i + 1; w < cgraph->n_nodes && w < i + 8; w++) {
                struct ggml_tensor * cand = cgraph->nodes[w];
                if (cand->op == GGML_OP_MUL_MAT && cand->src[1] == node->src[1] &&
                    cand->src[0]->ne[0] == 1024 && cand->src[0]->ne[1] == 1024 &&
                    !done.count(w)) {
                    if (n1 == nullptr) { n1 = cand; i1 = w; }
                    else               { n2 = cand; i2 = w; break; }
                }
            }
            if (n2 != nullptr) {
                xqkv_q[0] = node; xqkv_q[1] = n1; xqkv_q[2] = n2; xqkv_count = 3;
                uint64_t tt_q = axcl_us();
                const bool ok = axcl_qkv_try_flush();
                prof_t_qkv += axcl_us() - tt_q; prof_n_qkv++;
                if (ok) {
                    done.insert(i1); done.insert(i2);
                    // no map inserts: q/k/v consumers (rope, set_rows) read
                    // the host writebacks; lingering entries get consumed by
                    // whichever tensor galloc assigns the same host address
                    continue;
                }
                // engine unavailable: fall through, compute nodes individually
            }
        }
        // gate+up fusion: two consecutive MUL_MATs sharing src1 — executed
        // here in the main loop so the normed hidden is already computed
        if (node->op == GGML_OP_MUL_MAT && g_gate_up.model != 0 && i + 1 < cgraph->n_nodes &&
            getenv("GGML_AXCL_NO_FUSION") == nullptr) {
            struct ggml_tensor * n1 = cgraph->nodes[i+1];
            int i1 = i+1;
            if (n1->op == GGML_OP_MUL_MAT && n1->src[1] == node->src[1] &&
                node->src[1]->ne[1] == 1 &&
                node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 3072 &&
                n1->src[0]->ne[0] == 1024 && n1->src[0]->ne[1] == 3072 &&
                !done.count(i) && !done.count(i1)) {
                uint64_t tt_g = axcl_us();
                const bool gok = axcl_gate_up_run(node->src[1], node->src[0], n1->src[0],
                                      (float *) node->data, (float *) n1->data);
                prof_t_gateup += axcl_us() - tt_g; prof_n_gateup++;
                if (gok) {
                    done.insert(i1); // skip the up node below
                    continue;         // gate node's output written by the engine
                }
            }
        }
        // device-resident chain: small ops run as NPU engines, activations
        // stay on the card between engine calls (GGML_AXCL_CHAIN=1).
        // GGML_AXCL_CHAIN_OPS="norm,add,glu" gates individual routes.
        if (g_chain.engines_ok) {
            static char ops_buf[128] = {0};
            static const char * ops_env = getenv("GGML_AXCL_CHAIN_OPS");
            if (ops_env != nullptr && ops_buf[0] == 0) {
                snprintf(ops_buf, sizeof(ops_buf), ",%s,", ops_env);
            }
            // default: only call-eliminating fusions; the standalone small-op
            // engines add PCIe round-trips vs microsecond host scalars
            const bool ops_all = false;
            if (ops_env == nullptr) {
                snprintf(ops_buf, sizeof(ops_buf), ",qkvn,addnorm,gludown,");
            }
            auto op_enabled = [&](const char * n) -> bool {
                if (ops_all) return true;
                char want[32];
                snprintf(want, sizeof(want), ",%s,", n);
                return strstr(ops_buf, want) != nullptr;
            };
            if (node->op == GGML_OP_GET_ROWS) {
                static const char * cnt = getenv("GGML_AXCL_CNT");
                if (cnt) fprintf(stderr, "[cnt] qkvn=%d addnorm=%d gludown=%d norm=%d add=%d glu=%d\n",
                    g_dbg_qkvn, g_dbg_addnorm, g_dbg_gludown, g_dbg_norm, g_dbg_add, g_dbg_glu);
                g_chain.dev.clear(); // new forward pass begins
            } else if (node->op == GGML_OP_MUL && node->ne[0] == 1024 && node->ne[1] == 1 &&
                       ((node->src[0] != nullptr && node->src[0]->op == GGML_OP_RMS_NORM &&
                         node->src[1] != nullptr && node->src[1]->ne[0] == 1024) ||
                        (node->src[1] != nullptr && node->src[1]->op == GGML_OP_RMS_NORM &&
                         node->src[0] != nullptr && node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 1))) {
                // this graph applies the norm weight as a separate MUL; the
                // RMS_NORM beneath it carries no weight input. Either side of
                // the MUL can be the norm output — identify by op.
                struct ggml_tensor * nrm = (node->src[0]->op == GGML_OP_RMS_NORM) ? node->src[0] : node->src[1];
                struct ggml_tensor * wgt = (nrm == node->src[0]) ? node->src[1] : node->src[0];
                struct ggml_tensor * xin = nrm->src[0];
                if (xin != nullptr && xin->op == GGML_OP_ADD && xin->ne[0] == 1024 &&
                    g_chain.addnorm.model != 0 && op_enabled("addnorm")) {
                    // fused residual-add + rmsnorm + weight
                    void * dwv = axcl_chain_stage_w(&g_chain.addnorm, wgt, 1024*4);
                    if (dwv != nullptr) {
                        std::lock_guard<std::mutex> lock(axcl_exec_mutex);
                        axclrtMemcpy(g_chain.addnorm.dev_in[0], xin->src[0]->data, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                        axclrtMemcpy(g_chain.addnorm.dev_in[1], xin->src[1]->data, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                        axclrtEngineSetInputBufferByIndex(g_chain.addnorm.io, 0, g_chain.addnorm.dev_in[0], 1024*4);
                        axclrtEngineSetInputBufferByIndex(g_chain.addnorm.io, 1, g_chain.addnorm.dev_in[1], 1024*4);
                        axclrtEngineSetInputBufferByIndex(g_chain.addnorm.io, 2, dwv, 1024*4);
                        axclrtEngineSetOutputBufferByIndex(g_chain.addnorm.io, 0, g_chain.addnorm.dev_out[0], 1024*4);
                        if (axclrtEngineExecute(g_chain.addnorm.model, g_chain.addnorm.ectx, 0, g_chain.addnorm.io) == AXCL_SUCC) {
                            g_dbg_addnorm++;
                            axclrtMemcpy(node->data, g_chain.addnorm.dev_out[0], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                            // skip the ADD and RMS_NORM beneath us
                            for (int w = i - 1; w >= 0 && i - w < 4; w--) {
                                if (cgraph->nodes[w] == nrm || cgraph->nodes[w] == xin) done.insert(w);
                            }
                            continue;
                        }
                    }
                } else if (op_enabled("norm")) {
                    size_t ib2[2] = {1024*4, 1024*4};
                    struct ggml_tensor * keep[2] = { xin, wgt };
                    // inline: run norm engine on (x, w)
                    void * dwv = axcl_chain_stage_w(&g_chain.norm, wgt, 1024*4);
                    if (dwv != nullptr && xin != nullptr) {
                        std::lock_guard<std::mutex> lock(axcl_exec_mutex);
                        axclrtMemcpy(g_chain.norm.dev_in[0], xin->data, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                        axclrtEngineSetInputBufferByIndex(g_chain.norm.io, 0, g_chain.norm.dev_in[0], 1024*4);
                        axclrtEngineSetInputBufferByIndex(g_chain.norm.io, 1, dwv, 1024*4);
                        axclrtEngineSetOutputBufferByIndex(g_chain.norm.io, 0, g_chain.norm.dev_out[0], 1024*4);
                        if (axclrtEngineExecute(g_chain.norm.model, g_chain.norm.ectx, 0, g_chain.norm.io) == AXCL_SUCC) {
                            g_dbg_norm++;
                            axclrtMemcpy(node->data, g_chain.norm.dev_out[0], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                            for (int w = i - 1; w >= 0 && i - w < 4; w--) {
                                if (cgraph->nodes[w] == nrm) done.insert(w);
                            }
                            (void) keep;
                            continue;
                        }
                    }
                }
            } else if (node->op == GGML_OP_ADD && node->ne[0] == 1024 && node->ne[1] == 1 &&
                       node->src[0]->ne[0] == 1024 && node->src[1]->ne[0] == 1024 &&
                       op_enabled("add")) {
                size_t ib[2] = {1024*4, 1024*4};
                if (axcl_chain_run(&g_chain.add, node, ib, 2, 1024*4)) {
                    g_dbg_add++;
                    continue;
                }
            } else if (node->op == GGML_OP_GLU && node->ne[1] == 1 && node->ne[0] == 3072 &&
                       node->src[0]->ne[0] == 3072 &&
                       node->src[1] != nullptr && node->src[1]->ne[0] == 3072 &&
                       op_enabled("glu")) {
                size_t ib[2] = {3072*4, 3072*4};
                if (axcl_chain_run(&g_chain.glu, node, ib, 2, 3072*4)) {
                    g_dbg_glu++;
                    continue;
                }
            }
        }
        if (node->op == GGML_OP_MUL_MAT &&
            node->src[0]->ne[0] == 3072 && node->src[0]->ne[1] == 1024 &&
            node->src[1] != nullptr && node->src[1]->op == GGML_OP_GLU &&
            node->src[1]->ne[0] == 3072 && g_chain.gludown.model != 0 &&
            node->src[1]->ne[1] == 1) {
            // fused SwiGLU + down projection: g/u are the GLU inputs (gate/up
            // projections), dw is the down weight — one call replaces the GLU
            // engine, the down matmul, and the X upload entirely
            struct ggml_tensor * glun = node->src[1];
            void * ddw = axcl_fused_stage_w(&g_chain.gludown, node->src[0], (size_t)3072*1024*4);
            if (ddw != nullptr) {
                std::lock_guard<std::mutex> lock(axcl_exec_mutex);
                axclrtMemcpy(g_chain.gludown.dev_in[0], glun->src[0]->data, 3072*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtMemcpy(g_chain.gludown.dev_in[1], glun->src[1]->data, 3072*4, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtEngineSetInputBufferByIndex(g_chain.gludown.io, 0, g_chain.gludown.dev_in[0], 3072*4);
                axclrtEngineSetInputBufferByIndex(g_chain.gludown.io, 1, g_chain.gludown.dev_in[1], 3072*4);
                axclrtEngineSetInputBufferByIndex(g_chain.gludown.io, 2, ddw, (size_t)3072*1024*4);
                axclrtEngineSetOutputBufferByIndex(g_chain.gludown.io, 0, g_chain.gludown.dev_out[0], 1024*4);
                if (axclrtEngineExecute(g_chain.gludown.model, g_chain.gludown.ectx, 0, g_chain.gludown.io) == AXCL_SUCC) {
                    g_dbg_gludown++;
                    axclrtMemcpy(node->data, g_chain.gludown.dev_out[0], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                    g_chain.dev[node->data] = g_chain.gludown.dev_out[0];
                    // mark the GLU node done (it never runs separately)
                    for (int w = i - 1; w >= 0 && i - w < 3; w--) {
                        if (cgraph->nodes[w] == glun) { done.insert(w); break; }
                    }
                    continue;
                }
            }
        }
        {
            uint64_t tt_h = axcl_us();
            bool hok = ggml_axcl_host_op(node);
            prof_t_host += axcl_us() - tt_h; prof_n_host++;
            if (hok) continue;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * src0 = node->src[0];
            struct ggml_tensor * src1 = node->src[1];
            // chain: if the activation is device-resident, bind it as X.
            // ALWAYS reset first — a previous MUL_MAT that routed to the
            // attention path leaves the override set, and a stale pointer
            // would corrupt the next matmul's input binding.
            g_chain_x_override = nullptr;
            if (g_chain.engines_ok && src1->ne[1] == 1 &&
                getenv("GGML_AXCL_NO_OVERRIDE") == nullptr) {
                void * hit = axcl_chain_get(src1);
                // GGML_AXCL_OVR_K="3072,1024": restrict overrides to listed K
                static const char * ovr_k = getenv("GGML_AXCL_OVR_K");
                bool allowed = false; // default off: the glu->down handoff
                if (ovr_k != nullptr) {  // corrupts; under investigation
                    char want[32], buf[256];
                    snprintf(want, sizeof(want), ",%lld,", (long long) src0->ne[0]);
                    snprintf(buf, sizeof(buf), ",%s,", ovr_k);
                    allowed = strstr(buf, want) != nullptr;
                }
                if (hit != nullptr && !allowed) {
                    hit = nullptr; // entry consumed; consumer uses host data
                }
                g_chain_x_override = hit;
                if (hit != nullptr && getenv("GGML_AXCL_DEBUG_OVR")) {
                    fprintf(stderr, "[ovr] k=%lld n=%lld\n",
                            (long long)src0->ne[0], (long long)src0->ne[1]);
                }
            }
            // checksum trace: GGML_AXCL_TRACEMM=1 prints src1/dst sums per node
            static int tracemm = getenv("GGML_AXCL_TRACEMM") ? 400 : 0;
            double xin = 0.0;
            if (tracemm > 0) {
                const int64_t nel = src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3];
                for (int64_t e = 0; e < nel; e++)
                    xin += *(const float *) ((const char *) src1->data +
                        (e % (src1->ne[0]*src1->ne[1])) * src1->nb[0] + 0*src1->nb[1] +
                        ((e / (src1->ne[0]*src1->ne[1])) % src1->ne[2]) * src1->nb[2] +
                        (e / (src1->ne[0]*src1->ne[1]*src1->ne[2])) * src1->nb[3]);
                // (approximate: assumes dim0/1 contiguous pairing good enough for a fingerprint)
            }
            axcl_matmul *        mm   = axcl_matmul_get(src1->ne[1], src0->ne[0], src0->ne[1]);
            if (src0->ne[1] == 151936 && getenv("GGML_AXCL_CNT")) {
                fprintf(stderr, "[cnt] qkvn=%d addnorm=%d gludown=%d norm=%d add=%d glu=%d\n",
                        g_dbg_qkvn, g_dbg_addnorm, g_dbg_gludown, g_dbg_norm, g_dbg_add, g_dbg_glu);
            }
            if (mm != nullptr) {
                if (!ggml_axcl_compute_mul_mat(mm, src0, src1, node)) {
                    GGML_LOG_ERROR("ggml-axcl: node %d MUL_MAT failed\n", i);
                    return GGML_STATUS_ABORTED;
                }
                if (g_chain.engines_ok) {
                    g_chain.dev[node->data] = mm->dy; // host write-back already done
                }
            } else {
                // per-head attention matmuls (3D with ne[2]>1 for heads):
                // q@k: ne[0]=seq(growing), ne[1]=128(head_dim), ne[2]=8(kv_heads)
                // @v:  ne[0]=128(head_dim), ne[1]=seq(growing), ne[2]=8(kv_heads)
                if (g_attn.model != 0 && src0->ne[2] > 1 && src1->ne[1] == 1 &&
                    getenv("GGML_AXCL_NO_FUSION") == nullptr &&
                    (src0->ne[1] == 128 || src0->ne[0] == 128)) {
                    if (src0->ne[1] == 128 && src0->ne[0] > 128) {
                        // q@k: buffer Q (src1) and K cache (src0), skip computing
                        attn_q_buf = src1;
                        attn_k_buf = src0;
                        continue; // intermediates (scale/mask/softmax) run on garbage — harmless
                    }
                    if (src0->ne[0] == 128 && src0->ne[1] > 128 &&
                        attn_q_buf != nullptr && attn_k_buf != nullptr) {
                        // @v: run the attention engine with buffered Q, K + this V.
                        // K/V stay device-resident; only new tokens are uploaded.
                        const struct ggml_tensor * qt = attn_q_buf;
                        const struct ggml_tensor * kt = attn_k_buf;
                        const struct ggml_tensor * vt = src0;
                        const int HQ = 16, D = 128, HKV = 8;
                        const int seq = (int) kt->ne[0];
                        // Q view is [D, 1, HQ] → flat [HQ, D]
                        static float eq[HQ * D], out_flat[HQ * D];
                        for (int h = 0; h < HQ; h++)
                            for (int d = 0; d < D; d++)
                                eq[h * D + d] = *(const float *)((const char *)qt->data +
                                    (size_t)d * qt->nb[0] + (size_t)h * qt->nb[2]);
                        // K view is [seq, D, HKV] (token stride nb[0]); V view is
                        // [D, seq, HKV] (token stride nb[1])
                        uint64_t tt_a = axcl_us();
                        bool aok2 = axcl_attn_run(eq, kt->data, vt->data, kt->nb[0], vt->nb[1],
                                          kt->type, vt->type, seq, HKV, D, out_flat);
                        prof_t_attn += axcl_us() - tt_a; prof_n_attn++;
                        if (aok2) {
                            // unpack: [HQ, D] → node [D, 1, HQ]
                            for (int h = 0; h < HQ; h++)
                                for (int d = 0; d < D; d++)
                                    *(float *)((char *)node->data + (size_t)d * node->nb[0] + (size_t)h * node->nb[2]) =
                                        out_flat[h * D + d];
                            attn_q_buf = attn_k_buf = nullptr;
                            prof_hostops++;
                            continue;
                        }
                        attn_q_buf = attn_k_buf = nullptr;
                        // engine failed: fall through to scalar
                    }
                }
                // scalar fallback: full matmul for any src0 type and batch M.
                // This is the prefill path (M>1) — engines are single-token.
                {
                    const int64_t k = src0->ne[0], n = src0->ne[1];
                    const int64_t m = src1->ne[1];
                    const int64_t b2 = node->ne[2], b3 = node->ne[3];
                    const struct ggml_type_traits * tr = ggml_get_type_traits(src0->type);
                    std::vector<float> wrow(src0->type == GGML_TYPE_F32 ? (size_t) 1 : (size_t) k);
                    // self-check: independent double-precision reference of the
                    // first few rows (debug switch GGML_AXCL_SELFCHECK=1)
                    static int selfchk = getenv("GGML_AXCL_SELFCHECK") ? 3 : 0;
                    for (int64_t j3 = 0; j3 < b3; j3++) {
                        for (int64_t j2 = 0; j2 < b2; j2++) {
                            const char * wb = (const char *) src0->data +
                                (size_t)(j2 % src0->ne[2]) * src0->nb[2] + (size_t)(j3 % src0->ne[3]) * src0->nb[3];
                            const char * xb = (const char *) src1->data +
                                (size_t)(j2 % src1->ne[2]) * src1->nb[2] + (size_t)(j3 % src1->ne[3]) * src1->nb[3];
                            char * db = (char *) node->data +
                                (size_t)j2 * node->nb[2] + (size_t)j3 * node->nb[3];
                            for (int64_t nn = 0; nn < n; nn++) {
                                // weight row nn: dequant once, reuse for all m columns
                                const float * w32;
                                if (src0->type == GGML_TYPE_F32) {
                                    w32 = (const float *) (wb + (size_t) nn * src0->nb[1]);
                                } else {
                                    if (!tr || !tr->to_float) break;
                                    tr->to_float(wb + (size_t) nn * src0->nb[1], wrow.data(), k);
                                    w32 = wrow.data();
                                }
                                for (int64_t mm = 0; mm < m; mm++) {
                                    const char * xr = xb + (size_t) mm * src1->nb[1];
                                    float acc = 0.0f;
                                    for (int64_t kk = 0; kk < k; kk++)
                                        acc += w32[kk] * *(const float *) (xr + (size_t) kk * src1->nb[0]);
                                    *(float *) (db + (size_t) mm * node->nb[1] + (size_t) nn * node->nb[0]) = acc;
                                }
                                if (selfchk > 0 && nn < 4) {
                                    // independent reference for row nn, column 0
                                    double ref = 0.0;
                                    for (int64_t kk = 0; kk < k; kk++) {
                                        float wv;
                                        if (src0->type == GGML_TYPE_F32) {
                                            wv = *(const float *)(wb + (size_t)nn*src0->nb[1] + (size_t)kk*src0->nb[0]);
                                        } else {
                                            wv = wrow[kk];
                                        }
                                        ref += (double)wv * *(const float *)(xb + (size_t)kk * src1->nb[0]);
                                    }
                                    float got = *(const float *)(db + (size_t)nn * node->nb[0]);
                                    if (fabs(got - (float)ref) > 1e-3 * (1.0 + fabs(ref))) {
                                        fprintf(stderr, "[selfcheck] MUL_MAT mismatch nn=%lld got=%.6f ref=%.6f (k=%lld n=%lld m=%lld j2=%lld src0ty=%d)\n",
                                                (long long)nn, got, (float)ref, (long long)k, (long long)n, (long long)m, (long long)j2, (int)src0->type);
                                    }
                                }
                            }
                        }
                    }
                }
                if (tracemm > 0) {
                    tracemm--;
                    double ysum = 0.0;
                    const int64_t dnel = node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3];
                    for (int64_t e = 0; e < dnel; e++)
                        ysum += *(const float *) ((const char *) node->data +
                            (e % (node->ne[0]*node->ne[1])) * node->nb[0] +
                            ((e / (node->ne[0]*node->ne[1])) % node->ne[2]) * node->nb[2] +
                            (e / (node->ne[0]*node->ne[1]*node->ne[2])) * node->nb[3]);
                    fprintf(stderr, "[trace-mm] k=%lld n=%lld m=%lld xin=%.4f ysum=%.4f\n",
                            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src1->ne[1], xin, ysum);
                }
                // dump 3D attention matmul tensors for offline verification
                static const char * dumpdir = getenv("GGML_AXCL_DUMPDIR");
                static int ndumps = getenv("GGML_AXCL_DUMPDIR") ? 4 : 0;
                static int ndump2d = getenv("GGML_AXCL_DUMPDIR") ? 2 : 0;
                if (dumpdir != nullptr && src0->ne[2] > 1 && ndumps > 0) {
                    char p[512];
                    snprintf(p, sizeof(p), "%s/n%d_k%lld_n%lld", dumpdir, ndumps,
                             (long long) src0->ne[0], (long long) src0->ne[1]);
                    ndumps--;
                    mkdir(dumpdir, 0755);
                    auto dump = [&](const char * tag, const ggml_tensor * t) {
                        char q[1024];
                        snprintf(q, sizeof(q), "%s_%s.bin", p, tag);
                        FILE * f = fopen(q, "wb");
                        if (f) { fwrite(t->data, 1, ggml_nbytes(t), f); fclose(f); }
                        snprintf(q, sizeof(q), "%s_%s.meta", p, tag);
                        f = fopen(q, "w");
                        if (f) {
                            fprintf(f, "type=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n", (int)t->type,
                                (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
                                t->nb[0],t->nb[1],t->nb[2],t->nb[3]);
                            fclose(f);
                        }
                    };
                    dump("src0", src0); dump("src1", src1); dump("dst", node);
                    fprintf(stderr, "[dump] 3D matmul tensors written to %s\n", p);
                }
                if (dumpdir != nullptr && src0->ne[2] == 1 && (src0->ne[0] == 2048 || src0->ne[0] == 1024) && src0->ne[1] == 2048 && ndump2d > 0) {
                    char p[512];
                    snprintf(p, sizeof(p), "%s/q%d", dumpdir, ndump2d);
                    ndump2d--;
                    auto dump = [&](const char * tag, const ggml_tensor * t) {
                        char q[1024];
                        snprintf(q, sizeof(q), "%s_%s.bin", p, tag);
                        FILE * f = fopen(q, "wb");
                        if (f) { fwrite(t->data, 1, ggml_nbytes(t), f); fclose(f); }
                        snprintf(q, sizeof(q), "%s_%s.meta", p, tag);
                        f = fopen(q, "w");
                        if (f) {
                            fprintf(f, "type=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n", (int)t->type,
                                (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
                                t->nb[0],t->nb[1],t->nb[2],t->nb[3]);
                            fclose(f);
                        }
                    };
                    dump("src0", src0); dump("src1", src1); dump("dst", node);
                    fprintf(stderr, "[dump] o_proj tensors written to %s\n", p);
                }
                prof_hostops++;
            }
        } else {
            GGML_LOG_ERROR("ggml-axcl: node %d op %s not supported\n", i, ggml_op_name(node->op));
            return GGML_STATUS_ABORTED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

//
// events / synchronization
//
// our compute is fully synchronous, but the scheduler pipelines split graphs
// and relies on event_record/event_wait between backends — with NULL hooks it
// raced split-input copies against our compute (stale reads). A trivial
// atomic-flag event makes the ordering explicit.
//
struct axcl_event {
    std::atomic<int> flag{0};
};

static void ggml_backend_axcl_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // all axcl work is issued and completed synchronously
}

static void ggml_backend_axcl_event_record(ggml_backend_t backend, ggml_backend_event_t ev) {
    GGML_UNUSED(backend);
    auto * e = (axcl_event *) ev->context;
    e->flag.store(1, std::memory_order_release);
}

static void ggml_backend_axcl_event_wait(ggml_backend_t backend, ggml_backend_event_t ev) {
    GGML_UNUSED(backend);
    auto * e = (axcl_event *) ev->context;
    while (e->flag.load(std::memory_order_acquire) == 0) {
        // spin
    }
}

static ggml_backend_event_t ggml_backend_axcl_device_event_new(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    auto * ev = new ggml_backend_event();
    ev->device = nullptr;
    ev->context = new axcl_event();
    return ev;
}

static void ggml_backend_axcl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t ev) {
    GGML_UNUSED(dev);
    delete (axcl_event *) ev->context;
    delete ev;
}

static void ggml_backend_axcl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t ev) {
    GGML_UNUSED(dev);
    auto * e = (axcl_event *) ev->context;
    while (e->flag.load(std::memory_order_acquire) == 0) {
        // spin
    }
}

static const struct ggml_backend_i ggml_backend_axcl_interface = {
    /* .get_name           = */ ggml_backend_axcl_name,
    /* .free               = */ ggml_backend_axcl_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .set_tensor_2d_async= */ NULL,
    /* .get_tensor_2d_async= */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ ggml_backend_axcl_synchronize,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_axcl_graph_compute,
    /* .event_record       = */ ggml_backend_axcl_event_record,
    /* .event_wait         = */ ggml_backend_axcl_event_wait,
    /* .graph_optimize     = */ NULL,
};

ggml_backend_t ggml_backend_axcl_init(int32_t device) {
    if (device < 0 || device >= axcl_get_device_count()) {
        GGML_LOG_ERROR("ggml-axcl: invalid device %d\n", device);
        return nullptr;
    }
    if (axcl_engine_global_init()) {
        // GGML_AXCL_NO_ENGINES=1: activate the device but skip all engine
        // loads — bisecting host-memory corruption during model load
        const bool no_engines = getenv("GGML_AXCL_NO_ENGINES") != nullptr;
        if (!no_engines) {
        axcl_preload_all_engines(); // outside the activation mutex
        axcl_weight_pool_init();
        axcl_attn_load();
        if (getenv("GGML_AXCL_CHAIN") != nullptr) {
            axcl_chain_load();
        }
        // QKV fused engine: rms_norm(hidden) -> q, k, v in one call
        if (axcl_fused_load(&g_qkv, "/usr/local/share/ggml-axcl/qkv_nn_h1024_q2048_kv1024.axmodel",
                            {"h", "q_w", "k_w", "v_w"}, {"q", "k", "v"})) {
            axcl_fused_alloc(&g_qkv,
                {1024*4, (size_t)1024*2048*4, (size_t)1024*1024*4, (size_t)1024*1024*4},
                {2048*4, 1024*4, 1024*4});
            GGML_LOG_INFO("ggml-axcl: QKV no-norm engine loaded\n");
        }
        // gate+up fused engine: h @ gate_w, h @ up_w in one call
        if (axcl_fused_load(&g_gate_up, "/usr/local/share/ggml-axcl/gate_up_h1024_i3072.axmodel",
                            {"h", "gate_w", "up_w"}, {"gate", "up"})) {
            axcl_fused_alloc(&g_gate_up,
                {1024*4, (size_t)1024*3072*4, (size_t)1024*3072*4},
                {(size_t)3072*4, (size_t)3072*4});
            GGML_LOG_INFO("ggml-axcl: gate+up fused engine loaded\n");
        }
        }
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
    GGML_UNUSED(dev);
    // Host-memory backend (same contract as ggml-blas): compute reads and
    // writes tensors directly in host RAM.
    return ggml_backend_cpu_buffer_type();
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
    // UNIVERSAL BACKEND: accept every op — matmuls go to NPU engines,
    // everything else runs host-side in our graph_compute. This eliminates
    // ALL scheduler splits during inference.
    // Debug bisect: GGML_AXCL_SKIP_OPS="ROPE,SET_ROWS" pushes named ops to CPU.
    static const char * skip_env = getenv("GGML_AXCL_SKIP_OPS");
    if (skip_env != nullptr && skip_env[0] != '\0') {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s,", skip_env);
        char name[64];
        snprintf(name, sizeof(name), "%s,", ggml_op_name(op->op));
        if (strstr(buf, name) != nullptr) return false;
    }
    // Debug bisect: GGML_AXCL_MM_CPU="1024x2048,3072x1024" pushes MUL_MATs
    // with those (K x N) weight shapes to the CPU backend
    static const char * mm_env = getenv("GGML_AXCL_MM_CPU");
    if (mm_env != nullptr && op->op == GGML_OP_MUL_MAT && op->src[0] != nullptr) {
        char want[64];
        snprintf(want, sizeof(want), "%lldx%lld,", (long long) op->src[0]->ne[0], (long long) op->src[0]->ne[1]);
        char buf[512];
        snprintf(buf, sizeof(buf), "%s,", mm_env);
        if (strstr(buf, want) != nullptr) return false;
    }
    // Default: BLAS-style claim set — only MUL_MAT, verified correct
    // end-to-end. Claiming metadata ops (VIEW/PERMUTE) makes the scheduler's
    // split machinery create output copies we never write (metadata ops have
    // no compute), so downstream splits read stale copy buffers — corrupted
    // K/V views. GGML_AXCL_UNIVERSAL=2: claim all COMPUTE ops but leave
    // metadata on the CPU — eliminates splits between our compute ops while
    // avoiding the metadata split-copy hazard.
    // vocab-sized projections stay on CPU: dequantizing to f32 and shipping
    // 622MB/token dominates NPU DRAM bandwidth; the CPU's native Q8_0 GEMV
    // does it with no transfer (GGML_AXCL_NPU_VOCAB=1 to override)
    static const char * npu_vocab_env = getenv("GGML_AXCL_NPU_VOCAB");
    if (npu_vocab_env == nullptr && op->op == GGML_OP_MUL_MAT &&
        op->src[0] != nullptr && op->src[0]->ne[1] > 32768) {
        return false;
    }
    static const char * uni_env = getenv("GGML_AXCL_UNIVERSAL");
    if (uni_env != nullptr && uni_env[0] == '2') {
        switch (op->op) {
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
                return false;
            default:
                return true;
        }
    }
    // GGML_AXCL_CLAIM="RMS_NORM,ADD": MUL_MAT plus exactly the listed ops —
    // minimal bisect for the claim-hazard
    static const char * claim_env = getenv("GGML_AXCL_CLAIM");
    if (claim_env != nullptr) {
        if (op->op == GGML_OP_MUL_MAT) return true;
        char buf[512];
        snprintf(buf, sizeof(buf), ",%s,", claim_env);
        char name[64];
        snprintf(name, sizeof(name), ",%s,", ggml_op_name(op->op));
        return strstr(buf, name) != nullptr;
    }
    // GGML_AXCL_CHAIN=1: device-resident chain — claim every COMPUTE op.
    // Metadata ops and CONT stay on the CPU; FLASH_ATTN_EXT must NOT be
    // claimed: it flips llama.cpp to the FA path, and our FA host-op passes
    // the permuted K view's strides to the attention engine incorrectly —
    // the manual attention route is the verified-correct one.
    static const char * chain_env = getenv("GGML_AXCL_CHAIN");
    if (chain_env != nullptr) {
        switch (op->op) {
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
                return false;
            case GGML_OP_FLASH_ATTN_EXT:
                return false; // CPU flash-attn: engine output wrong on long
                              // prompts; needs tensor-level diff debugging
            default:
                return true;
        }
    }
    if (uni_env != nullptr) {
        return true;
    }
    return op->op == GGML_OP_MUL_MAT;
}

static bool ggml_backend_axcl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    // host-memory backend: any buffer marked host works (same as ggml-blas)
    return ggml_backend_buft_is_host(buft);
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
    /* .event_new            = */ ggml_backend_axcl_device_event_new,
    /* .event_free           = */ ggml_backend_axcl_device_event_free,
    /* .event_synchronize    = */ ggml_backend_axcl_device_event_synchronize,
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
