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
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

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

// bf16 (u16 bit pattern) -> f32. NEON processes 8/cycle; the scalar path is
// the bit-exact reference (shift into the high half of the f32 bits).
static inline void axcl_bf16_to_f32(const uint16_t * src, float * dst, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t b = vld1q_u16(src + i);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b), 16)));
    }
    for (; i < n; i++) {
        uint32_t u = (uint32_t) src[i] << 16;
        memcpy(&dst[i], &u, 4);
    }
#else
    for (int i = 0; i < n; i++) {
        uint32_t u = (uint32_t) src[i] << 16;
        memcpy(&dst[i], &u, 4);
    }
#endif
}

// f32 -> bf16 (truncate, matching the historical loader behavior)
static inline void axcl_f32_to_bf16(const float * src, uint16_t * dst, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        uint16x4_t rl = vshrn_n_u32(vreinterpretq_u32_f32(lo), 16);
        uint16x4_t rh = vshrn_n_u32(vreinterpretq_u32_f32(hi), 16);
        vst1q_u16(dst + i, vcombine_u16(rl, rh));
    }
    for (; i < n; i++) {
        uint32_t u;
        memcpy(&u, &src[i], 4);
        dst[i] = (uint16_t) (u >> 16);
    }
#else
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &src[i], 4);
        dst[i] = (uint16_t) (u >> 16);
    }
#endif
}

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
    if (!f && !env) {
        // prefer the largest installed context engine
        path = "/usr/local/share/ggml-axcl/attn_h16_d128_t512.axmodel";
        f = fopen(path, "r");
    }
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
                // match CPU arithmetic exactly: double accumulation,
                // float division for mean, float 1/sqrtf for scale
                double ss = 0.0;
                for (int64_t i = 0; i < ne0; i++) ss += (double)(xr[i] * xr[i]);
                const float mean = (float)(ss / ne0);
                const float rms = 1.0f / sqrtf(mean + eps);
                if (src1 != nullptr) {
                    const float * w = (const float *) ((const char *) src1->data + axcl_row_off(src1, r));
                    for (int64_t i = 0; i < ne0; i++) dr[i] = xr[i] * rms * w[i];
                } else {
                    for (int64_t i = 0; i < ne0; i++) dr[i] = xr[i] * rms;
                }
                if (getenv("GGML_AXCL_CHECKSUM") && node->ne[0] == 1024) {
                    static int rnn = 0;
                    double cs = 0;
                    for (int q = 0; q < 8; q++) cs += dr[q];
                    fprintf(stderr, "[CS-NORM] n=%d nrows=%lld cs=%.6f\n", rnn++, (long long) nr, cs);
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
            if (!mul && getenv("GGML_AXCL_CHECKSUM") && node->ne[0] == 1024 && node->ne[1] == 1 &&
                src1->ne[0] == 1024 && src0->ne[0] == 1024) {
                const float * f = (const float *) node->data;
                double cs = 0;
                for (int q = 0; q < 8; q++) cs += f[q];
                static int addn = 0;
                fprintf(stderr, "[CS-ADD] n=%d cs=%.6f\n", addn++, cs);
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
            static const char * srdbg = getenv("GGML_AXCL_CHECKSUM");
            bool sr_first = false;
            if (srdbg && src1->ne[0] == 1 && src0->ne[0] == 1024 && src0->ne[1] == 1) {
                static int srn = 0;
                if (srn < 8) { sr_first = true; }
            }
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
            if (getenv("GGML_AXCL_CHECKSUM") && node->ne[0] == 1024 && node->ne[1] == 1 &&
                src1->ne[0] == 1) {
                // K/V cache row write (CPU reference path): dump what lands
                static int srn = 0;
                if (srn < 8) {
                    const char * drow = (const char *) node->data +
                        (size_t)*(const int32_t *) src1->data * node->nb[1];
                    float f0[4] = {0, 0, 0, 0};
                    axcl_kv_fetch(drow, node->type, f0, 4);
                    fprintf(stderr, "[CS-SR] n=%d ty=%d v=%.6f %.6f %.6f %.6f\n",
                            srn++, (int) node->type, f0[0], f0[1], f0[2], f0[3]);
                }
            }
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            { static int fan = 0; if (getenv("GGML_AXCL_FACOUNT") && fan < 10) { fan++;
              fprintf(stderr, "[FA-HOSTOP] n=%d nq=%d seq=%d\n", fan, (int)src0->ne[1], -1); } }
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
            { static int fn = 0; if (fn < 4) { fn++;
              fprintf(stderr, "[fa-strides] Q ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] | K ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu] ty=%d | V nb=[%zu,%zu,%zu] ty=%d | out nb=[%zu,%zu,%zu]\n",
                (long long)qt->ne[0], (long long)qt->ne[1], (long long)qt->ne[2], qt->nb[0], qt->nb[1], qt->nb[2],
                (long long)kt->ne[0], (long long)kt->ne[1], (long long)kt->ne[2], kt->nb[0], kt->nb[1], kt->nb[2], (int)kt->type,
                vt->nb[0], vt->nb[1], vt->nb[2], (int)vt->type,
                node->nb[0], node->nb[1], node->nb[2]); } }
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
            if (seq_total > g_attn.t) {
                // engine ctx too small: scalar fallback (exact reference math)
                const int HQs = 16, Ds = 128, HKVs = 8, G = HQs / HKVs;
                float fa_s;
                memcpy(&fa_s, node->op_params, sizeof(float));
                if (fa_s == 0.0f) fa_s = 1.0f / sqrtf((float) Ds);
                std::vector<float> qrow(Ds), krow(Ds), vrow(Ds), scores(seq_total), wr(Ds);
                for (int tq_i = 0; tq_i < nq; tq_i++) {
                    const int seq_t = seq_total - nq + tq_i + 1;
                    for (int h = 0; h < HQs; h++) {
                        for (int d = 0; d < Ds; d++)
                            qrow[d] = *(const float *)((const char *)qt->data +
                                (size_t)tq_i * qt->nb[1] + (size_t)d * qt->nb[0] + (size_t)h * qt->nb[2]);
                        const int hk = h / G;
                        float mx = -INFINITY;
                        for (int t = 0; t < seq_t; t++) {
                            axcl_kv_fetch((const char *)kt->data + (size_t)(hk * Ds) * ggml_type_size(kt->type) +
                                          (size_t)t * kt->nb[1], kt->type, krow.data(), Ds);
                            float acc = 0.0f;
                            for (int d = 0; d < Ds; d++) acc += qrow[d] * krow[d];
                            scores[t] = acc * fa_s;
                            mx = fmaxf(mx, scores[t]);
                        }
                        float sum = 0.0f;
                        for (int t = 0; t < seq_t; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
                        for (int d = 0; d < Ds; d++) wr[d] = 0.0f;
                        for (int t = 0; t < seq_t; t++) {
                            axcl_kv_fetch((const char *)vt->data + (size_t)(hk * Ds) * ggml_type_size(vt->type) +
                                          (size_t)t * vt->nb[1], vt->type, vrow.data(), Ds);
                            const float w = scores[t] / sum;
                            for (int d = 0; d < Ds; d++) wr[d] += w * vrow[d];
                        }
                        for (int d = 0; d < Ds; d++)
                            *(float *)((char *)node->data + (size_t)tq_i * node->nb[1] +
                                (size_t)d * node->nb[0] + (size_t)h * node->nb[2]) = wr[d];
                    }
                }
                break;
            }

            // the attention-engine route here has known layout bugs
            // (single-token outputs validated wrong vs numpy reference);
            // always use the exact scalar fallback unless explicitly forced
            if (getenv("GGML_AXCL_FA_ENGINE") == nullptr) {
                const int HQs = 16, Ds = 128, HKVs = 8, G = HQs / HKVs;
                float fa_s2;
                memcpy(&fa_s2, node->op_params, sizeof(float));
                if (fa_s2 == 0.0f) fa_s2 = 1.0f / sqrtf((float) Ds);
                std::vector<float> qrow(Ds), krow(Ds), vrow(Ds), scores(seq_total), wr(Ds);
                for (int tq_i = 0; tq_i < nq; tq_i++) {
                    const int seq_t = seq_total - nq + tq_i + 1;
                    for (int h = 0; h < HQs; h++) {
                        for (int d = 0; d < Ds; d++)
                            qrow[d] = *(const float *)((const char *)qt->data +
                                (size_t)tq_i * qt->nb[1] + (size_t)d * qt->nb[0] + (size_t)h * qt->nb[2]);
                        const int hk = h / G;
                        float mx = -INFINITY;
                        for (int t = 0; t < seq_t; t++) {
                            axcl_kv_fetch((const char *)kt->data + (size_t)(hk * Ds) * ggml_type_size(kt->type) +
                                          (size_t)t * kt->nb[1], kt->type, krow.data(), Ds);
                            float acc = 0.0f;
                            for (int d = 0; d < Ds; d++) acc += qrow[d] * krow[d];
                            scores[t] = acc * fa_s2;
                            mx = fmaxf(mx, scores[t]);
                        }
                        float sum = 0.0f;
                        for (int t = 0; t < seq_t; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
                        for (int d = 0; d < Ds; d++) wr[d] = 0.0f;
                        for (int t = 0; t < seq_t; t++) {
                            axcl_kv_fetch((const char *)vt->data + (size_t)(hk * Ds) * ggml_type_size(vt->type) +
                                          (size_t)t * vt->nb[1], vt->type, vrow.data(), Ds);
                            const float w = scores[t] / sum;
                            for (int d = 0; d < Ds; d++) wr[d] += w * vrow[d];
                        }
                        for (int d = 0; d < Ds; d++)
                            *(float *)((char *)node->data + (size_t)tq_i * node->nb[1] +
                                (size_t)d * node->nb[0] + (size_t)h * node->nb[2]) = wr[d];
                    }
                }
                break;
            }
            void * dk = nullptr, * dv = nullptr;
            if (!axcl_attn_sync_kv(kt->data, vt->data, kt->nb[1], vt->nb[1],
                                   kt->type, vt->type, seq_total, HKV, &dk, &dv)) return false;

            const int base = seq_total - nq; // first cache slot of this ubatch
            // NOTE: the attention engine graph bakes the 1/sqrt(D) softmax
            // scale (see gemm/make_attention.py) — Q must NOT be pre-scaled
            // here or scores get scaled twice (the double-scale was the FA
            // wrong-output bug)
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

//
// whole-layer engines (vendor-class: 1 NPU call per transformer layer)
//
// Template axmodels built by pulsar2 llm_build (qwen3, bf16 IO, 3 shape
// groups). Weights are patched from the GGUF at load: per-(row, kgroup-256)
// int8 quant, nibble = (q8>>4)+8 scattered via a position table, scale
// entries patched per group. axclrtEngineLoadFromMem hangs in runtime
// V3.6.5_P1 (verified with unmodified bytes) — patched engines go through
// a temp file + LoadFromFile instead.
//
struct axcl_layer_engine {
    uint64_t model = 0, ectx = 0;
    axclrtEngineIOInfo info = nullptr;
    axclrtEngineIO     io   = nullptr;
    int ik = -1, iv = -1, ii = -1, ix = -1, im = -1;   // group-0 inputs
    int iko = -1, ivo = -1, iyo = -1;                  // group-0 outputs
    // device KV cache (this layer's, full ctx)
    void * dk = nullptr, * dv = nullptr;
    int wm = -1;  // watermark: highest written row + 1
    bool bound = false; // static IO bindings (K/V/idx/mask) done once
    axclrtEngineIO io_chunk[12] = {}; // dedicated IO handle PER shape group
                                      // (shared handles mis-bind internals;
                                      // discovered via phase_c harnesses)
};

// per-layer GGUF weight stash: 7 matrices + 4 norms by (rows,cols) shape
struct axcl_gguf_layer {
    const struct ggml_tensor * t[11] = {nullptr}; // q k v o gate up down in_ln post_ln q_norm k_norm
    bool complete() const {
        for (int i = 0; i < 11; i++) if (t[i] == nullptr) return false;
        return true;
    }
};
struct axcl_layer_ctx {
    axcl_layer_engine eng[64];  // per transformer layer
    int n_layer = 0;
    int ctx_len = 2048;
    bool loaded = false;
    // shared device IO
    void * dx_in = nullptr;   // bf16 [1,1,1024] hidden input
    void * d_idx  = nullptr;  // u32
    void * d_mask = nullptr;  // bf16 [ctx+1] rows for all positions
    void * d_kout = nullptr, * d_vout = nullptr, * d_yout = nullptr;
    void * d_yout_alt = nullptr;  // hidden double-buffer: the engine must never
                                  // read its input from the buffer it writes

    // host staging
    std::vector<uint16_t> h_bf16;
    std::vector<uint8_t>  h_row;
    // decode dispatch state
    bool armed = false;       // whole-layer decode active for this graph
    int  next_layer = 0;      // layer index of the next q_proj anchor
    void * d_hidden = nullptr; // device bf16 hidden feeding next layer
    // multi-token (prefill) armed passes: one hidden buffer per token
    std::vector<void *> d_hidden_m;
    int n_tok = 0;
    int pos_base = 0;
    uint64_t calls = 0, us = 0;
    // host KV cache views (captured from the decode graph's SET_ROWS dsts)
    struct ggml_tensor * host_k[64] = {nullptr};
    struct ggml_tensor * host_v[64] = {nullptr};
    int64_t k_nb1[64] = {0}, v_nb1[64] = {0};
    // full cache base tensors ([1024, n_ctx]) for resync
    struct ggml_tensor * kbase[64] = {nullptr};
    struct ggml_tensor * vbase[64] = {nullptr};
    int pos = -1, pos_last_pass = -1;
    struct ggml_tensor * final_norm = nullptr;
    struct ggml_tensor * out_add = nullptr;
    void * post_hidden = nullptr;
    size_t mask_row_bytes = 0;
    void * d_mask_row = nullptr;
    void * y_next = nullptr;
    // per-token device sync: idx upload + mask-row refresh are identical for
    // every layer at the same position — done once per (pos), not per layer
    int synced_pos = -1;
    // host write-back watermark per layer: rows [0, host_wm) are materialized
    // in llama.cpp's KV cache; [host_wm, wm) are device-authoritative only
    int host_wm[64] = {0};
    // pinned staging (unpinned small transfers cost ~1ms each on this stack)
    void * pin_idx = nullptr;    // 4 bytes
    void * pin_row = nullptr;    // 2048 bytes (one KV row, bf16)
    void * pin_rows = nullptr;   // 64 KV rows, bf16 (deferred flush batches)
    void * pin_hidden = nullptr; // 2048 bytes (hidden, bf16)
    uint16_t * pin_logits = nullptr; // [151936] bf16 (post engine output)
    // per-pass stats (debug print): wall vs engine time
    uint64_t last_wall_us = 0, last_eng_us = 0;
    uint64_t chain_t0 = 0;       // async chain: layer-0 enqueue timestamp
    // batched prefill (shape-group ladder): groups 1..9 process 128-token
    // chunks with growing cache prefix; feature-detected at load
    int n_groups = 1;
    int prefill_chunk = 128;
    void * d_chunk_in  = nullptr; // [128,1024] bf16 chunk input staging
    void * d_chunk_out = nullptr; // [128,1024] bf16 chunk output staging
    void * d_idx_m     = nullptr; // [128] u32
    void * d_mask_m    = nullptr; // [128,1152] bf16 max
    void * d_chunk_ko  = nullptr; // [128,1024] bf16 K rows out (dedicated —
                                  // offset output binds mis-execute)
    void * d_chunk_vo  = nullptr; // [128,1024] bf16 V rows out
    void * pin_idx_m   = nullptr; // [128] u32 pinned
    void * pin_mask_m  = nullptr; // [128,1152] bf16 pinned
    void * pin_chunk   = nullptr; // [128,1024] bf16 pinned (layer-0 H2D)
    // full-prefill hidden buffers (layer-major order: layer l-1's outputs
    // for ALL chunks must persist while layer l runs) + host staging
    void * h_all_a = nullptr, * h_all_b = nullptr; // [m<=1152,1024] bf16
    void * pin_h_all = nullptr;
    void * batched_last_hbuf = nullptr; // h_all_* holding the last layer's out
};
static axcl_layer_ctx g_layer;
static int n_layer_avail() { return g_layer.n_layer ? g_layer.n_layer : 28; }

// post engine: final norm + lm_head on NPU (bf16 hidden in, logits out)
struct axcl_post_engine {
    uint64_t model = 0, ectx = 0;
    axclrtEngineIOInfo info = nullptr;
    axclrtEngineIO io = nullptr;
    int ix = -1, iyo = -1;
    void * dy = nullptr;             // logits [n_out] bf16 or f32
    bool ok = false;
    bool tried = false;              // never retry a failed load (per-token
                                     // reloads of a bad engine cost ~4s/token)
    bool out_f32 = false;            // pulsar2-build heads emit f32 logits
    // vocab-trimmed post: n_out < 151936 logits, expanded via trim_ids
    int n_out = 151936;
    int32_t * trim_ids = nullptr;    // trimmed position -> full token id
    float * f32_scratch = nullptr;   // [n_out] converted logits
};
static axcl_post_engine g_post;
static bool g_layer_logits_on_npu = false; // set once the post engine runs

// multi-row vocab head (speculative verification): X [64,1024] f32 in,
// Y [64,151936] f32 out — ONE call produces logits for up to 64 positions
struct axcl_vocab64_engine {
    uint64_t model = 0, ectx = 0;
    axclrtEngineIOInfo info = nullptr;
    axclrtEngineIO io = nullptr;
    int ix = -1, iyo = -1;
    void * dx = nullptr;            // [64,1024] f32
    void * dy = nullptr;            // [64,151936] f32
    float * pin_in = nullptr;       // pinned staging, 64*1024 f32
    float * pin_out = nullptr;      // pinned, 64*151936 f32
    bool ok = false;
    bool tried = false;
};
static axcl_vocab64_engine g_v64;

static void axcl_vocab64_load() {
    if (g_v64.tried || g_v64.ok) return;
    g_v64.tried = true;
    const char * env = getenv("GGML_AXCL_VOCAB64");
    if (!env) return;
    if (axclrtEngineLoadFromFile(env, &g_v64.model) != AXCL_SUCC) {
        GGML_LOG_WARN("ggml-axcl: vocab64 %s failed to load\n", env);
        g_v64.model = 0;
        return;
    }
    axclrtEngineGetIOInfo(g_v64.model, &g_v64.info);
    axclrtEngineCreateIO(g_v64.info, &g_v64.io);
    axclrtEngineCreateContext(g_v64.model, &g_v64.ectx);
    g_v64.ix = axclrtEngineGetInputIndexByName(g_v64.info, "X");
    g_v64.iyo = axclrtEngineGetOutputIndexByName(g_v64.info, "Y");
    if (g_v64.ix < 0 || g_v64.iyo < 0) { g_v64.model = 0; return; }
    if (axclrtMalloc(&g_v64.dx, 64 * 1024 * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMalloc(&g_v64.dy, (size_t) 64 * 151936 * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
        axclrtMallocHost((void **) &g_v64.pin_in, 64 * 1024 * 4) != AXCL_SUCC ||
        axclrtMallocHost((void **) &g_v64.pin_out, (size_t) 64 * 151936 * 4) != AXCL_SUCC) {
        g_v64.model = 0;
        return;
    }
    g_v64.ok = true;
    GGML_LOG_INFO("ggml-axcl: vocab64 verify head ready (64-row logits)\n");
}

static void axcl_post_load() {
    if (g_post.ok || g_post.tried) return;
    g_post.tried = true;
    const char * env = getenv("GGML_AXCL_POST_MODEL");
    char p[600];
    snprintf(p, sizeof(p), "%s", env ? env : "/usr/local/share/ggml-axcl/layer/qwen3_post.axmodel");
    FILE * f = fopen(p, "r");
    if (!f) return;
    fclose(f);
    if (axclrtEngineLoadFromFile(p, &g_post.model) != AXCL_SUCC) return;
    axclrtEngineGetIOInfo(g_post.model, &g_post.info);
    axclrtEngineCreateIO(g_post.info, &g_post.io);
    axclrtEngineCreateContext(g_post.model, &g_post.ectx);
    g_post.ix = axclrtEngineGetInputIndexByName(g_post.info, "input");
    g_post.iyo = axclrtEngineGetOutputIndexByName(g_post.info, "output");
    if (g_post.ix < 0) g_post.ix = axclrtEngineGetInputIndexByName(g_post.info, "X");       // pulsar2-build engines
    if (g_post.iyo < 0) g_post.iyo = axclrtEngineGetOutputIndexByName(g_post.info, "Y");    // use ONNX names
    if (g_post.ix < 0 || g_post.iyo < 0) { g_post.model = 0; return; }
    uint64_t out_sz = axclrtEngineGetOutputSizeByIndex(g_post.info, 0, (uint32_t) g_post.iyo);
    if (out_sz > 4) g_post.n_out = (int) (out_sz / 2);   // assume bf16 logits
    if (g_post.n_out != 151936) {
        // vocab-trimmed post: load the kept-ids map (JSON int array); the
        // map count is ground truth for n_out — the engine may emit bf16
        // (llm_build) or f32 (pulsar2 build) logits
        const char * tenv = getenv("GGML_AXCL_POST_TRIM");
        if (!tenv) {
            GGML_LOG_ERROR("ggml-axcl: trimmed post (%d logits) needs GGML_AXCL_POST_TRIM\n", g_post.n_out);
            g_post.model = 0; return;
        }
        FILE * tf = fopen(tenv, "r");
        if (!tf) { GGML_LOG_ERROR("ggml-axcl: trim map %s unreadable\n", tenv); g_post.model = 0; return; }
        static char buf[1 << 20];
        int len = (int) fread(buf, 1, sizeof(buf) - 1, tf);
        fclose(tf);
        buf[len] = 0;
        static int32_t tmp_ids[200000];
        int n = 0; char * s = buf;
        while (n < 200000 && *s) {
            while (*s && (*s == ',' || *s == ' ' || *s == '[' || *s == ']')) s++;
            if (!*s) break;
            char * s0 = s;
            long v = strtol(s, &s, 10);
            if (s == s0) break;
            tmp_ids[n++] = (int32_t) v;
        }
        if (out_sz == (uint64_t) n * 2) {
            g_post.n_out = n; g_post.out_f32 = false;
        } else if (out_sz == (uint64_t) n * 4) {
            g_post.n_out = n; g_post.out_f32 = true;
        } else {
            GGML_LOG_ERROR("ggml-axcl: trim map has %d ids, engine output %llu B does not match bf16/f32\n",
                           n, (unsigned long long) out_sz);
            g_post.model = 0; return;
        }
        g_post.trim_ids = (int32_t *) malloc((size_t) g_post.n_out * 4);
        memcpy(g_post.trim_ids, tmp_ids, (size_t) g_post.n_out * 4);
        g_post.f32_scratch = (float *) malloc((size_t) g_post.n_out * 4);
        GGML_LOG_INFO("ggml-axcl: trimmed post (%d of 151936 vocab, %s logits)\n",
                      g_post.n_out, g_post.out_f32 ? "f32" : "bf16");
    } else if (out_sz == (uint64_t) 151936 * 4) {
        g_post.out_f32 = true;   // full-vocab pulsar2-build head
    }
    axclrtMalloc(&g_post.dy, out_sz ? out_sz : (size_t) 151936 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    g_post.ok = true;
    GGML_LOG_INFO("ggml-axcl: post engine ready (NPU logits)\n");
}


static axcl_gguf_layer g_gguf[64];
static int g_gguf_n = 0;          // layers seen
static bool g_layer_from_gguf = false;

// ---- dynamic GGUF weight patching (layout_v4) ----
// Template engines carry weights as RAW bf16 at deterministic positions
// (validated byte-exact against baked engines). The loader scatters each
// layer's GGUF weights (dequantized) + norm vectors into a copy of the
// template and loads it from a temp file (LoadFromMem hangs in V3.6.5).
struct axcl_layer_layout {
    struct M { char name[12]; uint32_t rows, cols; std::vector<uint64_t> off; };
    std::vector<M> mats;
    struct N { char name[36]; uint32_t n; std::vector<uint64_t> off; };
    std::vector<N> norms;
    bool ok = false;
};
static axcl_layer_layout g_layout;

static bool axcl_layer_load_layout(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t) sz) { fclose(f); return false; }
    fclose(f);
    if (sz < 16 || memcmp(buf.data(), "AXL4", 4) != 0) return false;
    uint32_t n_mat, n_norm;
    memcpy(&n_mat, buf.data() + 8, 4);
    memcpy(&n_norm, buf.data() + 12, 4);
    if (n_mat > 8 || n_norm > 8) return false;
    size_t p = 16;
    for (uint32_t i = 0; i < n_mat; i++) {
        axcl_layer_layout::M m;
        memcpy(m.name, buf.data() + p, 8); m.name[8] = 0; p += 8;
        memcpy(&m.rows, buf.data() + p, 4); p += 4;
        memcpy(&m.cols, buf.data() + p, 4); p += 4;
        m.off.resize((size_t) m.rows * m.cols);
        memcpy(m.off.data(), buf.data() + p, (size_t) m.rows * m.cols * 8);
        p += (size_t) m.rows * m.cols * 8;
        g_layout.mats.push_back(std::move(m));
    }
    for (uint32_t i = 0; i < n_norm; i++) {
        axcl_layer_layout::N n;
        memcpy(n.name, buf.data() + p, 32); n.name[32] = 0; p += 32;
        memcpy(&n.n, buf.data() + p, 4); p += 4;
        n.off.resize(n.n);
        memcpy(n.off.data(), buf.data() + p, (size_t) n.n * 8);
        p += (size_t) n.n * 8;
        g_layout.norms.push_back(std::move(n));
    }
    g_layout.ok = true;
    return true;
}

static inline uint16_t axcl_f32_to_bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t bits = u >> 16;
    uint32_t rem = u & 0xFFFF;
    uint32_t ru = (rem > 0x8000) || (rem == 0x8000 && (bits & 1));
    return (uint16_t) (bits + ru);
}

static void axcl_patch_matrix(char * eng, const axcl_layer_layout::M & m,
                              const struct ggml_tensor * w) {
    if (w == nullptr) return;
    const int64_t rows = w->ne[1], cols = w->ne[0];
    if ((uint32_t) rows != m.rows || (uint32_t) cols != m.cols) return;
    const struct ggml_type_traits * tr = ggml_get_type_traits(w->type);
    std::vector<float> row(cols);
    for (int64_t r = 0; r < rows; r++) {
        if (w->type == GGML_TYPE_F32) {
            memcpy(row.data(), (const char *) w->data + (size_t) r * w->nb[1], (size_t) cols * 4);
        } else {
            tr->to_float((const void *) ((const char *) w->data + (size_t) r * w->nb[1]), row.data(), cols);
        }
        const uint64_t * offr = m.off.data() + (size_t) r * m.cols;
        for (int64_t c = 0; c < cols; c++) {
            uint64_t o = offr[c];
            if (o == ~0ull) continue;
            uint16_t b = axcl_f32_to_bf16_bits(row[c]);
            eng[o] = (char) (b & 0xFF);
            eng[o + 1] = (char) (b >> 8);
        }
    }
}

static void axcl_patch_norm(char * eng, const axcl_layer_layout::N & n,
                            const struct ggml_tensor * w) {
    if (w == nullptr) return;
    const int64_t cnt = ggml_nelements(w);
    if ((uint32_t) cnt != n.n) return;
    const struct ggml_type_traits * tr = ggml_get_type_traits(w->type);
    std::vector<float> v(cnt);
    if (w->type == GGML_TYPE_F32) memcpy(v.data(), w->data, (size_t) cnt * 4);
    else tr->to_float(w->data, v.data(), cnt);
    for (int64_t i = 0; i < cnt; i++) {
        uint64_t o = n.off[i];
        if (o == ~0ull) continue;
        uint16_t b = axcl_f32_to_bf16_bits(v[i]);
        eng[o] = (char) (b & 0xFF);
        eng[o + 1] = (char) (b >> 8);
    }
}

// engine loads occasionally fail transiently when many loads/unloads churn
// the device manager (observed ~1 per 300 loads in back-to-back runs) —
// retry with a pause before giving up
static axclError axcl_engine_load_file_retry(const char * path, uint64_t * model) {
    axclError rc = axclrtEngineLoadFromFile(path, model);
    // the device manager needs SECONDS to recover when process churn
    // starves it (E2E back-to-back runs) — pace up to ~10s
    for (int attempt = 0; rc != AXCL_SUCC && attempt < 6; attempt++) {
        usleep(1500000 >> attempt); // 1.5s, 750ms, 375ms, ...
        rc = axclrtEngineLoadFromFile(path, model);
    }
    return rc;
}

static bool axcl_layer_load_engines(int n_layer) {
    const char * dir = getenv("GGML_AXCL_LAYER_DIR");
    std::string d = dir ? dir : "/usr/local/share/ggml-axcl/layer";
    g_layer.n_layer = n_layer;
    struct stat st;
    if (stat(d.c_str(), &st) != 0 || (st.st_mode & S_IFDIR) == 0) {
        GGML_LOG_WARN("ggml-axcl: layer engine dir %s missing (whole-layer mode off)\n", d.c_str());
        return false;
    }
    if (n_layer <= 0) {
        // count templates in the directory
        DIR * dp = opendir(d.c_str());
        if (!dp) return false;
        struct dirent * de;
        int cnt = 0;
        while ((de = readdir(dp)) != nullptr) {
            int l;
            if (sscanf(de->d_name, "qwen3_p128_l%d_together.axmodel", &l) == 1) cnt++;
        }
        closedir(dp);
        if (cnt <= 0 || cnt > 64) return false;
        if (getenv("GGML_AXCL_LAYER_MAXL")) {
            int mx = atoi(getenv("GGML_AXCL_LAYER_MAXL"));
            if (mx > 0 && cnt > mx) cnt = mx;
        }
        g_layer.n_layer = n_layer = cnt;
    }
    const char * stop_at = getenv("GGML_AXCL_LAYER_STOP"); // load1|a|kv|zero|shared|mask
    if (stop_at && strncmp(stop_at, "load", 4) == 0 && strcmp(stop_at, "load1") != 0) {
        int want = stop_at[4] ? atoi(stop_at + 4) : n_layer;
        for (int l = 0; l < want && l < n_layer; l++) {
            char p1[600];
            snprintf(p1, sizeof(p1), "%s/qwen3_p128_l%d_together.axmodel", d.c_str(), l);
            uint64_t m1 = 0;
            if (axclrtEngineLoadFromFile(p1, &m1) != AXCL_SUCC) return false;
        }
        GGML_LOG_INFO("ggml-axcl: loadN probe (%d engines only)\n", want);
        return false;
    }
    if (stop_at && strcmp(stop_at, "load1") == 0) {
        char p1[600];
        snprintf(p1, sizeof(p1), "%s/qwen3_p128_l0_together.axmodel", d.c_str());
        uint64_t m1 = 0;
        if (axclrtEngineLoadFromFile(p1, &m1) != AXCL_SUCC) return false;
        GGML_LOG_INFO("ggml-axcl: load1 probe only (model=%llx)\n", (unsigned long long) m1);
        return false;
    }
    // dynamic GGUF patching: when enabled and the weight registry is full,
    // build each layer's engine from the template + GGUF values; cache the
    // patched files keyed by the weight pointers so reloads are free
    const char * gguf_dir = getenv("GGML_AXCL_GGUF_DIR");
    const bool want_gguf = getenv("GGML_AXCL_GGUF") != nullptr && g_gguf_n >= n_layer;
    // GGUF mode: skip loading template engines at init — the weight registry
    // only fills during the first graph's prescan, and the engines loaded
    // now would be swapped out (a full wasted 28-engine load cycle).
    // Shared buffers + mask still get set up so the prescan can arm.
    if (getenv("GGML_AXCL_GGUF") && g_gguf_n == 0 && n_layer <= 0) {
        DIR * dp = opendir(d.c_str());
        int cnt = 0;
        if (dp) {
            struct dirent * de;
            while ((de = readdir(dp)) != nullptr) {
                int l;
                if (sscanf(de->d_name, "qwen3_p128_l%d_together.axmodel", &l) == 1) cnt++;
            }
            closedir(dp);
        }
        if (cnt > 0 && cnt <= 64) g_layer.n_layer = cnt;
        const int T = 2048;
        axclrtMalloc(&g_layer.dx_in, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_idx, 4, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_mask, ((size_t) T * (((size_t) (T + 1) * 2 + 7) & ~(size_t) 7) + 4096),
                     AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_mask_row, (((size_t) (T + 1) * 2 + 7) & ~(size_t) 7), AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_kout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_vout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_yout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        axclrtMalloc(&g_layer.d_yout_alt, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
        {
            const size_t rowb = ((size_t) (T + 1) * 2 + 7) & ~(size_t) 7;
            std::vector<char> m((size_t) T * rowb, 0);
            for (int p = 0; p < T; p++) {
                for (int t = 0; t <= T; t++) {
                    const bool allow = (t < p) || (t == T);
                    float v = allow ? 0.0f : -1e9f;
                    uint32_t u;
                    memcpy(&u, &v, 4);
                    uint16_t b = (uint16_t) (u >> 16);
                    memcpy(&m[(size_t) p * rowb + (size_t) t * 2], &b, 2);
                }
            }
            axclrtMemcpy(g_layer.d_mask, m.data(), m.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
            g_layer.mask_row_bytes = rowb;
        }
        g_layer.ctx_len = T;
        g_layer.loaded = true;
        GGML_LOG_INFO("ggml-axcl: GGUF mode — templates deferred (%d layers)\n", g_layer.n_layer);
        return true;
    }
    char cache_dir[512];
    if (want_gguf) {
        snprintf(cache_dir, sizeof(cache_dir), "%s", gguf_dir ? gguf_dir : "/tmp/axcl-gguf");
        mkdir(cache_dir, 0755);
    }
    char p[600];
    for (int l = 0; l < n_layer; l++) {
        if (want_gguf) {
            // layout is loaded once; patch this layer from its stashed tensors
            static bool layout_loaded = false;
            if (!layout_loaded) {
                const char * lp = getenv("GGML_AXCL_LAYOUT");
                char lpath[600];
                snprintf(lpath, sizeof(lpath), "%s", lp ? lp : "/usr/local/share/ggml-axcl/layer/layout_v4.bin");
                if (!axcl_layer_load_layout(lpath)) {
                    GGML_LOG_ERROR("ggml-axcl: layout %s failed to load; GGUF patching off\n", lpath);
                    return false;
                }
                layout_loaded = true;
            }
            char tpl[600], dst[600];
            snprintf(tpl, sizeof(tpl), "%s/qwen3_p128_l0_together.axmodel", d.c_str());
            // cache key: FNV-1a over sampled bytes of each weight tensor —
            // the same GGUF reuses patched engines across processes
            uint64_t h = 1469598103934665603ull;
            for (int ti = 0; ti < 11; ti++) {
                const struct ggml_tensor * wt = g_gguf[l].t[ti];
                if (wt == nullptr) continue;
                size_t nb = ggml_nbytes(wt);
                size_t step = nb / 512 ? nb / 512 : 1;
                for (size_t o = 0; o < nb; o += step) {
                    h ^= ((const unsigned char *) wt->data)[o];
                    h *= 1099511628211ull;
                }
            }
            snprintf(dst, sizeof(dst), "%s/l%d_%016llx.axmodel", cache_dir, l,
                     (unsigned long long) h);
            FILE * probe = fopen(dst, "rb");
            bool cache_hit = (probe != nullptr);
            if (probe) fclose(probe);
            FILE * tf = nullptr;
            if (!cache_hit) tf = fopen(tpl, "rb");
            if (!cache_hit) {
                if (!tf) return false;
                fseek(tf, 0, SEEK_END);
                long sz = ftell(tf);
                fseek(tf, 0, SEEK_SET);
                std::vector<char> eng(sz);
                if (fread(eng.data(), 1, sz, tf) != (size_t) sz) { fclose(tf); return false; }
                fclose(tf);
                for (size_t mi = 0; mi < g_layout.mats.size(); mi++) {
                    axcl_patch_matrix(eng.data(), g_layout.mats[mi], g_gguf[l].t[mi]);
                }
                for (size_t ni = 0; ni < g_layout.norms.size(); ni++) {
                    axcl_patch_norm(eng.data(), g_layout.norms[ni], g_gguf[l].t[7 + ni]);
                }
                FILE * of = fopen(dst, "wb");
                if (!of) return false;
                fwrite(eng.data(), 1, sz, of);
                fclose(of);
                GGML_LOG_INFO("ggml-axcl: layer %d patched from GGUF (%ld bytes)\n", l, sz);
            } else {
                GGML_LOG_INFO("ggml-axcl: layer %d cache hit\n", l);
            }
            snprintf(p, sizeof(p), "%s", dst);
            g_layer_from_gguf = true;
        } else {
            snprintf(p, sizeof(p), "%s/qwen3_p128_l%d_together.axmodel", d.c_str(), l);
        }
        axcl_layer_engine * e = &g_layer.eng[l];
        if (axcl_engine_load_file_retry(p, &e->model) != AXCL_SUCC) {
            GGML_LOG_WARN("ggml-axcl: layer engine %s failed to load\n", p);
            return false;
        }
        if (stop_at && strcmp(stop_at, "a1") == 0) continue;
        axclrtEngineGetIOInfo(e->model, &e->info);
        if (stop_at && strcmp(stop_at, "a2") == 0) continue;
        axclrtEngineCreateIO(e->info, &e->io);
        if (stop_at && strcmp(stop_at, "a3") == 0) continue;
        axclrtEngineCreateContext(e->model, &e->ectx);
        if (stop_at && strcmp(stop_at, "a4") == 0) continue;
        e->ik = axclrtEngineGetInputIndexByName(e->info, "K_cache");
        if (l == 0 && e->ik >= 0) {
            axclrtEngineIODims dims;
            if (axclrtEngineGetInputDims(e->info, 0, (uint32_t) e->ik, &dims) == AXCL_SUCC &&
                dims.dimCount >= 2 && dims.dims[1] > 0 && dims.dims[1] <= 8192) {
                g_layer.ctx_len = dims.dims[1];
            }
        }
        e->iv = axclrtEngineGetInputIndexByName(e->info, "V_cache");
        e->ii = axclrtEngineGetInputIndexByName(e->info, "indices");
        e->ix = axclrtEngineGetInputIndexByName(e->info, "input");
        e->im = axclrtEngineGetInputIndexByName(e->info, "mask");
        e->iko = axclrtEngineGetOutputIndexByName(e->info, "K_cache_out");
        e->ivo = axclrtEngineGetOutputIndexByName(e->info, "V_cache_out");
        e->iyo = axclrtEngineGetOutputIndexByName(e->info, "output");
        if (e->ik < 0 || e->iv < 0 || e->ii < 0 || e->ix < 0 || e->im < 0 ||
            e->iko < 0 || e->ivo < 0 || e->iyo < 0) {
            GGML_LOG_WARN("ggml-axcl: layer %d IO names not found\n", l);
            return false;
        }
        if (l == 0) {
            int32_t ngrp = 1;
            if (axclrtEngineGetShapeGroupsCount(e->info, &ngrp) == AXCL_SUCC && ngrp > 1) {
                g_layer.n_groups = ngrp;
                GGML_LOG_INFO("ggml-axcl: %d shape groups (batched prefill ladder available)\n", ngrp);
            }
        }
        if (stop_at && strcmp(stop_at, "ctx") == 0) {
            axclrtEngineGetIOInfo(e->model, &e->info);
            axclrtEngineCreateIO(e->info, &e->io);
            axclrtEngineCreateContext(e->model, &e->ectx);
            continue;
        }
        if (stop_at && strcmp(stop_at, "kvm") == 0) {
            axclrtEngineGetIOInfo(e->model, &e->info);
            axclrtEngineCreateIO(e->info, &e->io);
            axclrtEngineCreateContext(e->model, &e->ectx);
            const size_t kvb2 = (size_t) g_layer.ctx_len * 1024 * 2;
            axclrtMalloc(&e->dk, kvb2, AXCL_MEM_MALLOC_HUGE_FIRST);
            axclrtMalloc(&e->dv, kvb2, AXCL_MEM_MALLOC_HUGE_FIRST);
            continue;
        }
        if (stop_at && stop_at[0] == 'a') continue;
        const size_t kv_bytes = (size_t) g_layer.ctx_len * 1024 * 2;
        if (axclrtMalloc(&e->dk, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&e->dv, kv_bytes, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC) {
            GGML_LOG_WARN("ggml-axcl: layer %d KV alloc failed\n", l);
            return false;
        }
        if (stop_at && strcmp(stop_at, "kv") == 0) continue;
        // zero-fill: uninitialized garbage can be NaN-shaped, and NaN scores
        // poison the softmax even through the -inf mask (NaN + -inf = NaN)
        {
            static std::vector<char> zeros;
            zeros.resize(1 << 20, 0);
            for (size_t off = 0; off < kv_bytes; off += zeros.size()) {
                size_t n = std::min(zeros.size(), kv_bytes - off);
                axclrtMemcpy((char *) e->dk + off, zeros.data(), n, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtMemcpy((char *) e->dv + off, zeros.data(), n, AXCL_MEMCPY_HOST_TO_DEVICE);
            }
        }
        e->wm = 0;
    }
    if (stop_at && (strcmp(stop_at, "zero") == 0 || strcmp(stop_at, "kvm") == 0 ||
                    strcmp(stop_at, "ctx") == 0)) { return false; }
    // shared IO buffers
    const int T = g_layer.ctx_len;
    axclrtMalloc(&g_layer.dx_in, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_idx, 2048 * 4 + 8, AXCL_MEM_MALLOC_HUGE_FIRST);
    {
        std::vector<uint32_t> idxs(2048);
        for (int i = 0; i < 2048; i++) idxs[i] = (uint32_t) i;
        axclrtMemcpy(g_layer.d_idx, idxs.data(), 2048 * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    }
    axclrtMalloc(&g_layer.d_mask, ((size_t) T * (((size_t) (T + 1) * 2 + 7) & ~(size_t) 7) + 4096), AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_mask_row, (((size_t) (T + 1) * 2 + 7) & ~(size_t) 7), AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_kout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_vout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_yout, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    axclrtMalloc(&g_layer.d_yout_alt, 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST);
    if (stop_at && strcmp(stop_at, "shared") == 0) { g_layer.loaded = true; return true; }
    if (stop_at && strcmp(stop_at, "maskup") == 0) {
        const size_t rowbx = ((size_t) (T + 1) * 2 + 7) & ~(size_t) 7;
        std::vector<char> mx((size_t) T * rowbx, 0);
        axclrtMemcpy(g_layer.d_mask, mx.data(), mx.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
        g_layer.mask_row_bytes = rowbx;
        g_layer.loaded = true;
        return true;
    }
    // precompute mask rows. mask layout: [0..T-1] = cache slots, [T] = the
    // engine's appended SELF slot (the current token's own K/V). Row p:
    // allow cache[0..p) + the self slot; mask stale cache[p..T).
    // Rows are padded to an 8-byte stride: card input buffers need sane
    // alignment (a 4098-byte stride broke every odd position).
    {
        const size_t rowb = ((size_t) (T + 1) * 2 + 7) & ~(size_t) 7;
        std::vector<char> m((size_t) T * rowb, 0);
        for (int p = 0; p < T; p++) {
            for (int t = 0; t <= T; t++) {
                const bool allow = (t < p) || (t == T);
                float v = allow ? 0.0f : -1e9f;
                uint32_t u;
                memcpy(&u, &v, 4);
                uint16_t b = (uint16_t) (u >> 16);
                memcpy(&m[(size_t) p * rowb + (size_t) t * 2], &b, 2);
            }
        }
        if (axclrtMemcpy(g_layer.d_mask, m.data(), m.size(), AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: mask table upload failed\n");
            return false;
        }
        {
            std::vector<char> chk(rowb);
            axclrtMemcpy(chk.data(), (char *) g_layer.d_mask + rowb, rowb, AXCL_MEMCPY_DEVICE_TO_HOST);
            if (memcmp(chk.data(), &m[rowb], rowb) != 0) {
                GGML_LOG_ERROR("ggml-axcl: mask table verify FAILED\n");
                return false;
            }
        }
        g_layer.mask_row_bytes = rowb;
    }
    g_layer.loaded = true;
    GGML_LOG_INFO("ggml-axcl: %d whole-layer engines ready (ctx %d)\n", n_layer, T);
    return true;
}

// materialize deferred host-cache rows for layer l (rows [host_wm, wm)).
// Decode never reads the host KV while the engines own the graph, so this
// runs only at flush points (resync, periodic, bail, non-armed graph end).
// Rows are contiguous in the device cache -> one D2H per 64-row chunk per side.
static void axcl_layer_flush_kv(int l) {
    axcl_layer_engine * e = &g_layer.eng[l];
    if (g_layer.host_wm[l] >= e->wm) return;
    struct ggml_tensor * kb = g_layer.kbase[l];
    struct ggml_tensor * vb = g_layer.vbase[l];
    if (kb == nullptr || vb == nullptr || g_layer.pin_row == nullptr) {
        g_layer.host_wm[l] = e->wm; // nowhere to write; drop the range
        return;
    }
    if (g_layer.pin_rows == nullptr) {
        if (axclrtMallocHost(&g_layer.pin_rows, 64 * 2048) != AXCL_SUCC) {
            g_layer.host_wm[l] = e->wm;
            return;
        }
    }
    const int T = g_layer.ctx_len;
    int lo = g_layer.host_wm[l];
    const int lim = std::min(e->wm, T);
    while (lo < lim) {
        const int rows = std::min(lim - lo, 64);
        for (int side = 0; side < 2; side++) {
            struct ggml_tensor * v = side ? vb : kb;
            char * base = (char *) (side ? e->dv : e->dk);
            axclrtMemcpy(g_layer.pin_rows, base + (size_t) lo * 1024 * 2,
                         (size_t) rows * 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
            const uint16_t * src = (const uint16_t *) g_layer.pin_rows;
            for (int p = 0; p < rows; p++) {
                char * dst = (char *) v->data + (size_t) (lo + p) * v->nb[1];
                if (v->type == GGML_TYPE_BF16) {
                    memcpy(dst, src + (size_t) p * 1024, 2048);
                } else if (v->type == GGML_TYPE_F16) {
                    float f[1024];
                    axcl_bf16_to_f32(src + (size_t) p * 1024, f, 1024);
                    ggml_fp16_t * h = (ggml_fp16_t *) dst;
                    for (int i = 0; i < 1024; i++) h[i] = GGML_COMPUTE_FP32_TO_FP16(f[i]);
                } else if (v->type == GGML_TYPE_F32) {
                    axcl_bf16_to_f32(src + (size_t) p * 1024, (float *) dst, 1024);
                }
            }
        }
        lo += rows;
    }
    g_layer.host_wm[l] = std::max(g_layer.host_wm[l], lim);
}

static void axcl_layer_flush_kv_all() {
    for (int l = 0; l < g_layer.n_layer; l++) axcl_layer_flush_kv(l);
}

// upload host cache rows [0, pos) into layer l's device caches (bf16)
static void axcl_layer_resync(int l, int pos) {
    if (g_layer.kbase[l] == nullptr || g_layer.vbase[l] == nullptr || pos <= 0) return;
    axcl_layer_flush_kv(l); // device rows beyond host_wm must land before host wins
    axcl_layer_engine * e = &g_layer.eng[l];
    const int T = g_layer.ctx_len;
    std::vector<float> row(1024);
    std::vector<uint16_t> buf((size_t) T * 1024);
    for (int side = 0; side < 2; side++) {
        struct ggml_tensor * v = side ? g_layer.vbase[l] : g_layer.kbase[l];
        const int64_t nb1 = v->nb[1];
        for (int t = 0; t < pos && t < T; t++) {
            axcl_kv_fetch((const char *) v->data + (size_t) t * nb1, v->type, row.data(), 1024);
            for (int i = 0; i < 1024; i++) {
                uint32_t u;
                memcpy(&u, &row[i], 4);
                buf[(size_t) t * 1024 + i] = (uint16_t) (u >> 16);
            }
        }
        axclrtMemcpy(side ? e->dv : e->dk, buf.data(), (size_t) pos * 1024 * 2,
                     AXCL_MEMCPY_HOST_TO_DEVICE);
    }
    e->wm = pos;
    g_layer.host_wm[l] = pos; // host cache is authoritative through pos now
}

// run one whole-layer decode step for layer l at KV position pos.
// d_hidden (device bf16) is consumed; the new hidden lands in d_yout
// (kept device-resident). The new K/V row lands directly in this layer's
// device cache (output bound into the cache row); the host-cache write-back
// is DEFERRED via the host_wm watermark.
static bool axcl_layer_run(int l, int pos,
                           struct ggml_tensor * host_k_view, struct ggml_tensor * host_v_view,
                           int64_t k_nb1, int64_t v_nb1, void ** out_hidden_dev) {
    GGML_UNUSED(host_k_view); GGML_UNUSED(host_v_view);
    GGML_UNUSED(k_nb1); GGML_UNUSED(v_nb1);
    axcl_layer_engine * e = &g_layer.eng[l];
    const int T = g_layer.ctx_len;
    if (pos >= T) {
        axcl_layer_flush_kv_all();
        return false;
    }
    if (e->dk == nullptr || e->dv == nullptr || g_layer.d_idx == nullptr ||
        g_layer.d_hidden == nullptr || g_layer.d_mask_row == nullptr ||
        g_layer.d_kout == nullptr || g_layer.d_vout == nullptr) {
        static int niln = 0;
        if (niln++ < 5) {
            GGML_LOG_ERROR("ggml-axcl: layer %d null bind: dk=%p dv=%p idx=%p hid=%p mrow=%p ko=%p vo=%p\n",
                           l, e->dk, e->dv, g_layer.d_idx, g_layer.d_hidden, g_layer.d_mask_row,
                           g_layer.d_kout, g_layer.d_vout);
        }
        return false;
    }
    // lazy pinned staging (first call)
    if (g_layer.pin_idx == nullptr) {
        if (axclrtMallocHost(&g_layer.pin_idx, 8) != AXCL_SUCC ||
            axclrtMallocHost(&g_layer.pin_row, 2048) != AXCL_SUCC ||
            axclrtMallocHost(&g_layer.pin_hidden, 2048) != AXCL_SUCC) {
            GGML_LOG_ERROR("ggml-axcl: pinned staging alloc failed\n");
            return false;
        }
    }
    // per-TOKEN refresh: idx + mask row are identical for every layer at the
    // same position — upload once per new pos, not once per layer
    if (g_layer.synced_pos != pos) {
        *(uint32_t *) g_layer.pin_idx = (uint32_t) pos;
        axclrtMemcpy(g_layer.d_idx, g_layer.pin_idx, 4, AXCL_MEMCPY_HOST_TO_DEVICE);
        axclrtMemcpy(g_layer.d_mask_row, (char *) g_layer.d_mask + (size_t) pos * g_layer.mask_row_bytes,
                     (size_t) (T + 1) * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
        g_layer.synced_pos = pos;
    }
    // periodic deferred write-back flush: bound the device-authoritative
    // window so save-state / crash paths never lose more than 32 rows
    if (l == 0 && (pos & 31) == 31) axcl_layer_flush_kv_all();

    const size_t kv_bytes = (size_t) T * 1024 * 2;
    // static bindings once per engine; per-call: hidden ping-pong + K/V rows
    if (!e->bound) {
        axclrtEngineSetInputBufferByIndex(e->io, e->ik, e->dk, kv_bytes);
        axclrtEngineSetInputBufferByIndex(e->io, e->iv, e->dv, kv_bytes);
        axclrtEngineSetInputBufferByIndex(e->io, e->ii, g_layer.d_idx, 4);
        axclrtEngineSetInputBufferByIndex(e->io, e->im, g_layer.d_mask_row, (size_t) (T + 1) * 2);
        e->bound = true;
    }
    // K/V outputs land straight in the cache row (2KB-aligned; the row is
    // mask-protected from this call's own attention reads). GGML_AXCL_KV_INPLACE=0
    // restores the old kout-buffer + scatter path for A/B testing.
    static const bool kv_inplace = getenv("GGML_AXCL_KV_INPLACE") == nullptr ||
                                   atoi(getenv("GGML_AXCL_KV_INPLACE")) != 0;
    void * krow = kv_inplace ? (char *) e->dk + (size_t) pos * 1024 * 2 : g_layer.d_kout;
    void * vrow = kv_inplace ? (char *) e->dv + (size_t) pos * 1024 * 2 : g_layer.d_vout;
    axclrtEngineSetInputBufferByIndex(e->io, e->ix, g_layer.d_hidden, 1024 * 2);
    g_layer.y_next = (g_layer.d_hidden == g_layer.d_yout) ? g_layer.d_yout_alt : g_layer.d_yout;
    axclrtEngineSetOutputBufferByIndex(e->io, e->iko, krow, 1024 * 2);
    axclrtEngineSetOutputBufferByIndex(e->io, e->ivo, vrow, 1024 * 2);
    axclrtEngineSetOutputBufferByIndex(e->io, e->iyo, g_layer.y_next, 1024 * 2);

    uint64_t t0 = axcl_us();
    // single-token decode chains pipeline on one stream: the hidden
    // double-buffer + per-layer cache rows have no cross-call hazards, so
    // the 28 executes enqueue back-to-back and the sync happens once after
    // layer 27. Prefill (m>1) and diagnostics stay synchronous.
    static axclrtStream chain_stream = nullptr;
    static const bool use_chain_stream = getenv("GGML_AXCL_STREAM") != nullptr;
    if (use_chain_stream && g_layer.n_tok == 1 && !getenv("GGML_AXCL_CHECKSUM") &&
        !getenv("GGML_AXCL_DUMPSTATE")) {
        if (chain_stream == nullptr) axclrtCreateStream(&chain_stream);
        if (l == 0) g_layer.chain_t0 = t0;
        if (axclrtEngineExecuteAsync(e->model, e->ectx, 0, e->io, chain_stream) == AXCL_SUCC) {
            g_layer.calls++;
            // eng time accumulates at the sync point (see below)
            if (l == g_layer.n_layer - 1) {
                axclrtSynchronizeStream(chain_stream);
                g_layer.us += axcl_us() - g_layer.chain_t0;
            }
        } else {
            GGML_LOG_ERROR("ggml-axcl: layer %d async execute failed\n", l);
            return false;
        }
    } else if (axclrtEngineExecute(e->model, e->ectx, 0, e->io) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: layer %d execute failed\n", l);
        return false;
    } else {
        g_layer.us += axcl_us() - t0;
        g_layer.calls++;
    }

    if (!kv_inplace) {
        // scatter the new row into the device cache
        axclrtMemcpy((char *) e->dk + (size_t) pos * 1024 * 2, g_layer.d_kout, 1024 * 2,
                     AXCL_MEMCPY_DEVICE_TO_DEVICE);
        axclrtMemcpy((char *) e->dv + (size_t) pos * 1024 * 2, g_layer.d_vout, 1024 * 2,
                     AXCL_MEMCPY_DEVICE_TO_DEVICE);
    }
    if (pos >= e->wm) e->wm = pos + 1;

    // host cache write-back: DEFERRED by default (llama.cpp never reads these
    // rows while the engines own decode — attention runs on-card). GGML_AXCL_KVWB=now
    // restores the old per-call write-back for A/B testing.
    static const char * wb_env = getenv("GGML_AXCL_KVWB");
    if (wb_env && strcmp(wb_env, "now") == 0) {
        axcl_layer_flush_kv(l);
    }
    if (out_hidden_dev) *out_hidden_dev = g_layer.y_next;
    return true;
}

// batched prefill chunk: run 128 tokens at absolute position p of layer l
// through shape-group (1 + p/128) — the vendor ladder (groups 1..9, prefix
// grows 0..1024). Input rows are staged into d_chunk_in (offset bindings are
// unreliable on this runtime); K/V rows land in-place in the cache; the
// chunk's hidden output lands in d_chunk_out. Returns false if the ladder
// can't serve this chunk (caller falls back to per-token group 0).
static bool axcl_layer_run_chunk(int l, int p, int ntok) {
    axcl_layer_engine * e = &g_layer.eng[l];
    const int g = 1 + p / 128;
    if (g < 1 || g >= g_layer.n_groups || ntok != 128) return false;
    const int T = g_layer.ctx_len;
    if (p + ntok > T) return false;
    // lazy chunk buffers
    if (g_layer.d_chunk_in == nullptr) {
        if (axclrtMalloc(&g_layer.d_chunk_in, 128 * 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&g_layer.d_chunk_out, 128 * 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&g_layer.d_idx_m, 128 * 4, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&g_layer.d_mask_m, (size_t) 128 * 1152 * 2, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            // K/V_out must land in DEDICATED buffers — binding outputs into
            // cache rows at byte offsets silently mis-executes on this
            // runtime (phase_c_refcheck: offset output binds are the reason
            // the chunk ladder was ever believed broken)
            axclrtMalloc(&g_layer.d_chunk_ko, 128 * 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMalloc(&g_layer.d_chunk_vo, 128 * 1024 * 2, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
            axclrtMallocHost(&g_layer.pin_idx_m, 128 * 4) != AXCL_SUCC ||
            axclrtMallocHost(&g_layer.pin_mask_m, (size_t) 128 * 1152 * 2) != AXCL_SUCC) {
            g_layer.n_groups = 1; // batched prefill unavailable from now on
            return false;
        }
    }
    // dedicated IO handle PER shape group (the canonical runner uses one
    // handle per group; sharing one across groups mis-binds internals)
    if (g >= 12 || e->io_chunk[g] == nullptr) {
        if (g >= 12 ||
            axclrtEngineCreateIO(e->info, &e->io_chunk[g]) != AXCL_SUCC) {
            g_layer.n_groups = 1;
            return false;
        }
    }
    // indices: rope positions p..p+127
    for (int i = 0; i < ntok; i++) ((uint32_t *) g_layer.pin_idx_m)[i] = (uint32_t) (p + i);
    axclrtMemcpy(g_layer.d_idx_m, g_layer.pin_idx_m, (size_t) ntok * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
    // causal mask: token p+i attends columns j <= p+i, width p+128.
    // GGML_AXCL_CMSK=strict excludes the diagonal (decode's self-attention
    // runs through the engine's internal self slot, not the mask)
    static const bool strict_diag = getenv("GGML_AXCL_CMSK") != nullptr &&
                                    strcmp(getenv("GGML_AXCL_CMSK"), "strict") == 0;
    const int w = p + ntok;
    uint16_t * m = (uint16_t *) g_layer.pin_mask_m;
    for (int i = 0; i < ntok; i++) {
        const int lim = strict_diag ? (p + i - 1) : (p + i);
        for (int j = 0; j < w; j++) {
            float v = (j <= lim) ? 0.0f : -1e9f;
            uint32_t u;
            memcpy(&u, &v, 4);
            m[(size_t) i * w + j] = (uint16_t) (u >> 16);
        }
    }
    axclrtMemcpy(g_layer.d_mask_m, m, (size_t) ntok * w * 2, AXCL_MEMCPY_HOST_TO_DEVICE);
    // cache prefix: group 1 reads a single dummy row (masked out)
    const size_t kvb = (size_t) (p ? p : 1) * 1024 * 2;
    axclrtEngineSetInputBufferByIndex(e->io_chunk[g], e->ik, e->dk, kvb);
    axclrtEngineSetInputBufferByIndex(e->io_chunk[g], e->iv, e->dv, kvb);
    axclrtEngineSetInputBufferByIndex(e->io_chunk[g], e->ii, g_layer.d_idx_m, (size_t) ntok * 4);
    axclrtEngineSetInputBufferByIndex(e->io_chunk[g], e->ix, g_layer.d_chunk_in, (size_t) ntok * 1024 * 2);
    axclrtEngineSetInputBufferByIndex(e->io_chunk[g], e->im, g_layer.d_mask_m, (size_t) ntok * w * 2);
    axclrtEngineSetOutputBufferByIndex(e->io_chunk[g], e->iko, g_layer.d_chunk_ko,
                                       (size_t) ntok * 1024 * 2);
    axclrtEngineSetOutputBufferByIndex(e->io_chunk[g], e->ivo, g_layer.d_chunk_vo,
                                       (size_t) ntok * 1024 * 2);
    axclrtEngineSetOutputBufferByIndex(e->io_chunk[g], e->iyo, g_layer.d_chunk_out, (size_t) ntok * 1024 * 2);

    uint64_t t0 = axcl_us();
    if (axclrtEngineExecute(e->model, e->ectx, (uint32_t) g, e->io_chunk[g]) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: layer %d chunk execute (group %d) failed\n", l, g);
        return false;
    }
    // scatter K/V rows from staging into their cache slots (outputs execute
    // into dedicated buffers; offset binds into the cache mis-execute)
    axclrtMemcpy((char *) e->dk + (size_t) p * 1024 * 2, g_layer.d_chunk_ko,
                 (size_t) ntok * 1024 * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
    axclrtMemcpy((char *) e->dv + (size_t) p * 1024 * 2, g_layer.d_chunk_vo,
                 (size_t) ntok * 1024 * 2, AXCL_MEMCPY_DEVICE_TO_DEVICE);
    g_layer.us += axcl_us() - t0;
    g_layer.calls++;
    if (p + ntok > e->wm) e->wm = p + ntok;
    return true;
}

// cross-split QKV: detect 3 MUL_MATs sharing src1 (shape-based, no norm)
static struct ggml_tensor * xqkv_q[3] = {nullptr, nullptr, nullptr};
static const void * xqkv_src1 = nullptr; // shared src1 of the projections
static int xqkv_count = 0;
static float xqkv_h[1024];               // X snapshot at stash time
static uint64_t xqkv_reuse_hits = 0;

static bool axcl_qkv_try_flush() {
    bool ok = false;
    if (xqkv_count == 3 && g_qkv.model != 0) {
        // h = the shared src1 (already normed by the host-side RMS_NORM)
        axclrtMemcpy(g_qkv.dev_in[0], xqkv_h, 1024*4, AXCL_MEMCPY_HOST_TO_DEVICE);
        static const bool qkv_swap = getenv("GGML_AXCL_QKV_SWAP") != nullptr;
        struct ggml_tensor * kt = qkv_swap ? xqkv_q[2] : xqkv_q[1];
        struct ggml_tensor * vt = qkv_swap ? xqkv_q[1] : xqkv_q[2];
        void * dqw = axcl_fused_stage_w(&g_qkv, xqkv_q[0]->src[0], (size_t)1024*2048*4);
        void * dkw = axcl_fused_stage_w(&g_qkv, kt->src[0], (size_t)1024*1024*4);
        void * dvw = axcl_fused_stage_w(&g_qkv, vt->src[0], (size_t)1024*1024*4);
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
                axclrtMemcpy(kt->data, g_qkv.dev_out[1], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
                axclrtMemcpy(vt->data, g_qkv.dev_out[2], 1024*4, AXCL_MEMCPY_DEVICE_TO_HOST);
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
    // dirty-KV guard: if this graph runs ZERO layer-engine calls (pure CPU
    // fallback — e.g. a graph variant that reads the KV cache directly), the
    // deferred host write-back must materialize first
    const uint64_t calls_at_entry = g_layer.calls;
    // whole-layer prescan: arm only for single-token decode graphs with the
    // expected anchor structure (embedding GET_ROWS + one [1024->2048]
    // q_proj per layer + final norm). Anything else (prefill, batch, shift)
    // takes the legacy per-op path.
    static const bool layer_env = getenv("GGML_AXCL_LAYER") != nullptr;
    bool layer_armed = false;
    if (layer_env && g_layer.loaded) {
        int anchors = 0, gets = 0;
        bool all_m1 = true;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_GET_ROWS && n->src[0] && n->src[0]->ne[0] == 1024 &&
                n->src[0]->ne[1] > 32768) {
                gets++;
            } else if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[0]->ne[0] == 1024 &&
                       n->src[0]->ne[1] == 2048 && n->src[1]) {
                anchors++;
                if (n->src[1]->ne[1] != 1) all_m1 = false;
            }
        }
        if (getenv("GGML_AXCL_LAYER_DEBUG")) {
            static int pd = 0;
            if (pd < 4) { pd++;
                fprintf(stderr, "[layer-prescan] nodes=%d gets=%d anchors=%d (need %d) all_m1=%d ops:",
                        cgraph->n_nodes, gets, anchors, g_layer.n_layer, (int) all_m1);
                for (int j = 0; j < cgraph->n_nodes && j < 40; j++) {
                    fprintf(stderr, " %s", ggml_op_name(cgraph->nodes[j]->op));
                }
                fprintf(stderr, "\n");
            }
        }
        const bool armable = gets == 1 && anchors == g_layer.n_layer;
        (void) all_m1;
        if (armable) {
            layer_armed = true;
            g_layer.armed = true;
            g_layer.next_layer = 0;
            // GGUF weight registry: stash layer tensors by shape. Layer order
            // = anchor order; matrices by (rows, cols); norms by size.
            if (getenv("GGML_AXCL_GGUF") && !g_layer_from_gguf) {
                int anchor_i = -1;
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[0]->op == GGML_OP_NONE &&
                        n->src[0]->ne[0] == 1024 &&
                        n->src[0]->ne[1] == 2048) {
                        anchor_i++;
                        if (anchor_i >= 64) break;
                        g_gguf[anchor_i].t[0] = n->src[0]; // q
                    }
                }
                // fill k/v/o/gate/up/down by shape on the SAME layer index
                int qi = -1;
                static bool seq_dumped = false;
                if (getenv("GGML_AXCL_GGUF_DEBUG") && !seq_dumped) {
                    seq_dumped = true;
                    fprintf(stderr, "[mm-seq] leaf matmuls (ne0 x ne1, src0-op):\n");
                    for (int i = 0; i < cgraph->n_nodes; i++) {
                        struct ggml_tensor * n = cgraph->nodes[i];
                        if (n->op == GGML_OP_MUL_MAT && n->src[0]) {
                            fprintf(stderr, "  [%lld x %lld] op=%s\n",
                                    (long long) n->src[0]->ne[0], (long long) n->src[0]->ne[1],
                                    ggml_op_name(n->src[0]->op));
                        }
                    }
                }
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op != GGML_OP_MUL_MAT || !n->src[0] || n->src[0]->op != GGML_OP_NONE ||
                        n->src[0]->ne[0] != 1024) continue;
                    const int64_t nr = n->src[0]->ne[1];
                    if (nr == 2048) { qi++; continue; }
                    if (qi < 0 || qi >= 64) continue;
                    if (getenv("GGML_AXCL_GGUF_DEBUG") && qi < 2) {
                        fprintf(stderr, "[kv-fill] qi=%d nr=%lld data=%p\n", qi, (long long) nr,
                                n->src[0]->data);
                    }
                    if (nr == 1024) {
                        // node order is q,v,k (k's norm/rope chain is longer) —
                        // collect the pair, then assign k = lower address (k_proj
                        // is allocated before v_proj in the model block)
                        if (g_gguf[qi].t[1] == nullptr) g_gguf[qi].t[1] = n->src[0];
                        else if (g_gguf[qi].t[2] == nullptr) g_gguf[qi].t[2] = n->src[0];
                    }
                    else if (nr == 3072 && g_gguf[qi].t[4] == nullptr) g_gguf[qi].t[4] = n->src[0]; // gate
                    else if (nr == 3072 && g_gguf[qi].t[5] == nullptr) g_gguf[qi].t[5] = n->src[0]; // up
                }
                // finalize k/v: k_proj is the lower-address tensor
                for (int q2 = 0; q2 < 64; q2++) {
                    if (g_gguf[q2].t[1] && g_gguf[q2].t[2] &&
                        (char *) g_gguf[q2].t[1]->data > (char *) g_gguf[q2].t[2]->data) {
                        const struct ggml_tensor * tmp = g_gguf[q2].t[1];
                        g_gguf[q2].t[1] = g_gguf[q2].t[2];
                        g_gguf[q2].t[2] = tmp;
                    }
                }
                // o (k=2048), down (k=3072): MUL_MATs with ne[0] != 1024
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op != GGML_OP_MUL_MAT || !n->src[0] || n->src[0]->op != GGML_OP_NONE) continue;
                    const int64_t kc = n->src[0]->ne[0], nr = n->src[0]->ne[1];
                    if (kc == 2048 && nr == 1024) {
                        // o_proj appears after its layer's attention; map by
                        // position: count layers whose q we've seen
                        for (int q2 = 0; q2 < 64; q2++) {
                            if (g_gguf[q2].t[0] && g_gguf[q2].t[3] == nullptr &&
                                (q2 == 0 || g_gguf[q2 - 1].t[3])) {
                                g_gguf[q2].t[3] = n->src[0];
                                break;
                            }
                        }
                    } else if (kc == 3072 && nr == 1024) {
                        for (int q2 = 0; q2 < 64; q2++) {
                            if (g_gguf[q2].t[0] && g_gguf[q2].t[6] == nullptr &&
                                (q2 == 0 || g_gguf[q2 - 1].t[6])) {
                                g_gguf[q2].t[6] = n->src[0];
                                break;
                            }
                        }
                    }
                }
                // norms: RMS_NORM weight muls near anchors (1024 = ln/post, 128 = q/k norm)
                int ni2 = -1;
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op == GGML_OP_MUL && n->ne[0] == 1024 && n->ne[1] == 1 &&
                        n->src[1] && n->src[1]->ne[0] == 1024 && n->src[1]->ne[1] == 1) {
                        // norm-gain mul: src1 = weight
                        struct ggml_tensor * wgt = (n->src[1]->ne[0] == 1024 && n->src[1]->nb[1] == 0) ? n->src[1] : nullptr;
                        (void) wgt;
                    }
                }
                (void) ni2;
                int nl_seen = 0;
                for (int i = 0; i < cgraph->n_nodes && nl_seen < 64; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op != GGML_OP_MUL || n->ne[0] != 1024) continue;
                    // norm gain: leaf [1024,1] weight whose other side is an
                    // RMS_NORM output (chain-mode pattern)
                    struct ggml_tensor * wgt = nullptr;
                    if (n->src[1] && n->src[1]->op == GGML_OP_NONE && n->src[1]->ne[0] == 1024 &&
                        n->src[1]->ne[1] == 1 && n->src[0] && n->src[0]->op == GGML_OP_RMS_NORM) wgt = n->src[1];
                    else if (n->src[0] && n->src[0]->op == GGML_OP_NONE && n->src[0]->ne[0] == 1024 &&
                             n->src[0]->ne[1] == 1 && n->src[1] && n->src[1]->op == GGML_OP_RMS_NORM) wgt = n->src[0];
                    if (!wgt) continue;
                    if (g_gguf[nl_seen].t[7] == nullptr) g_gguf[nl_seen].t[7] = wgt;
                    else if (g_gguf[nl_seen].t[8] == nullptr) { g_gguf[nl_seen].t[8] = wgt; nl_seen++; }
                }
                // q_norm/k_norm (128)
                int qn_seen = 0;
                for (int i = 0; i < cgraph->n_nodes && qn_seen < 64; i++) {
                    struct ggml_tensor * n = cgraph->nodes[i];
                    if (n->op != GGML_OP_MUL || n->ne[0] != 128) continue;
                    struct ggml_tensor * wgt = nullptr;
                    if (n->src[1] && n->src[1]->op == GGML_OP_NONE && n->src[1]->ne[1] == 1 &&
                        ggml_nelements(n->src[1]) == 128 && n->src[0] &&
                        n->src[0]->op == GGML_OP_RMS_NORM) wgt = n->src[1];
                    else if (n->src[0] && n->src[0]->op == GGML_OP_NONE && n->src[0]->ne[1] == 1 &&
                             ggml_nelements(n->src[0]) == 128 && n->src[1] &&
                             n->src[1]->op == GGML_OP_RMS_NORM) wgt = n->src[0];
                    if (!wgt) continue;
                    if (g_gguf[qn_seen].t[9] == nullptr) g_gguf[qn_seen].t[9] = wgt;
                    else if (g_gguf[qn_seen].t[10] == nullptr) { g_gguf[qn_seen].t[10] = wgt; qn_seen++; }
                }
                int done_layers = 0;
                for (int q2 = 0; q2 < 64 && g_gguf[q2].complete(); q2++) done_layers++;
                if (done_layers > g_gguf_n) {
                    g_gguf_n = done_layers;
                    if (getenv("GGML_AXCL_LAYER_DEBUG")) {
                        int filled[11] = {0,0,0,0,0,0,0,0,0,0,0};
                        for (int q2 = 0; q2 < 64; q2++) {
                            for (int ti = 0; ti < 11; ti++) if (g_gguf[q2].t[ti]) filled[ti]++;
                        }
                        GGML_LOG_ERROR("[gguf-reg] complete=%d per-tensor: q=%d k=%d v=%d o=%d gate=%d up=%d down=%d in=%d post=%d qn=%d kn=%d\n",
                                       done_layers, filled[0], filled[1], filled[2], filled[3], filled[4],
                                       filled[5], filled[6], filled[7], filled[8], filled[9], filled[10]);
                    }
                    if (done_layers == n_layer_avail()) {
                        GGML_LOG_INFO("ggml-axcl: GGUF weight registry complete (%d layers)\n", done_layers);
                        // swap NOW, before this graph's nodes execute — the
                        // prefill must run entirely on the GGUF engines so KV
                        // caches and hidden state stay consistent
                        static bool swapped = false;
                        if (!swapped) {
                            swapped = true;
                            for (int l = 0; l < g_layer.n_layer; l++) {
                                if (g_layer.eng[l].model != 0) {
                                    for (int gi = 0; gi < 12; gi++) {
                                        if (g_layer.eng[l].io_chunk[gi])
                                            axclrtEngineDestroyIO(g_layer.eng[l].io_chunk[gi]);
                                    }
                                    axclrtEngineDestroyIO(g_layer.eng[l].io);
                                    axclrtEngineDestroyIOInfo(g_layer.eng[l].info);
                                    axclrtEngineUnload(g_layer.eng[l].model);
                                    g_layer.eng[l].model = 0;
                                    g_layer.eng[l].ectx = 0;
                                    memset(g_layer.eng[l].io_chunk, 0, sizeof(g_layer.eng[l].io_chunk));
                                }
                                g_layer.eng[l].bound = false; // fresh IO handles after reload
                                g_layer.host_wm[l] = 0;
                            }
                            g_layer.synced_pos = -1;
                            g_layer.loaded = false;
                            g_layer.n_layer = 0;
                            GGML_LOG_INFO("ggml-axcl: swapping to GGUF-patched engines\n");
                            if (!axcl_layer_load_engines(done_layers)) {
                                GGML_LOG_WARN("ggml-axcl: swap failed, retrying in 3s\n");
                                sleep(3);
                                if (!axcl_layer_load_engines(done_layers)) {
                                    GGML_LOG_ERROR("ggml-axcl: GGUF engine swap failed twice\n");
                                }
                            }
                        }
                    }
                }
            }
            // locate the FINAL norm: the RMS_NORM whose (mul'd) output
            // feeds the vocab-sized MUL_MAT — NOT the first RMS_NORM after
            // the anchors (that one is layer 27's post-attention norm!)
            g_layer.final_norm = nullptr;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                struct ggml_tensor * n = cgraph->nodes[i];
                if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[0]->ne[1] > 32768 &&
                    n->src[1] && n->src[1]->ne[0] == 1024) {
                    struct ggml_tensor * t = n->src[1];
                    for (int w = 0; w < 3 && t != nullptr; w++) {
                        if (t->op == GGML_OP_RMS_NORM) { g_layer.final_norm = t; break; }
                        t = t->src[0];
                    }
                }
            }
            g_layer.out_add = nullptr;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                struct ggml_tensor * n2 = cgraph->nodes[i];
                if (n2->op == GGML_OP_ADD && n2->ne[0] == 1024) g_layer.out_add = n2;
            }
            if (getenv("GGML_AXCL_LAYER_DEBUG")) {
                fprintf(stderr, "[prescan2] out_add=%p final_norm=%p nodes=%d tail:", (void *) g_layer.out_add, (void *) g_layer.final_norm, cgraph->n_nodes);
                for (int j = cgraph->n_nodes - 12; j < cgraph->n_nodes; j++) {
                    fprintf(stderr, " %s[%lld,%lld]", ggml_op_name(cgraph->nodes[j]->op),
                            (long long) cgraph->nodes[j]->ne[0], (long long) cgraph->nodes[j]->ne[1]);
                }
                fprintf(stderr, "\n");
            }
            // token count: the embedding GET_ROWS output row count
            g_layer.n_tok = 1;
            g_layer.pos_base = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                struct ggml_tensor * n = cgraph->nodes[i];
                if (n->op == GGML_OP_GET_ROWS && n->src[0] && n->src[0]->ne[0] == 1024 &&
                    n->src[0]->ne[1] > 32768 && n->ne[1] > 1) {
                    g_layer.n_tok = (int) n->ne[1];
                }
            }
            if (g_layer.n_tok > 2048) {
                // beyond engine context: legacy path
                g_layer.armed = false;
                layer_armed = false;
            }
            // prescan capture: KV cache views (SET_ROWS dsts in order) + pos
            int kv_seen = 0;
            int pos = -1;
            for (int i = 0; i < cgraph->n_nodes && kv_seen < 2 * g_layer.n_layer; i++) {
                struct ggml_tensor * n = cgraph->nodes[i];
                if (n->op == GGML_OP_SET_ROWS && getenv("GGML_AXCL_LAYER_DEBUG2") && kv_seen < 4) {
                    fprintf(stderr, "[sr] dst ne=[%lld,%lld,%lld] nb1=%zu ty=%d | ids ne=[%lld] ty=%d id0=%d\n",
                        (long long)(n->src[0]?n->src[0]->ne[0]:-1), (long long)(n->src[0]?n->src[0]->ne[1]:-1),
                        (long long)(n->src[0]?n->src[0]->ne[2]:-1), n->nb[1], (int)n->type,
                        (long long)(n->src[1]?n->src[1]->ne[0]:-1),
                        (int)(n->src[1]?n->src[1]->type:-1),
                        (n->src[1] && n->src[1]->type==GGML_TYPE_I32) ? *(const int32_t*)n->src[1]->data : -999);
                }
                if (n->op == GGML_OP_SET_ROWS && n->src[0] && n->src[0]->ne[0] == 1024 &&
                    n->src[1] && (n->src[1]->type == GGML_TYPE_I32 || n->src[1]->type == GGML_TYPE_I64) &&
                    n->src[1]->ne[0] >= 1 &&
                    n->nb[1] > 0) {
                    int64_t id64 = n->src[1]->type == GGML_TYPE_I64 ? *(const int64_t *) n->src[1]->data
                                                                    : *(const int32_t *) n->src[1]->data;
                    int id = (int) id64;
                    if (kv_seen == 0) pos = id;
                    int l = kv_seen / 2;
                    if (kv_seen % 2 == 0) { g_layer.host_k[l] = n; g_layer.k_nb1[l] = n->nb[1]; }
                    else                  { g_layer.host_v[l] = n; g_layer.v_nb1[l] = n->nb[1]; }
                    kv_seen++;
                }
            }
            g_layer.pos_last_pass = g_layer.pos;
            g_layer.pos = pos;
            // cache-base detection for resync: distinct VIEW/RESHAPE bases of
            // [1024, >=ctx] tensors, in graph order = K0,V0,K1,V1...
            {
                struct ggml_tensor * bases[128];
                int nb = 0;
                for (int i = 0; i < cgraph->n_nodes && nb < 2 * g_layer.n_layer; i++) {
                    struct ggml_tensor * n2 = cgraph->nodes[i];
                    for (int s = 0; s < 2; s++) {
                        struct ggml_tensor * b = s ? n2->src[1] : n2->src[0];
                        if (b == nullptr || b->ne[0] != 1024 || b->ne[1] < g_layer.ctx_len ||
                            b->ne[2] != 1 || b->ne[1] > 8192) { // not embed/lm_head
                            continue;
                        }
                        bool dup = false;
                        for (int j = 0; j < nb; j++) if (bases[j] == b) { dup = true; break; }
                        if (!dup) bases[nb++] = b;
                    }
                }
                if (nb == 2 * g_layer.n_layer) {
                    for (int l = 0; l < g_layer.n_layer; l++) {
                        g_layer.kbase[l] = bases[2 * l];
                        g_layer.vbase[l] = bases[2 * l + 1];
                    }
                }
            }
            // rewind / first-use: resync device caches from the host caches
            if (pos > 0 && kv_seen == 2 * g_layer.n_layer) {
                for (int l = 0; l < g_layer.n_layer; l++) {
                    if (g_layer.eng[l].wm != pos) {
                        axcl_layer_resync(l, pos);
                    }
                }
            }
        }
    }
    // a graph that will NOT run the layer engines executes on the CPU and may
    // read the KV caches directly — materialize deferred rows before it does
    if (layer_env && g_layer.loaded && !layer_armed && g_layer.calls > 0) {
        axcl_layer_flush_kv_all();
    }

    struct ggml_tensor * out_add = nullptr;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (done.count(i)) continue;
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue; // metadata-only: data pointer already correct
        }
        // whole-layer decode: run 1 engine call per layer at each q_proj
        // anchor; all other layer-internal nodes are subsumed by the engine
        if (g_layer.armed) {
            if (node->op == GGML_OP_GET_ROWS && node->src[0] != nullptr &&
                node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] > 32768) {
                // embedding lookup: host compute, then bf16 -> device inputs
                if (!ggml_axcl_host_op(node)) return GGML_STATUS_ABORTED;
                const int m = (int) node->ne[1];
                if (getenv("GGML_AXCL_DUMPSTATE")) {
                    FILE * ef = fopen("/tmp/llama_emb_rows.bin", "ab");
                    if (ef) { fwrite(node->data, 4, (size_t) m * 1024, ef); fclose(ef); }
                }
                if (getenv("GGML_AXCL_LAYER_DEBUG")) {
                    fprintf(stderr, "[emb] m=%d ids:", m);
                    for (int t = 0; t < m; t++) {
                        fprintf(stderr, " %d", *(const int32_t *)((const char *)node->src[1]->data + (size_t)t * node->src[1]->nb[0]));
                    }
                    fprintf(stderr, "\n");
                }
                if (g_layer.d_hidden_m.size() < (size_t) m) {
                    g_layer.d_hidden_m.resize(m, nullptr);
                }
                if (g_layer.pin_hidden == nullptr) {
                    if (axclrtMallocHost(&g_layer.pin_hidden, 2048) != AXCL_SUCC) return GGML_STATUS_ABORTED;
                }
                // batched-ladder staging: one contiguous bf16 slab for the
                // whole batch (m <= 1152 covers the group ladder)
                const bool can_batch = g_layer.n_groups > 1 && m > 1 && m <= 1152;
                if (can_batch) {
                    if (g_layer.h_all_a == nullptr) {
                        if (axclrtMalloc(&g_layer.h_all_a, (size_t) m * 2048, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
                            axclrtMalloc(&g_layer.h_all_b, (size_t) m * 2048, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC ||
                            axclrtMallocHost(&g_layer.pin_h_all, (size_t) m * 2048) != AXCL_SUCC) {
                            g_layer.n_groups = 1;
                        }
                    }
                }
                if (g_layer.n_groups > 1 && m > 1 && m <= 1152 && g_layer.h_all_a != nullptr) {
                    uint16_t * hb = (uint16_t *) g_layer.pin_h_all;
                    for (int t = 0; t < m; t++) {
                        axcl_f32_to_bf16((const float *) node->data + (size_t) t * node->nb[1] / 4,
                                         hb + (size_t) t * 1024, 1024);
                    }
                    axclrtMemcpy(g_layer.h_all_a, hb, (size_t) m * 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
                    g_layer.batched_last_hbuf = nullptr;
                }
                uint16_t * b = (uint16_t *) g_layer.pin_hidden; // pinned: unpinned H2D costs ~1ms
                // per-token staging: the chunk-ladder remainder (the LAST
                // m%128 tokens) plus ALWAYS token 0 — the first decode after
                // prefill reads its embedding from d_hidden_m[0].
                // MUST match the dispatch-level batching condition exactly
                // (env-gated): rows [1, t0_stage) are only skipped when the
                // chunk ladder will actually consume them.
                const bool batching = g_layer.n_groups > 1 && m > 1 && m <= 1152 && g_layer.h_all_a != nullptr &&
                                       getenv("GGML_AXCL_BATCH") != nullptr;
                const int t0_stage = batching ? (m / 128) * 128 : 0;
                for (int t = 0; t < m; t++) {
                    if (batching && t >= 1 && t < t0_stage) continue;
                    axcl_f32_to_bf16((const float *) node->data + (size_t) t * node->nb[1] / 4, b, 1024);
                    if (g_layer.d_hidden_m[t] == nullptr) {
                        axclrtMalloc(&g_layer.d_hidden_m[t], 2048, AXCL_MEM_MALLOC_HUGE_FIRST);
                    }
                    axclrtMemcpy(g_layer.d_hidden_m[t], b, 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
                }
                g_layer.d_hidden = g_layer.d_hidden_m[0];
                continue;
            }
            if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr &&
                node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 2048) {
                const int l = g_layer.next_layer++;
                if (l >= g_layer.n_layer || g_layer.pos < 0) {
                    GGML_LOG_ERROR("ggml-axcl: whole-layer state broken (l=%d pos=%d)\n", l, g_layer.pos);
                    return GGML_STATUS_ABORTED;
                }
                // per-token loop: the engine is a single-token machine; each
                // token's output immediately lands in its own persistent slot
                // (the shared yout buffer aliases across iterations)
                const int m = g_layer.n_tok;
                void * last_out = nullptr;
                if (getenv("GGML_AXCL_DUMPSTATE") && g_layer.pos <= 1) {
                    char fn[160];
                    FILE * sf;
                    uint16_t buf[1024];
                    snprintf(fn, sizeof(fn), "/tmp/lstate_p%d_L%d_in.bin", g_layer.pos, l);
                    sf = fopen(fn, "wb");
                    if (sf) { axclrtMemcpy(buf, g_layer.d_hidden_m[0], 2048, AXCL_MEMCPY_DEVICE_TO_HOST); fwrite(buf, 2, 1024, sf); fclose(sf); }
                }
                // GGML_AXCL_BATCH=1 opts INTO the 128-token chunk ladder.
                // DISABLED by default: chunk groups (1..9) are unusable
                // through axcl host runtime V3.6.5 — the engine ignores the
                // bound input for those groups (verified: zeroing 127/128
                // input rows leaves output unchanged), the runtime logs an
                // internal nil-pointer memcpy, and the vendor's own host
                // runtime never calls groups != 0 (traced: 18k group-0
                // executes, zero chunk groups, on a 300-token prompt).
                // Per-token prefill through group 0 runs ~55 t/s.
                const bool batching = m > 1 && g_layer.n_groups > 1 && m <= 1152 && g_layer.h_all_a != nullptr &&
                                       getenv("GGML_AXCL_BATCH") != nullptr;
                if (batching) {
                    // chunk ladder: full 128-token chunks through shape groups
                    // 1..n; the tail is zero-padded to 128 by default
                    // (GGML_AXCL_NOPAD: per-token remainder instead). Pad
                    // tokens are causally after every real token; their KV
                    // rows self-heal as decode advances.
                    static const bool pad_tail = getenv("GGML_AXCL_NOPAD") == nullptr;
                    const int nch = pad_tail ? (m + 127) / 128 : m / 128;
                    void * h_prev = (l == 0) ? g_layer.h_all_a
                                   : ((l % 2 == 1) ? g_layer.h_all_b : g_layer.h_all_a);
                    void * h_cur  = (l % 2 == 1) ? g_layer.h_all_a : g_layer.h_all_b;
                    if (l == 0) { h_prev = g_layer.h_all_a; h_cur = g_layer.h_all_b; }
                    for (int c = 0; c < nch; c++) {
                        const int p = g_layer.pos + c * 128;
                        const int real = std::min(128, m - c * 128);
                        // stage chunk input rows (offset bindings unreliable):
                        // real rows + zero pad for the tail chunk
                        axclrtMemcpy(g_layer.d_chunk_in, (char *) h_prev + (size_t) c * 128 * 2048,
                                     (size_t) real * 2048, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                        static const bool no_pad = getenv("GGML_AXCL_NOPAD") != nullptr;
                        if (real < 128 && !no_pad) {
                            static char * zpad = nullptr;
                            if (zpad == nullptr) zpad = (char *) calloc(1, 128 * 2048);
                            axclrtMemcpy((char *) g_layer.d_chunk_in + (size_t) real * 2048, zpad,
                                         (size_t) (128 - real) * 2048, AXCL_MEMCPY_HOST_TO_DEVICE);
                        }
                        (void) no_pad;
                        if (!axcl_layer_run_chunk(l, p, 128)) {
                            GGML_LOG_ERROR("ggml-axcl: chunk ladder failed at layer %d chunk %d\n", l, c);
                            return GGML_STATUS_ABORTED;
                        }
                        axclrtMemcpy((char *) h_cur + (size_t) c * 128 * 2048, g_layer.d_chunk_out,
                                     (size_t) real * 2048, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                        if (real < 128 && pad_tail) {
                            // pad rows wrote KV beyond the real watermark — pull it back
                            axcl_layer_engine * e = &g_layer.eng[l];
                            e->wm = g_layer.pos + m;
                        }
                    }
                    if (!pad_tail && m > nch * 128) {
                        // NOPAD remainder: per-token through decode group 0
                        for (int t = nch * 128; t < m; t++) {
                            g_layer.d_hidden = g_layer.d_hidden_m[t];
                            void * outdev = nullptr;
                            if (!axcl_layer_run(l, g_layer.pos + t, g_layer.host_k[l], g_layer.host_v[l],
                                                g_layer.k_nb1[l], g_layer.v_nb1[l], &outdev)) {
                                return GGML_STATUS_ABORTED;
                            }
                            axclrtMemcpy(g_layer.d_hidden_m[t], outdev, 2048,
                                         AXCL_MEMCPY_DEVICE_TO_DEVICE);
                            last_out = outdev;
                        }
                        g_layer.batched_last_hbuf = nullptr;
                    } else {
                        g_layer.batched_last_hbuf = h_cur;
                        g_layer.d_hidden = (char *) h_cur + (size_t) (m - 1) * 2048;
                        last_out = g_layer.d_hidden;
                    }
                    // restore decode-group bindings for the next pass
                    g_layer.synced_pos = -1;
                } else {
                    // legacy per-token path (single-token engines)
                    for (int t = 0; t < m; t++) {
                        // l==0: per-token embedding; m>1: previous layer's output
                        // was staged into d_hidden_m; m==1 && l>0: keep the
                        // chained hidden (the previous engine's output buffer)
                        if (l == 0 || m > 1) g_layer.d_hidden = g_layer.d_hidden_m[(size_t) t];
                        void * outdev = nullptr;
                        if (!axcl_layer_run(l, g_layer.pos + t, g_layer.host_k[l], g_layer.host_v[l],
                                            g_layer.k_nb1[l], g_layer.v_nb1[l], &outdev)) {
                            return GGML_STATUS_ABORTED;
                        }
                        if (m > 1) {
                            axclrtMemcpy(g_layer.d_hidden_m[(size_t) t], outdev, 2048,
                                         AXCL_MEMCPY_DEVICE_TO_DEVICE);
                            outdev = g_layer.d_hidden_m[(size_t) t];
                        }
                        g_layer.d_hidden = outdev;
                        last_out = outdev;
                    }
                }
                                                if (getenv("GGML_AXCL_CHECKSUM")) {
                    uint16_t hb[8];
                    axclrtMemcpy(hb, last_out, 16, AXCL_MEMCPY_DEVICE_TO_HOST);
                    double cs = 0;
                    for (int q = 0; q < 8; q++) { uint32_t u = (uint32_t)hb[q] << 16; float f; memcpy(&f, &u, 4); cs += f; }
                    fprintf(stderr, "[CS] pass=%d L%d pos=%d cs=%.6f v=", g_layer.pos_last_pass, l, g_layer.pos, cs);
                    for (int q = 0; q < 4; q++) { uint32_t u = (uint32_t)hb[q] << 16; float f; memcpy(&f, &u, 4); fprintf(stderr, "%g ", f); }
                    uint16_t kr[4], vr[4];
                    const int prow = g_layer.pos + m - 1; // row of the last executed token
                    axclrtMemcpy(kr, (char *) g_layer.eng[l].dk + (size_t) prow * 1024 * 2, 8, AXCL_MEMCPY_DEVICE_TO_HOST);
                    axclrtMemcpy(vr, (char *) g_layer.eng[l].dv + (size_t) prow * 1024 * 2, 8, AXCL_MEMCPY_DEVICE_TO_HOST);
                    fprintf(stderr, "| k=");
                    for (int q = 0; q < 2; q++) { uint32_t u = (uint32_t)kr[q] << 16; float f; memcpy(&f, &u, 4); fprintf(stderr, "%g ", f); }
                    fprintf(stderr, "v=");
                    for (int q = 0; q < 2; q++) { uint32_t u = (uint32_t)vr[q] << 16; float f; memcpy(&f, &u, 4); fprintf(stderr, "%g ", f); }
                    uint16_t hin[4];
                    axclrtMemcpy(hin, l == 0 ? g_layer.dx_in : last_out, 8, AXCL_MEMCPY_DEVICE_TO_HOST);
                    fprintf(stderr, "in=");
                    for (int q = 0; q < 2; q++) { uint32_t u = (uint32_t)hin[q] << 16; float f; memcpy(&f, &u, 4); fprintf(stderr, "%g ", f); }
                    fprintf(stderr, "\n");
                }
                if (getenv("GGML_AXCL_LAYER_DEBUG") && g_layer.next_layer <= 1) {
                    const uint64_t w = axcl_us();
                    fprintf(stderr, "[layer] pass pos=%d l=%d calls=%llu us=%llu"
                            " | tok: wall=%lld eng=%lld host=%lld (us)\n",
                            g_layer.pos, l, (unsigned long long) g_layer.calls,
                            (unsigned long long) g_layer.us,
                            (long long) (w - g_layer.last_wall_us),
                            (long long) (g_layer.us - g_layer.last_eng_us),
                            (long long) ((w - g_layer.last_wall_us) - (g_layer.us - g_layer.last_eng_us)));
                    g_layer.last_wall_us = w;
                    g_layer.last_eng_us = g_layer.us;
                }
                continue;
            }
            if (g_layer.next_layer == g_layer.n_layer) {
                // past the last layer: at the FINAL residual ADD, write the
                // engine chain's hidden and DISARM — the final norm + weight
                // mul are in this fragment and compute via host ops from it;
                // only the vocab matmul lives on the CPU side.
                if (node == g_layer.out_add && node->op == GGML_OP_ADD) {
                    // remember the device hidden (bf16) for the post engine
                    axcl_post_load();
                    const int rows = (int) node->ne[1];
                    const int src_rows = std::max(1, g_layer.n_tok);
                    if (g_layer.batched_last_hbuf != nullptr) {
                        // chunk-ladder prefill: final hidden rows live in the
                        // batch buffer; stage the last row for the post engine
                        // (offset INPUT bindings are unreliable on this runtime)
                        axclrtMemcpy(g_layer.d_yout,
                                     (char *) g_layer.batched_last_hbuf + (size_t) (src_rows - 1) * 2048,
                                     2048, AXCL_MEMCPY_DEVICE_TO_DEVICE);
                        g_layer.post_hidden = g_layer.d_yout;
                    } else {
                        g_layer.post_hidden = g_layer.d_hidden;
                    }
                    if (g_layer.pin_hidden == nullptr) {
                        if (axclrtMallocHost(&g_layer.pin_hidden, 2048) != AXCL_SUCC) return GGML_STATUS_ABORTED;
                    }
                    for (int t = 0; t < rows; t++) {
                        uint16_t * b = (uint16_t *) g_layer.pin_hidden;
                        const int st = (rows == 1) ? (src_rows - 1) : t;
                        const void * src = (g_layer.batched_last_hbuf != nullptr)
                            ? (const void *) ((char *) g_layer.batched_last_hbuf + (size_t) st * 2048)
                            : (const void *) ((rows == 1 && src_rows == 1) ? g_layer.d_hidden
                                                                            : g_layer.d_hidden_m[(size_t) st]);
                        axclrtMemcpy(b, src, 2048, AXCL_MEMCPY_DEVICE_TO_HOST);
                        if (getenv("GGML_AXCL_DUMPSTATE") && g_layer.pos <= 5 && t == rows - 1) {
                            char fn[160];
                            snprintf(fn, sizeof(fn), "/tmp/llama_out_p%d_t%d.bin", g_layer.pos, t);
                            FILE * sf = fopen(fn, "wb");
                            if (sf) { fwrite(b, 2, 1024, sf); fclose(sf); }
                        }
                        float * f = (float *)((char *) node->data + (size_t) t * node->nb[1]);
                        axcl_bf16_to_f32(b, f, 1024);
                    }
                    g_layer.armed = false;
                    g_layer.batched_last_hbuf = nullptr; // decode passes must use the chain hidden
                    if (getenv("GGML_AXCL_CHECKSUM")) {
                        float f8[8];
                        memcpy(f8, node->data, 32);
                        double cs = 0;
                        for (int q = 0; q < 8; q++) cs += f8[q];
                        fprintf(stderr, "[CS-OUT] pos=%d rows=%d cs=%.6f\n", g_layer.pos, rows, cs);
                    }
                    continue;
                }
                continue;
            }
            continue; // layer-internal node: subsumed by the engine
        }
        // sharing src1. Executed at the FIRST node — computing late (at the
        // v node) violates the allocator's lifetime assumptions and corrupts
        // downstream reads via reused chunks.
        // fused rmsnorm+qkv: intercept the q_proj whose src1 is a norm node —
        // one engine call replaces the norm engine + qkv fusion
        // FA arrives as a claimed op; dispatch it to the host-op engine
        // handler directly (the generic host_op call below is nested under
        // chain-mode gates that do not apply here)
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            uint64_t tt_a = axcl_us();
            const bool aok = ggml_axcl_host_op(node);
            prof_t_attn += axcl_us() - tt_a; prof_n_attn++;
            if (aok) continue;
            GGML_LOG_ERROR("ggml-axcl: claimed FLASH_ATTN_EXT not handled\n");
            return GGML_STATUS_ABORTED;
        }


        // in SEPARATE fragments (CPU view ops between). Stash q and k
        // uncomputed; at the v fragment run the fused engine once and write
        // all three outputs. Scheduler order guarantees q/k outputs are
        // consumed (rope) only after the v fragment.
        // NOTE: default-off. The scheduler interleaves CPU fragments (q_norm,
        // rope) between the q/k/v projection fragments — those read q/k
        // outputs before this fusion's v-fragment writeback, corrupting the
        // KV cache. Enable only with a scheduler that guarantees ordering.
        if (node->op == GGML_OP_MUL_MAT && g_qkv.model != 0 &&
            node->src[1]->ne[1] == 1 &&
            getenv("GGML_AXCL_QKV_X") != nullptr) {
            const bool is_q = node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 2048;
            const bool is_kv = node->src[0]->ne[0] == 1024 && node->src[0]->ne[1] == 1024;
            const void * xs = node->src[1]->data;
            if (is_q) {
                if (xqkv_q[0] != nullptr) { xqkv_q[0] = xqkv_q[1] = xqkv_q[2] = nullptr; }
                xqkv_q[0] = node; xqkv_src1 = xs; xqkv_count = 1;
                // snapshot the activation NOW: the buffer may be reused by
                // CPU ops in the fragments between q and v
                {
                    const struct ggml_tensor * h = node->src[1];
                    if (h->type == GGML_TYPE_F32) memcpy(xqkv_h, h->data, sizeof(xqkv_h));
                    else {
                        const auto * tr = ggml_get_type_traits(h->type);
                        if (!tr || !tr->to_float) { xqkv_count = 0; goto qkv_done; }
                        tr->to_float(h->data, xqkv_h, 1024);
                    }
                }
                continue; // deferred: computed by the fused engine at v
            }
            if (is_kv && xqkv_q[0] != nullptr && xqkv_src1 == xs && xqkv_q[0]->src[1]->data == xs) {
                if (xqkv_count == 1) {
                    xqkv_q[1] = node; xqkv_count = 2;
                    continue; // deferred
                } else if (xqkv_count == 2 && xqkv_q[1]->src[1]->data == xs) {
                    xqkv_q[2] = node; xqkv_count = 3;
                    uint64_t tt_q = axcl_us();
                    const bool ok = axcl_qkv_try_flush();
                    prof_t_qkv += axcl_us() - tt_q; prof_n_qkv++;
                    if (ok) continue;
                    // engine failed: outputs unwritten -> fall through to
                    // compute this v normally (q/k lost: NOT recoverable;
                    // abort loud rather than emit garbage)
                    GGML_LOG_ERROR("ggml-axcl: cross-fragment QKV engine failed\n");
                    return GGML_STATUS_ABORTED;
                }
            }
            // unrelated [1024,1020] matmul while stashing: reset defensively
            if (xqkv_q[0] != nullptr && !is_kv) {
                xqkv_q[0] = xqkv_q[1] = xqkv_q[2] = nullptr; xqkv_count = 0;
            }
        }
        qkv_done: ;
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
        if (node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
            node->src[0]->ne[1] > 32768 && node->src[0]->ne[0] == 1024 &&
            node->src[1]->ne[1] > 1 && node->src[1]->ne[1] <= 64 &&
            node->src[1]->ne[0] == 1024 && node->src[1]->nb[0] == 4 &&
            node->src[1]->nb[1] == 4096 && node->src[1]->data != nullptr &&
            node->nb[1] >= 151936 * 4 &&
            getenv("GGML_AXCL_VOCAB64") != nullptr) {
            // multi-position vocab matmul (speculative-verification batch):
            // ONE 64-row head call replaces m rows of CPU vocab math
            axcl_vocab64_load();
            if (g_v64.ok) {
                struct ggml_tensor * s1 = node->src[1];
                const int64_t m = s1->ne[1];
                float * st = g_v64.pin_in;
                for (int64_t t = 0; t < m; t++)
                    memcpy(st + (size_t) t * 1024, (const char *) s1->data + (size_t) t * s1->nb[1], 4096);
                if (m < 64) memset(st + (size_t) m * 1024, 0, (size_t) (64 - m) * 4096);
                std::lock_guard<std::mutex> lock(axcl_exec_mutex);
                axclrtMemcpy(g_v64.dx, st, 64 * 1024 * 4, AXCL_MEMCPY_HOST_TO_DEVICE);
                axclrtEngineSetInputBufferByIndex(g_v64.io, g_v64.ix, g_v64.dx, 64 * 1024 * 4);
                axclrtEngineSetOutputBufferByIndex(g_v64.io, g_v64.iyo, g_v64.dy, (size_t) 64 * 151936 * 4);
                if (axclrtEngineExecute(g_v64.model, g_v64.ectx, 0, g_v64.io) != AXCL_SUCC) {
                    GGML_LOG_ERROR("ggml-axcl: vocab64 execute failed\n");
                    return GGML_STATUS_ABORTED;
                }
                axclrtMemcpy(g_v64.pin_out, g_v64.dy, (size_t) m * 151936 * 4, AXCL_MEMCPY_DEVICE_TO_HOST);
                for (int64_t t = 0; t < m; t++) {
                    float * dstf = (float *)((char *) node->data + (size_t) t * node->nb[1]);
                    memcpy(dstf, g_v64.pin_out + (size_t) t * 151936,
                           (size_t) std::min<int64_t>(node->ne[0], 151936) * 4);
                }
                g_layer.post_hidden = nullptr;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT && g_post.ok && node->src[0] &&
            node->src[0]->ne[1] > 32768 && node->src[0]->ne[0] == 1024 &&
            g_layer.post_hidden != nullptr) {
            // vocab head via the post engine: input = the device-resident
            // final hidden (no H2D), output = bf16 logits -> f32 dst
            std::lock_guard<std::mutex> lock(axcl_exec_mutex);
            const size_t elem = g_post.out_f32 ? 4 : 2;
            axclrtEngineSetInputBufferByIndex(g_post.io, g_post.ix, g_layer.post_hidden, 1024 * 2);
            axclrtEngineSetOutputBufferByIndex(g_post.io, g_post.iyo, g_post.dy,
                                               (size_t) g_post.n_out * elem);
            if (axclrtEngineExecute(g_post.model, g_post.ectx, 0, g_post.io) != AXCL_SUCC) {
                GGML_LOG_ERROR("ggml-axcl: post engine execute failed\n");
                return GGML_STATUS_ABORTED;
            }
            if (g_layer.pin_logits == nullptr) {
                if (axclrtMallocHost((void **) &g_layer.pin_logits, 151936 * 4) != AXCL_SUCC) {
                    GGML_LOG_ERROR("ggml-axcl: pinned logits staging alloc failed\n");
                    return GGML_STATUS_ABORTED;
                }
            }
            axclrtMemcpy(g_layer.pin_logits, g_post.dy, (size_t) g_post.n_out * elem, AXCL_MEMCPY_DEVICE_TO_HOST);
            // llama.cpp samples from the LAST position's logits; for
            // multi-token graphs write the row it reads
            const int64_t mrows = node->ne[1];
            float * dstf = (float *)((char *) node->data + (size_t)(mrows - 1) * node->nb[1]);
            const int64_t nv = node->ne[0];
            if (g_post.trim_ids != nullptr) {
                // trimmed post: scatter kept logits, -inf elsewhere
                const float * src;
                if (g_post.out_f32) {
                    src = (const float *) g_layer.pin_logits;
                } else {
                    axcl_bf16_to_f32(g_layer.pin_logits, g_post.f32_scratch, g_post.n_out);
                    src = g_post.f32_scratch;
                }
                const int64_t nvf = std::min<int64_t>(nv, 151936);
                for (int64_t v = 0; v < nvf; v++) dstf[v] = -INFINITY;
                for (int k = 0; k < g_post.n_out; k++) {
                    const int32_t id = g_post.trim_ids[k];
                    if (id >= 0 && id < nvf) dstf[id] = src[k];
                }
            } else if (g_post.out_f32) {
                memcpy(dstf, g_layer.pin_logits, (size_t) std::min<int64_t>(nv, 151936) * 4);
            } else {
                axcl_bf16_to_f32(g_layer.pin_logits, dstf, (int) std::min<int64_t>(nv, 151936));
            }
            g_layer_logits_on_npu = true;
            g_layer.post_hidden = nullptr;
            continue;
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
                    if (src0->ne[1] == 128 && src0->ne[0] > 128) { // gate: engine K/V stride handling verified only for seq > 128 under unsplit graphs
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
    // a graph that ran zero engine calls executed on the CPU with our
    // device-authoritative KV rows unmaterialized — flush before it could
    // have read them is impossible retroactively, so flush now (bounded loss)
    if (g_layer.calls == calls_at_entry) {
        axcl_layer_flush_kv_all();
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
        // whole-layer mode replaces the ENTIRE legacy path (per-op matmul
        // engines + 2.5GB weight pool + fused engines) — armed graphs never
        // touch them. Skip ~3.4GB of CMM that would sit idle.
        const bool layer_only = getenv("GGML_AXCL_LAYER") != nullptr ||
                                getenv("GGML_AXCL_GGUF") != nullptr;
        if (!no_engines && !layer_only) {
        axcl_preload_all_engines(); // outside the activation mutex
        axcl_weight_pool_init();
        axcl_attn_load();
        if (getenv("GGML_AXCL_CHAIN") != nullptr) {
            axcl_chain_load();
        }
        }
        // whole-layer engines load regardless of the legacy-skip above; in
        // GGUF mode the loader defers to patched engines (see load_engines)
        if (getenv("GGML_AXCL_LAYER") != nullptr || getenv("GGML_AXCL_GGUF") != nullptr) {
            // LOAD ORDER MATTERS (vendor runtime bug, V3.10.2): loading a
            // `pulsar2 build` engine AFTER any llm_build engine drops the
            // PCIe device ("recv dma size 0" — 2-engine repro in FINDINGS).
            // pulsar2-build heads (vocab64, trimmed post) load FIRST.
            axcl_vocab64_load();
            axcl_post_load();
            axcl_layer_load_engines(-1); // layer count from template files
        }
        if (!no_engines && !layer_only) {
        // QKV fused engine: rms_norm(hidden) -> q, k, v in one call
        if (!layer_only && axcl_fused_load(&g_qkv, "/usr/local/share/ggml-axcl/qkv_nn_h1024_q2048_kv1024.axmodel",
                            {"h", "q_w", "k_w", "v_w"}, {"q", "k", "v"})) {
            axcl_fused_alloc(&g_qkv,
                {1024*4, (size_t)1024*2048*4, (size_t)1024*1024*4, (size_t)1024*1024*4},
                {2048*4, 1024*4, 1024*4});
            GGML_LOG_INFO("ggml-axcl: QKV no-norm engine loaded\n");
        }
        // gate+up fused engine: h @ gate_w, h @ up_w in one call
        if (!layer_only && axcl_fused_load(&g_gate_up, "/usr/local/share/ggml-axcl/gate_up_h1024_i3072.axmodel",
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
        // whole-layer mode claims the vocab head for the post engine (final
        // norm + lm_head computed on the NPU; hidden stays device-resident)
        if (getenv("GGML_AXCL_LAYER") || getenv("GGML_AXCL_GGUF")) {
            return op->src[0]->ne[0] == 1024;
        }
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
            case GGML_OP_FLASH_ATTN_EXT:
                return getenv("GGML_AXCL_FA") != nullptr;
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
    // GGML_AXCL_LAYER=1 implies the same claim set: the whole-layer path
    // needs the unsplit graph to find its layer anchors.
    static const char * chain_env = getenv("GGML_AXCL_CHAIN") != nullptr ? "1" : getenv("GGML_AXCL_LAYER");
    if (chain_env != nullptr) {
        const bool layer_mode = getenv("GGML_AXCL_LAYER") != nullptr;
        switch (op->op) {
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT: {
                // whole-layer mode claims metadata ops too: the layer-engine
                // interception needs the UNSPLIT decode graph (one fragment
                // with all 28 anchors). Metadata ops are no-ops in our
                // graph_compute (data pointers already correct).
                // GGML_AXCL_META="VIEW,PERMUTE,..." bisects per-op.
                static const char * meta_env = getenv("GGML_AXCL_META");
                if (meta_env != nullptr && layer_mode) {
                    char buf[128];
                    snprintf(buf, sizeof(buf), ",%s,", meta_env);
                    char want[32];
                    snprintf(want, sizeof(want), ",%s,", ggml_op_name(op->op));
                    return strstr(buf, want) != nullptr;
                }
                return layer_mode;
            }
            case GGML_OP_FLASH_ATTN_EXT:
                // FA only when explicitly enabled: the FA host-op was the
                // default-claim inversion bug (claimed when env UNSET) that
                // corrupted all chain/universal runs. GGML_AXCL_FA=1 routes
                // it to the attention engine (Q no longer double-scaled).
                // Whole-layer mode additionally claims SINGLE-QUERY FA so
                // decode graphs arrive unsplit (the armed path skips it —
                // the layer engine subsumes attention); prefill stays CPU.
                // NEVER claim FA implicitly: claiming FLASH_ATTN_EXT flips
                // llama.cpp onto its FA graph path, whose tensor layout our
                // legacy per-op path mishandles (the historical corruption).
                // Whole-layer mode needs no FA claims: with metadata claimed
                // the manual-attention decode graph already arrives unsplit.
                return getenv("GGML_AXCL_FA") != nullptr;
            default:
                return true;
        }
    }
    if (uni_env != nullptr) {
        return true;
    }
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        return getenv("GGML_AXCL_FA") != nullptr;
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
