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
    void *  ptr;
    size_t  size;
};

static void ggml_backend_axcl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    axclrtFree(ctx->ptr);
    delete ctx;
}

static void * ggml_backend_axcl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_axcl_buffer_context * ctx = (ggml_backend_axcl_buffer_context *) buffer->context;
    return ctx->ptr;
}

static void ggml_backend_axcl_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                   uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    void * dst = (char *) tensor->data + offset;
    if (axclrtMemset(dst, value, size) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: memset failed\n");
    }
}

static void ggml_backend_axcl_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    void * dst = (char *) tensor->data + offset;
    if (axclrtMemcpy(dst, data, size, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: H2D memcpy failed (%zu bytes)\n", size);
    }
}

static void ggml_backend_axcl_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                                void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    const void * src = (const char *) tensor->data + offset;
    if (axclrtMemcpy(data, src, size, AXCL_MEMCPY_DEVICE_TO_HOST) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: D2H memcpy failed (%zu bytes)\n", size);
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
    if (axclrtMemset(ctx->ptr, value, ctx->size) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: clear failed\n");
    }
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

    void * ptr = nullptr;
    if (axclrtMalloc(&ptr, size, AXCL_MEM_MALLOC_HUGE_FIRST) != AXCL_SUCC || ptr == nullptr) {
        GGML_LOG_ERROR("ggml-axcl: axclrtMalloc failed for %zu bytes\n", size);
        // null buffer: the caller will fall back to another device
        return ggml_backend_buffer_init(buft, ggml_backend_axcl_buffer_interface, nullptr, 0);
    }

    auto * ctx = new ggml_backend_axcl_buffer_context{0, ptr, size};

    return ggml_backend_buffer_init(buft, ggml_backend_axcl_buffer_interface, ctx, size);
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
    std::vector<float> x_h, w_h, y_h;    // host staging
};

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

static axcl_matmul * axcl_matmul_load(int64_t k, int64_t n) {
    if (!axcl_engine_global_init()) {
        return nullptr;
    }

    axcl_matmul * mm = new axcl_matmul();
    mm->m = 1;
    mm->k = k;
    mm->n = n;

    std::string path = axcl_matmul_model_path(k, n);
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

    GGML_LOG_INFO("ggml-axcl: matmul engine loaded %s (inputs '%s','%s' -> '%s', idx %d,%d -> %d)\n",
                  path.c_str(), mm->x_name.c_str(), mm->w_name.c_str(), mm->y_name.c_str(), mm->x_idx,
                  mm->w_idx, mm->y_idx);

    return mm;
}

// shape-keyed cache with negative-result memoization
static axcl_matmul * axcl_matmul_get(int64_t k, int64_t n) {
    struct Cache {
        std::mutex mutex;
        std::unordered_map<uint64_t, axcl_matmul *> engines; // key: (k << 32) | n
        std::unordered_map<uint64_t, bool>        missing;
    };
    static Cache cache;

    const uint64_t key = ((uint64_t) k << 32) | (uint32_t) n;

    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.engines.find(key);
    if (it != cache.engines.end()) {
        return it->second;
    }
    if (cache.missing.count(key)) {
        return nullptr;
    }
    axcl_matmul * mm = axcl_matmul_load(k, n);
    if (mm) {
        cache.engines[key] = mm;
    } else {
        cache.missing[key] = true;
    }
    return mm;
}

// dequantize src0 (ggml weight, [K x N] row-major, ne0 = K) into a
// transposed f32 [K, N] row-major host buffer: w[k*N + n] = src0[n][k]
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
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * w = (const ggml_fp16_t *) t->data;
        for (int64_t nn = 0; nn < n; nn++) {
            for (int64_t kk = 0; kk < k; kk++) {
                w32[kk * n + nn] = GGML_COMPUTE_FP16_TO_FP32(w[nn * k + kk]);
            }
        }
    } else {
        GGML_ABORT("ggml-axcl: unsupported weight type %s", ggml_type_name(t->type));
    }
}

static bool ggml_axcl_compute_mul_mat(axcl_matmul * mm, const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t k = mm->k;
    const int64_t n = mm->n;

    // X: activation, src1 is [K, 1] f32 contiguous
    memcpy(mm->x_h.data(), src1->data, (size_t) k * 4);

    // W: dequant + transpose into [K, N]
    axcl_dequant_any_to_f32_transposed(src0, mm->w_h.data());

    if (axclrtMemcpy(mm->dx, mm->x_h.data(), (size_t) k * 4, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC ||
        axclrtMemcpy(mm->dw, mm->w_h.data(), (size_t) k * n * 4, AXCL_MEMCPY_HOST_TO_DEVICE) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: H2D copy failed\n");
        return false;
    }

    if (axclrtEngineSetInputBufferByIndex(mm->io, mm->x_idx, mm->dx, (size_t) k * 4) != AXCL_SUCC ||
        axclrtEngineSetInputBufferByIndex(mm->io, mm->w_idx, mm->dw, (size_t) k * n * 4) != AXCL_SUCC ||
        axclrtEngineSetOutputBufferByIndex(mm->io, mm->y_idx, mm->dy, (size_t) n * 4) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: bind device buffers failed\n");
        return false;
    }

    if (axclrtEngineExecute(mm->model_id, mm->context_id, 0, mm->io) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: engine execute failed\n");
        return false;
    }

    // Y: f32 [N] device -> host -> dst
    if (axclrtMemcpy(mm->y_h.data(), mm->dy, (size_t) n * 4, AXCL_MEMCPY_DEVICE_TO_HOST) != AXCL_SUCC) {
        GGML_LOG_ERROR("ggml-axcl: D2H copy failed\n");
        return false;
    }
    memcpy(dst->data, mm->y_h.data(), (size_t) n * 4);
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

static enum ggml_status ggml_backend_axcl_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_axcl_context * ctx = (ggml_backend_axcl_context *) backend->context;
    axclrtSetDevice(ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

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
            GGML_LOG_ERROR("ggml-axcl: node %d op %s not supported\n", i, ggml_op_name(node->op));
            return GGML_STATUS_ABORTED;
        }
    }

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
    if (device < 0 || device >= axcl_get_device_count()) {
        GGML_LOG_ERROR("ggml-axcl: invalid device %d\n", device);
        return nullptr;
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
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    // f32 compute path; quantized weight layouts land in milestone 3
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_n_dims(src0) != 2 || ggml_n_dims(src1) != 2) {
        return false;
    }
    if (src1->ne[1] != 1) {
        return false; // decode only (M == 1); prefill stays on CPU
    }
    // engine must exist for this (K, N)
    return axcl_matmul_get(src0->ne[0], src0->ne[1]) != nullptr;
}

static bool ggml_backend_axcl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return dev == buft->device || (buft->iface.get_name(buft) == std::string_view("AXCL"));
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
