#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Axera AXCL NPU backend (AX8850 / AX650 family, M.2 accelerator cards)
//
// Milestone 1: device enumeration + CMM buffer management.
// No ops are supported yet; the scheduler will not assign work to the NPU
// until ggml_backend_axcl_supports_op() returns true for an op.

GGML_API ggml_backend_reg_t ggml_backend_axcl_reg(void);

GGML_API ggml_backend_t ggml_backend_axcl_init(int32_t device);

GGML_API bool ggml_backend_is_axcl(ggml_backend_t backend);

// AXCL buffer type for device memory (CMM)
GGML_API ggml_backend_buffer_type_t ggml_backend_axcl_buffer_type(int32_t device);

#ifdef __cplusplus
}
#endif
