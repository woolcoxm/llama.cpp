# ggml-axcl: a ggml backend for Axera NPUs

**Status:** milestone 1 (device enumeration + CMM buffers) — code in
`ggml/src/ggml-axcl/`, compiles against the real AXCL SDK headers.

## Goal

Make every GGUF model in llama.cpp usable on Axera AX8850/AX650 NPUs
(M5Stack LLM-8850 M.2 card and friends) instead of only the vendor-ported
`.axmodel` models.

## Hardware context

| | |
|---|---|
| Card | M5Stack LLM-8850 (Axera AX8850) on Raspberry Pi 5 M.2 HAT+ |
| NPU | 24 TOPS INT8, **graph-only** execution engine |
| Card RAM | 8 GB LPDDR4x (~7 GB usable CMM), separate from the Pi's 16 GB |
| Link | PCIe Gen2, **currently x1** (x2 possible via EEPROM/config fix) |
| Host driver | `axclhost` DKMS package (M5Stack apt repo), kernel modules `axcl_host` etc. |

## The core constraint

The AXCL runtime (like Rockchip RKNN, Hailo) executes **precompiled
graphs** (`.axmodel`), not individual ops. There is no op-level GEMM
primitive. Therefore a ggml backend cannot naively offload `MUL_MAT`.

Prior art that solved the same problem: **rk-llama.cpp** (RK3588) —
they pre-compile NPU matmul graphs for fixed shapes and call them from a
ggml backend. We follow the same strategy.

## Architecture

```
llama.cpp scheduler
   │  ops assigned per-backend as usual
   ▼
ggml-axcl backend
   │  supports_op: MUL_MAT (decode: batch=1 static shapes)
   ▼
axcl_matmul_cache: shape-keyed cache of engine contexts
   │  lookup: (N, K, type, quant) → axclrtEngine context
   │  miss:   load precompiled matmul .axmodel for that shape bucket
   ▼
AXCL runtime: axclrtEngineExecute on NPU
   ▼
weights staged in CMM (axclrtMalloc), activations DMA'd per call
```

Key decisions:

1. **Weights live in on-card CMM.** The ggml buffer type backed by
   `axclrtMalloc` holds model weights on the card. Per-token activations
   cross PCIe (small, KBs), so the x1 link is acceptable; weight reads
   stay on-card at LPDDR4x bandwidth.
2. **Static-shape decode first.** During decode, llama.cpp matmuls are
   batch 1 with fixed N/K per layer — a *finite set of shapes*. We
   precompile one matmul graph per (N, K) bucket with Pulsar2 for the
   common tile sizes and pad/replicate the input buffer. Prefill (dynamic
   batch) stays on CPU initially; `axclrtEngineSetDynamicBatchSize` may
   unlock it later.
3. **Quantization handling.** The NPU computes in int8/int16 with w4a16/
   w8a16-style graphs. The .axmodel matmul graphs are compiled GPTQ-int4;
   ggml Q4_0/Q8_0 blocks must be repacked to the layout the graph expects
   at buffer `set_tensor` time (weights are written once, so the cost is
   amortized) — or we compile the graph to consume raw block layouts via
   a custom pre-pass.
4. **Everything else stays on CPU.** Attention/KV/rope/sampling remain on
   the Pi's CPU backend. This mirrors the RK backend and keeps the first
   working version small.

## AXCL runtime API (verified from installed headers, SDK 3.6.5)

- init/device: `axclInit`, `axclrtGetDeviceCount`, `axclrtSetDevice`,
  `axclrtGetDeviceProperties` (swVersion, pci ids, temp, totalCmmSize KB,
  freeCmmSize KB, npuLoading)
- memory: `axclrtMalloc/MallocCached/MallocHost`, `axclrtFree`,
  `axclrtMemcpy(Async)` with `AXCL_MEMCPY_HOST_TO_DEVICE` etc.,
  `axclrtMemFlush/MemInvalidate` (cache maintenance),
  `axclrtMemset`, policy `AXCL_MEM_MALLOC_HUGE_FIRST`
- engine: `axclrtEngineLoadFromFile/Mem`, `EngineInit/Finalize`,
  `EngineCreateContext/IO`, `EngineGet{Input,Output}{Dims,DataType,
  BufferByName,...}`, `EngineExecute(Async)`,
  **`EngineSetDynamicBatchSize`** (dynamic batching exists!)
- streams/events: `axclrtCreateStream`, `SynchronizeStream`

## Milestones

- [x] M0 — driver on Debian 13 (DKMS patch: `__DATE__`/`__TIME__` vs
      `-Werror=date-time` on the 6.18 rpt kernel)
- [x] M1 — backend skeleton: device + buffer type + registration, compiles
- [ ] M2 — first op: MUL_MAT via precompiled matmul axmodel
      - build a test matmul .axmodel with Pulsar2 (x86 host, `llm_build2`
        or a minimal ONNX→Pulsar2 path for a pure GEMM graph)
      - engine cache keyed by (N, K, dtype), weights uploaded via
        buffer set_tensor, correctness vs CPU reference
- [ ] M3 — end-to-end: llama-cli with `-ngl 99` offloading matmuls,
      llama-bench vs CPU on Pi 5
- [ ] M4 — prefill via dynamic batch, KV/attention offload exploration,
      f16/f32 paths, MUL_MAT_ID (MoE)
- [ ] M5 — upstreaming: GGML_AXCL option, CI, docs; consider
      GGML_BACKEND_DL standalone .so distribution

## Build

On the Pi (aarch64, headers + lib from `axclhost` package):

```
cmake -B build -DGGML_AXCL=ON -DGGML_BACKEND_DL=ON
cmake --build build -j4
```

On x86 (syntax/dev only, headers vendored in `../vendor/axcl-include`):

```
g++ -std=c++17 -c ggml/src/ggml-axcl/ggml-axcl.cpp \
    -Iggml/include -Iggml/src -I../vendor/axcl-include
```

## References

- AXERA-TECH GitHub: ax-llm, axcl-samples, axcl-docs, ax-samples
- Pulsar2 LLM build docs: pulsar2-docs.readthedocs.io (LLM build chapter)
- rk-llama.cpp (invisiofficial) — RK3588 NPU ggml backend precedent
- In-tree NPU backend precedents: ggml-cann (Ascend), ggml-hexagon (QCOM)
- M5Stack LLM-8850 docs: docs.m5stack.com (axclhost install, NPU benchmarks)
