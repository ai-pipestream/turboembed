# SOLIDIFY (7) Intel — remote USM wrap + OV accuracy (Machine B)

Hostnames stay out of this file (Machine B only).

This item is **not** the Machine A CUDA latency bench
(`docs/bench-turbo-machine-a.md`, `make bench-machine-a`). This page is
the Intel wrap / accuracy gate.

## Done criteria

| # | Gate | Proof |
|---|---|---|
| 1 | Remote-OCL / `USM_USER_BUFFER` wrap of **turbo_buffer ZE SHARED** is either LIVE or proven unavailable | `make probe-remote-usm` + `testdata/receipts/turborerank/intel-remote-usm-probe.txt`. C++ `test_ov_remote_usm_wrap_unavailable`. |
| 2 | No greenwash | Receipts keep `remote_ocl_usm_wrap: false` unless `create_tensor(USM_USER_BUFFER)` actually accepts a ZE SHARED pointer. Plugin-owned `USM_HOST_BUFFER` is a different allocator and does not count. |
| 3 | Berlin abs error improves vs the FP16-IR floor (~8.3e-4) when possible | LIVE: GPU `max_abs_logit_err` **1.43e-6** (was 8.32e-4). CPU **3.34e-6**. `onnx_to_ir` saves `compress_to_fp16=false`. Compile uses `ACCURACY` + `inference_precision=f32` + `LATENCY` + `dynamic_quantization_group_size=0`. |

## Software stack (this host)

| piece | value |
|---|---|
| OpenVINO Runtime | 2026.3.1-22476 (`pkg-config openvino`) |
| GPU | Intel(R) Graphics [0xe223] (dGPU), uarch 20.2.0, 256 EUs |
| GPU plugin capabilities | `FP32 BIN FP16 INT8 GPU_HW_MATMUL GPU_USM_MEMORY` |
| Default GPU hints | `INFERENCE_PRECISION_HINT=f16`, `EXECUTION_MODE_HINT=PERFORMANCE` |
| Compiled-model context | `CONTEXT_TYPE=OCL` (`OCL_CONTEXT` set, `OCL_QUEUE=0`) |
| turbo_buffer tokens | `zeMemAllocShared` (Level Zero USM, place SHARED=3) |
| OpenCL ICD | `intel-opencl-icd` 26.05.37020.3 — `clinfo` lists the dGPU |
| `CL/cl2.hpp` | **missing** (`opencl-headers` / `opencl-clhpp-headers` not installed). `intel_gpu/ocl/ocl.hpp` does not compile on this stack. |

2026.3 documents Level Zero–OpenCL interop (LEO) so an OCL remote
context can sit on a ZE engine. This **installed** plugin still
serves an **OCL engine**: wrap of a user pointer goes through
`ocl_engine.cpp` `get_usm_allocation_size` (`clGetMemAllocInfoINTEL`).

## Live probe (Machine B)

`make probe-remote-usm` compiles
[`native/turborerank/tools/probe_remote_usm.cpp`](../native/turborerank/tools/probe_remote_usm.cpp)
without `ocl.hpp` (RemoteContext + `AnyMap` only).

Observed:

1. `compiled.get_context()` succeeds. Context type is **OCL**.
2. `create_tensor(..., USM_USER_BUFFER, ze_shared_ptr)` throws:

   `[GPU] shared USM buffer has smaller size (0) than specified layout (32)`

   at `src/plugins/intel_gpu/src/runtime/ocl/ocl_engine.cpp:311`.

   OpenCL USM size query does not see a `zeMemAllocShared` pointer
   from turbo_buffer's Level Zero context. Size 0 → assert.
3. Plugin-allocated `USM_HOST_BUFFER` + infer **works**. The remote
   API is live; the pointer just has to be OCL-owned.
4. Today's path `ov::Tensor(element::i32, shape, usm_pointer)` +
   `set_tensor` still infers (host-side wrap; plugin may copy).
5. `ocl.hpp` is unusable here without Khronos C++ headers. Installing
   those headers would not change (2): the size-0 check is in the
   GPU plugin, not in our compile.

**Verdict: wrapping turbo_buffer ZE SHARED as OV remote OCL
`USM_USER_BUFFER` is impossible on this Machine B software stack.**
Switching the token arena to OV-allocated OpenCL USM would abandon
the ZE arena proofs (SOLIDIFY 1 / 4 / 5) and is out of scope.

## Accuracy

`native/turborerank/tools/onnx_to_ir.cpp` previously called
`ov::save_model(..., /*compress_to_fp16=*/true)`. That is the
~8.3e-4 Berlin floor: compile-time `ACCURACY` + `f32` cannot
restore bits dropped at IR save.

This item saves FP32 IR and compiles TurboRerank + TurboEmbed with:

- `ov::hint::execution_mode = ACCURACY`
- `ov::hint::inference_precision = f32`
- `ov::hint::performance_mode = LATENCY`
- `ov::hint::dynamic_quantization_group_size = 0`

TurboEmbed previously compiled with **no** hints (GPU default f16).
MiniLM embed cosine vs the Intel golden moved from ~0.99999827 to
**0.99999976** after the compile hints (same IR).

Berlin CE logits vs HF on this host after the FP32 IR:

| device | max abs logit err | previous FP16 IR |
|---|---|---|
| OPENVINO_GPU | 1.43e-6 | 8.32e-4 |
| OPENVINO_CPU | 3.34e-6 | (same band as GPU) |

## Commands

```bash
make probe-remote-usm            # writes intel-remote-usm-probe.txt
make convert-rerank-ov           # FP32 IR from pinned ONNX
make verify-rerank-ov            # SHA pin
make test-turborerank-intel      # includes probe + Berlin receipts
make test-turboembed-intel       # compile-hint GPU/CPU MiniLM
```

## What is not this item

CUDA / Metal wrap. Latency benches. Replacing turbo_buffer ZE with
plugin-owned OpenCL USM. NPU.
