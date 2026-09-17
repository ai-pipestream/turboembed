# turboembed — shared C ABI embeds

[`include/turboembed.h`](../include/turboembed.h) is the frozen in-process
catalog embed ABI (v1). Clients create an engine, `load_model("minilm")`,
and `embed_one` / `embed` to get a real FP32 sentence vector.

There is **no mock path** for catalog aliases and **no silent device
swap**. A CUDA request never becomes CPU. Explicit `TURBOEMBED_DEVICE_CPU`
is a real CPU EP path. Without a real provider feature the default no-feature link answers
`mock-embed` and returns `NOT_IMPLEMENTED` for `minilm` (and every other
catalog name). Mock is ABI smoke only; catalog aliases never become 8-d
FNV.

| arch | provider | crate feature | create device |
|---|---|---|---|
| nvidia | ONNX Runtime **CUDA EP** + IoBinding, or explicit **CPU EP** | `ort-cuda` | `TURBOEMBED_DEVICE_CUDA` / `AUTO` or `TURBOEMBED_DEVICE_CPU` |
| nvidia | ONNX Runtime **TensorRT EP** (same MiniLM ONNX, CUDA IoBinding buffers) | `ort-cuda` | `TURBOEMBED_DEVICE_TENSORRT` |
| intel | `ov::genai::Tokenizer` + CompiledModel on `"GPU"` / `"CPU"` (ZE SHARED / HOST USM); `"NPU"` create fails loud until a host lists the NPU plugin | `genai` | `TURBOEMBED_DEVICE_OPENVINO_GPU` / `_CPU` / `_NPU` |
| apple | MLX (Swift `@_cdecl`, other binary) | — | `TURBOEMBED_DEVICE_METAL` |

Pooling for MiniLM is the sentence-transformers recipe: attention-mask-weighted
**mean** over tokens, then **L2** normalize. On NVIDIA CUDA that math runs on
DEVICE into a mapped PINNED 384-d row (`d2h_hidden_bytes` == 0). The graph
itself runs on GPU; a CPU-resident output tensor is a hard error. Explicit
CPU EP still pools on the host from rented HOST hidden.

## NVIDIA M4 qualification

The NVIDIA provider passed the roadmap M4 gates on Machine A on 2026-09-16:
matched native overhead vs direct ORT CUDA, allocator isolation under two
live engines, an installable `turboembed-cuda-sdk` archive with a
clean-consumer acceptance, and a second qualified model contract
(`bge-small`, CLS+L2). See
[nvidia-m4-qualification-2026-09-16.md](nvidia-m4-qualification-2026-09-16.md)
for receipts, `make bench-nvidia-overhead`, `make nvidia-sdk-release`, and
`make nvidia-sdk-acceptance`. The prepared `turboembed_prepared_v1_*` SDK
remains Intel-only; this qualification is on the `turboembed.h` ABI below.

## NVIDIA (Machine A) — live GPU proof

Host requirements are the same as `inferstream-nvidia` + `ort-cuda`: NVIDIA
driver for CUDA 13, and CUDA 13 user-space libs. Bundle them once:

```bash
scripts/fetch-runtime-libs.sh nvidia
```

Exact commands that produce `testdata/receipts/turboembed/nvidia-minilm.json`:

```bash
export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
cargo test -p turboembed --features ort-cuda -- --include-ignored --nocapture
```

Equivalent Make target (sets `LD_LIBRARY_PATH` for you):

```bash
make test-turboembed-nvidia
```

The ignored tests:

1. `Engine::create(Device::Cuda)` then `load_model("minilm")` — creates a
   CUDA `turbo_buffer` arena, registers CUDA EP with `error_on_failure`
   and `gpu_external_alloc` → DEVICE rent, writes tokens into PINNED
   mapped rows, binds those + a DEVICE hidden view via IoBinding, and
   errors if the bound output is not CUDA. After warmup, embed must see
   arena allocs == 0. Mean+L2 (mask-weighted) runs on DEVICE into the
   mapped PINNED result row (`d2h_hidden_bytes` == 0).
2. `embed_one("minilm", text, mean+L2)` for `"hello world"` and every
   `parity:*` item in `testdata/e2e/goldens/nvidia/minilm.json`.
3. Requires cosine ≥ **0.99** vs the golden (384-d). Failures name the item.
4. Requires `/proc/self/maps` to contain `libonnxruntime_providers_cuda` and
   `libcudart`, and to omit `libpython`.
5. Also runs the raw C ABI (`turboembed_engine_create` /
   `turboembed_load_model` / `turboembed_embed_one`).
6. Writes the receipt on success (`device=CUDA`, `dims`, `worst_cosine`,
   `pass=true`, git sha, the commands above).

Without `--features ort-cuda`, `load_model("minilm")` returns
`NOT_IMPLEMENTED` and names `--features ort-cuda`.
`Engine::create(Device::Cuda)` + `load_model("minilm")` fails loud if
the CUDA EP is missing (CPU is not a fallback).
`Engine::create(Device::Cpu)` + `load_model("minilm")` is a real CPU EP
session (same ONNX, same mean+L2). Receipt:
`testdata/receipts/turboembed/nvidia-minilm-cpu.json`.
`backend = "mock"` in a catalog file is rejected.

`Engine::create(Device::TensorRt)` + `load_model("minilm")` is the ORT
TensorRT EP (`error_on_failure`). Same ONNX as CUDA. Missing
`libnvinfer.so.10` / `libnvonnxparser.so.10` is a hard error — not a
CUDA-only or CPU session. Fetch the SONAMEs (opt-in, ~3.7 GiB):

```bash
scripts/fetch-runtime-libs.sh nvidia-trt
```

Live receipt on Machine A (RTX 4080 SUPER): cosine ~1.0 vs nvidia MiniLM
goldens, dim 384, `/proc/self/maps` contains
`libonnxruntime_providers_tensorrt` + `libnvinfer`, no `libpython`.
File: `testdata/receipts/turboembed/nvidia-minilm-tensorrt.json`.

```bash
export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
cargo test -p turboembed --features ort-cuda --test nvidia_minilm \
  -- --ignored --nocapture minilm_ort_tensorrt_matches_golden
```

Anti-mock: `tensorrt_never_silent_cuda_cpu_or_fnv8` (create/list; no
engine compile) plus the ignored receipt test.

## C ABI

```c
#include "turboembed.h"

turboembed_engine *eng = NULL;
turboembed_engine_create(TURBOEMBED_DEVICE_CUDA, NULL, &eng);
turboembed_load_model(eng, "minilm", 6);

const char *text = "hello world";
turboembed_embed_result *out = NULL;
turboembed_embed_one(eng, "minilm", 6, text, 11, NULL, &out);
/* out->dim == 384, values are mean+L2 FP32 */
turboembed_embed_result_free(out);
turboembed_engine_destroy(eng);
```

Rust wrapper:

```rust
let engine = turboembed::Engine::create(turboembed::Device::Cuda)?;
engine.load_model("minilm")?;
let opts = turboembed::EmbedOptions {
    pooling: turboembed::Pooling::Mean,
    normalize: Some(true),
    ..Default::default()
};
let row = engine.embed_one("minilm", "hello world", &opts)?;
assert_eq!(row.dim(), 384);
```

Header: [`include/turboembed.h`](../include/turboembed.h) (Apple copy must
stay identical). Library: `cargo build -p turboembed --features ort-cuda`.
