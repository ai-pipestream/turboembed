# turboembed — shared C ABI embeds

[`include/turboembed.h`](../include/turboembed.h) is the frozen in-process
catalog embed ABI (v1). Clients create an engine, `load_model("minilm")`,
and `embed_one` / `embed` to get a real FP32 sentence vector.

There is **no mock path** for catalog aliases and **no silent CPU
fallback**. Without a real provider feature the stub answers `mock-embed`
and returns `NOT_IMPLEMENTED` for `minilm` (and every other catalog name).

| arch | provider | crate feature | create device |
|---|---|---|---|
| nvidia | ONNX Runtime **CUDA EP** + **IoBinding device buffers** | `ort-cuda` | `TURBOEMBED_DEVICE_CUDA` |
| intel | `ov::genai::TextEmbeddingPipeline` on `"GPU"` | `genai` | `TURBOEMBED_DEVICE_OPENVINO_GPU` |
| apple | MLX (Swift `@_cdecl`, other binary) | — | `TURBOEMBED_DEVICE_METAL` |

Pooling for MiniLM is the sentence-transformers recipe: attention-mask-weighted
**mean** over tokens, then **L2** normalize. On NVIDIA that math runs on the
host after the hidden-state tensor is copied back from CUDA. The graph itself
runs on GPU; a CPU-resident output tensor is a hard error.

## NVIDIA (krick) — live GPU proof

Host requirements are the same as `inferstream-nvidia` + `ort-cuda`: NVIDIA
driver for CUDA 13, and CUDA 13 user-space libs. Bundle them once:

```bash
scripts/fetch-runtime-libs.sh nvidia
```

Exact commands that produce `testdata/receipts/turboembed/nvidia-minilm.json`:

```bash
export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
cargo test -p turboembed --features ort-cuda -- --ignored --nocapture
```

Equivalent Make target (sets `LD_LIBRARY_PATH` for you):

```bash
make test-turboembed-nvidia
```

The ignored tests:

1. `Engine::create(Device::Cuda)` then `load_model("minilm")` — registers
   CUDA EP with `error_on_failure`, creates a CUDA device allocator, copies
   inputs onto `AllocationDevice::CUDA`, binds the output to CUDA via
   IoBinding, and errors if the output is CPU-accessible.
2. `embed_one("minilm", text, mean+L2)` for `"hello world"` and every
   `parity:*` item in `testdata/e2e/goldens/nvidia/minilm.json`.
3. Requires cosine ≥ **0.99** vs the golden (384-d). Failures name the item.
4. Requires `/proc/self/maps` to contain `libonnxruntime_providers_cuda` and
   `libcudart`, and to omit `libpython`.
5. Also runs the raw C ABI (`turboembed_engine_create` /
   `turboembed_load_model` / `turboembed_embed_one`).
6. Writes the receipt on success (`device=CUDA`, `dims`, `worst_cosine`,
   `pass=true`, git sha, the commands above).

Without `--features ort-cuda`, `load_model("minilm")` on a CUDA engine
returns `NOT_IMPLEMENTED` and names `--features ort-cuda`.
`Engine::create(Device::Cpu)` + `load_model("minilm")` is
`UNSUPPORTED_DEVICE` when the feature is on (CPU is not success).
`backend = "mock"` in a catalog file is rejected.

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
