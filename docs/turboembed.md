# turboembed — shared C ABI embeds

`include/turboembed.h` is the in-process catalog embed ABI. Clients call
`embed("minilm", text)` and get a real FP32 sentence vector. There is **no
mock path** for catalog aliases and **no silent CPU fallback**.

| arch | provider | crate feature | device string |
|---|---|---|---|
| nvidia | ONNX Runtime **CUDA EP** + **IoBinding device buffers** | `ort-cuda` | `CUDA` |
| intel | OpenVINO GenAI (other binary) | — | — |
| apple | MLX (other binary) | — | — |

Pooling for MiniLM is the sentence-transformers recipe: attention-mask-weighted
**mean** over tokens, then **L2** normalize. That math runs on the host after
the hidden-state tensor is copied back from CUDA. The graph itself runs on
GPU; a CPU-resident output tensor is a hard error.

## NVIDIA (krick) — live GPU proof

Host requirements are the same as `inferstream-nvidia` + `ort-cuda`: NVIDIA
driver for CUDA 13, and CUDA 13 user-space libs. Bundle them once:

```bash
scripts/fetch-runtime-libs.sh nvidia
```

Exact commands that produced `testdata/receipts/turboembed/nvidia-minilm.json`:

```bash
export LD_LIBRARY_PATH="$(pwd)/.libs/nvidia/lib:${LD_LIBRARY_PATH:-}"
cargo test -p turboembed --features ort-cuda -- --ignored --nocapture
```

Equivalent Make target (sets `LD_LIBRARY_PATH` for you):

```bash
make test-turboembed-nvidia
```

The ignored test:

1. Opens `Engine::open(nvidia)` — registers CUDA EP with `error_on_failure`,
   creates a CUDA device allocator, and refuses any catalog `device` other
   than `cuda`.
2. Calls Rust `engine.embed("minilm", text)` and the C ABI
   `turboembed_embed(..., "minilm", text, ...)` for every `parity:*` item in
   `testdata/e2e/goldens/nvidia/minilm.json`.
3. Requires cosine ≥ **0.99** vs the golden (384-d). Failures name the item.
4. Writes the receipt on success (`device=CUDA`, `dims`, `worst_cosine`,
   `pass=true`, git sha, the commands above).

Without `--features ort-cuda`, `Engine::open(nvidia)` and every catalog alias
error naming the feature. `backend = "mock"` in a catalog file is rejected.

## C ABI

```c
#include "turboembed.h"

char *err = NULL;
TurboEmbedEngine *eng = turboembed_create("nvidia", NULL, &err);
float *vec = NULL;
size_t dim = 0;
int rc = turboembed_embed(eng, "minilm", "hello world", &vec, &dim, &err);
/* dim == 384, device == "CUDA" */
turboembed_free(vec);
turboembed_destroy(eng);
```

Header: [`include/turboembed.h`](../include/turboembed.h).
Library: `cargo build -p turboembed --features ort-cuda` (`libturboembed.so`).
