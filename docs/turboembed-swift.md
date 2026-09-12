# TurboEmbed on Apple (Swift)

Apple exposes the **same C ABI** as Linux. The header is
[`include/turboembed.h`](../include/turboembed.h). The Swift package keeps
an identical copy at `swift/Sources/TurboEmbedC/include/turboembed.h`
(the `turboembed` crate tests they match).

On this host the symbols come from **`libTurboEmbed.dylib`**, not the
C++ stub. `crates/turboembed/build.rs` on macOS runs
`swift build --product TurboEmbed` and **refuses** to link
`native/turboembed/src/stub.cpp` (that would hide a fake MiniLM).

## How Swift exports C

1. **`@_cdecl("turboembed_engine_create")`** (and the rest of the header)
   in `swift/Sources/TurboEmbed/ABI.swift`.
2. **Metal MiniLM** in `MlxProvider.swift` → `MlxEngine.embed`
   (mlx-swift `MLXEmbedders` on `Device.gpu`).
3. **`mock-embed`** stays for ABI smoke only (8-d FNV). Catalog aliases
   never take that path.

## Device policy

| requested | if missing | fallback |
|---|---|---|
| `METAL` / `AUTO` | `UNAVAILABLE` | **none** — never CPU |
| `CUDA` / `TENSORRT` / `OPENVINO_GPU` / `NPU` | `UNSUPPORTED_DEVICE` | **none** — this dylib is Metal-only |
| `CPU` / `OPENVINO_CPU` | (explicit) | mock-embed only; MiniLM not served |
| `MOCK` | (explicit smoke) | 8-d FNV |

`AUTO` means host-default **GPU** (Metal here), not "CPU if GPU is down".
`MlxEngine` is constructed only on the Metal path (`Device.gpu`). Explicit
CPU never calls `Device.gpu`.

Do not also link `native/turboembed/src/stub.cpp` into this module.

## Pooling — fail loud if fake

mlx-swift-lm BERT returns `pooledOutput = tanh(pooler(CLS))` (NSP head).
`Pooling.Strategy.cls` uses that tensor. sentence-transformers MiniLM
does **not**: `1_Pooling/config.json` is `pooling_mode_mean_tokens`, then
L2 (`2_Normalize`). Extra `applyLayerNorm: true` also scrambles the
384-d space (apple↔nvidia cosine ≈ 0).

`MlxEngine.poolHidden` is the system of record:

* **mean** — mask-weighted last hidden states (MiniLM)
* **cls** — first-token hidden state (BGE), **not** `pooledOutput`
* L2 when `normalize=true`
* never an extra LayerNorm

`turboembed_embed(..., "minilm", ...)` with default / MEAN options
calls that path. Dim ≠ 384 or dim == 8 is `INTERNAL` (`FAKE`).

## Package targets

| target | role |
|---|---|
| `TurboEmbedC` | Clang module: the frozen header only |
| `TurboEmbed` | **dynamic** `libTurboEmbed.dylib`: `@_cdecl` + MLX + mock-embed |

`inferstream-apple` still uses `MlxEngine` in-process (no C ABI). The
dylib is for Rust / C callers.

## Proof on a Mac

Weights: `make fetch-mlx ALIASES=minilm` → `models/mlx/minilm`
(FP `sentence-transformers/all-MiniLM-L6-v2`, not 4-bit).

```bash
# Rust → turboembed_embed(minilm) → Metal. Writes the receipt.
make test-turboembed-apple
# or:
cargo test -p turboembed --features mlx-live -- --ignored --nocapture apple_minilm
```

Receipt: [`testdata/receipts/turboembed/apple-minilm.json`](../testdata/receipts/turboembed/apple-minilm.json).
Floors: vs nvidia goldens **≥ 0.97**; vs apple goldens **≥ 0.99**.
A BERT-pooler regression lands near cosine 0 and the test panics.

No Python. `mlx.metallib` is built by `scripts/build-apple-metallib.sh`
next to the dylib (same as the Swift server).

## Linux CI

`swift build` is not required. `build.rs` compiles the C++ stub. Catalog
aliases stay `NOT_IMPLEMENTED` there.
