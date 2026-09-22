# Metal provider

`libturbo_provider_metal.dylib` runs BERT-family embedding and cross-encoder
reranking models on Apple GPUs through Metal directly: no MLX, no Core ML,
no shader toolchain. The kernels in `src/kernels.inc` (Metal Shading
Language, ported from the 2026-09-21 proof of concept) are compiled by the
provider when it loads, with `MTLMathModeSafe` so the arithmetic matches
the FP32 references. Weights come from an F32 `safetensors` checkpoint in
the bundle and the architecture from the model's own `config.json`
(`hf_config` artifact); the two are cross-checked at load.

This is the "fastest allowed" layer for the platform in the sense PLAN.md
uses: the GPU's own API, with the tokenizer on the host and every other
stage (encode, pool, normalize, the NSP pooler and the classifier head) on
the GPU. `fully_accelerated` is 0 because of the tokenizer.

## What unified memory changes

Apple GPUs share the SoC's memory with the CPU. The provider therefore:

- reports `TURBO_DEVICE_IGPU` with `TURBO_CAP_DEVICE_RESULT` and
  `TURBO_CAP_UNIFIED_MEMORY`;
- allocates `TURBO_PLACE_SHARED` (an `MTLBuffer` with
  `MTLResourceStorageModeShared`) and `TURBO_PLACE_HOST`; `DEVICE` and
  `PINNED` are refused with `TURBO_E_UNSUPPORTED_PLACEMENT` because they are
  not distinct placements on this hardware;
- writes token rows straight into the shared buffers the kernels read, and
  hands results back as `SHARED` buffers whose `host_ptr` is the GPU's
  memory, so `h2d_bytes` and `d2h_bytes` stay at 0;
- exports a shared result as `TURBO_HANDLE_MTL_BUFFER` (the `id<MTLBuffer>`)
  or `TURBO_HANDLE_HOST_PTR` (its contents), and imports host pointers.

## Bundles

```sh
turbo-bundle import --source ~/opt/models/all-MiniLM-L6-v2 --output ~/opt/bundles/minilm-metal --license Apache-2.0 \
    --artifact safetensors=~/opt/models/all-MiniLM-L6-v2/model.safetensors --artifact hf_config=~/opt/models/all-MiniLM-L6-v2/config.json
turbo-bundle import --source ~/opt/models/ms-marco-MiniLM-L-12-v2 --output ~/opt/bundles/rerank-metal --license Apache-2.0 \
    --artifact safetensors=~/opt/models/ms-marco-MiniLM-L-12-v2/model.safetensors --artifact hf_config=~/opt/models/ms-marco-MiniLM-L-12-v2/config.json
```

The checkpoint must be F32 (`TURBO_E_UNSUPPORTED_DTYPE` otherwise), the
config's `model_type` must be `bert` with the erf GELU, and
`num_hidden_layers` must equal the layers the checkpoint holds. Tensor
names with or without the `bert.` prefix are both accepted. A reranker
needs `classifier.weight` of shape `[1, hidden]`; the NSP pooler is used
when present.

## Build and test

```sh
make -C providers/metal                      # build/metal/libturbo_provider_metal.dylib
TURBO_LIVE_BUNDLE=~/opt/bundles/minilm-metal TURBO_LIVE_RERANK_BUNDLE=~/opt/bundles/rerank-metal \
    make -C providers/metal test             # 22 vtable-level cases
TURBO_LIVE_LIB=build/metal/libturbo_provider_metal.dylib TURBO_LIVE_PROVIDER=metal \
    TURBO_LIVE_BUNDLE=~/opt/bundles/minilm-metal cargo test -p turbo-conformance --test live_embed
```

The vtable tests need only `clang++`, `make`, and the Command Line Tools.
Without a Metal device they report the provider's reason and skip.

## Limits

- BERT-family encoders only (`model_type: bert`, erf GELU), at most 24
  layers, FP32 weights. Other checkpoints are refused with a message naming
  what was found.
- `output_dtype` F16 and I8 are `TURBO_E_UNSUPPORTED_OPTION` on field 8.
- Left truncation of query/document pairs is not offered (field 2).
- Rows are encoded one after another in a single command buffer; the
  kernels are straightforward (one thread per output element), so this is
  correct and device-resident rather than tuned. Receipts under
  `testdata/receipts/turbo/bench/` state what it measures.
