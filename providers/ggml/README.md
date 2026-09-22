# ggml provider

`turbo-provider-ggml` runs GGUF generation and GGUF embeddings through
llama.cpp (via the `llama-cpp-2` crate, which builds llama.cpp from source
with cmake). One
Turbo device per ggml backend device: the GPUs a compiled backend exposes,
plus the CPU, which `AUTO` never selects. See the crate documentation in
`src/lib.rs` for the data path and `PLAN.md` section 10 (P6) for scope.

## Building

```bash
cargo build -p turbo-provider-ggml                     # CPU backend only
CUDAHOSTCXX=/usr/bin/g++-13 cargo build -p turbo-provider-ggml --features cuda
cargo build -p turbo-provider-ggml --features metal    # macOS
```

The CUDA build needs a toolkit whose `nvcc` accepts the host compiler
(`CUDAHOSTCXX` points nvcc 12.4 at GCC 13 on `krick`, whose default GCC is
15) and a CUDA library directory the `llama-cpp-sys-2` build finds: on
`krick` the runtime lives in `/usr/lib/x86_64-linux-gnu`, so
`CUDA_LIBRARY_PATH` names a directory whose `lib64` links there
(`~/opt/cuda-sys`). `.cargo/config.toml` sets
`CMAKE_POSITION_INDEPENDENT_CODE=ON` for every build, because the llama.cpp
objects end up inside a shared library. CI builds the CPU backend only.

## Bundle

A generative bundle with a `gguf` artifact:

```bash
turbo-bundle import --source <dir with model.gguf> --output <bundle> \
  --license Apache-2.0 --model-id Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --kind generative --max-seq 4096 --artifact gguf=<dir>/model.gguf
```

`contract.max_seq` is the context length (it must not exceed the model's
training context). The chat template is `tokenizer.chat_template` from the
manifest when present, else the GGUF metadata's; a bundle with neither is
refused.

## Embedding bundles

An embedding bundle with a `gguf` artifact (a BERT-family encoder converted
by llama.cpp) is imported like an ONNX one; the sentence-transformers
configuration files next to the GGUF give the importer the pooling,
normalization, dimension, and sequence limit:

```bash
turbo-bundle import --source <dir with model.gguf and the ST config files> \
  --output <bundle> --license Apache-2.0 --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --artifact gguf=<dir>/model.gguf
```

Pooling runs in the llama.cpp graph (mean, CLS, or last, fixed when the
session's context is created, so a pooling override is not offered); L2
normalization and `output_dim` truncation run on the host. llama.cpp hands
the pooled vectors back in host memory on every device, so results are
`TURBO_PLACE_HOST` and the device does not advertise
`TURBO_CAP_DEVICE_RESULT`; the bytes it moved back are counted in
`d2h_bytes`. Tokenization is llama.cpp's own from the GGUF vocabulary, with
the model's special tokens; `truncate` cuts the content between them.

Honored: `truncate`, `max_tokens`, `prompt_role`, `normalize`,
`output_dim`; refused naming the field: `pooling` other than the contract's
and `output_dtype` other than `MODEL`/`F32`.

## Options

Honored (each behind its capability bit): `temperature`, `top_k`, `top_p`,
`min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`,
`logit_bias`, `logprobs`, `stop` strings, `stop_tokens`, `min_new_tokens`,
`echo`, `seed`, and `structured_kind = GRAMMAR` with a GBNF grammar.
Refused naming the field: `structured_kind = JSON_SCHEMA`, `n_sequences >
1`, `tools`. `max_new_tokens = 0` means 512.

## Running the live tests

```bash
TURBO_LIVE_LIB=$PWD/target/debug/libturbo_provider_ggml.so TURBO_LIVE_PROVIDER=ggml \
TURBO_LIVE_GGUF_BUNDLE=~/opt/bundles/qwen05-gguf \
cargo test -p turbo-conformance --test live_generate -- --test-threads=1
TURBO_LIVE_LIB=$PWD/target/debug/libturbo_provider_ggml.so TURBO_LIVE_PROVIDER=ggml \
TURBO_LIVE_BUNDLE=~/opt/bundles/minilm-gguf \
cargo test -p turbo-conformance --test live_embed -- --test-threads=1
```

`TURBO_LIVE_ORDINAL` selects a device (default: the CPU device).

## Status

Every cell is `EXPERIMENTAL`. Not yet: rerank and classification through
GGUF, tokenize/detokenize for GGUF vocabularies, JSON-schema constrained
output, and the throughput receipts.
