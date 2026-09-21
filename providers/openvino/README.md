# OpenVINO provider

`libturbo_provider_openvino.so` implements `include/turbo/turbo_provider.h`
on OpenVINO. It is loaded at runtime through `turbo_runtime_load_provider`
or `turbo_runtime_desc.provider_paths`.

## What it offers

| device | tasks (text) | capability status |
|---|---|---|
| OpenVINO GPU (`GPU.N`, ordinal N) | embed, rerank, classify, token classify | EXPERIMENTAL |
| OpenVINO CPU (ordinal after the last GPU; explicit selection only) | same | EXPERIMENTAL |
| OpenVINO NPU | none offered | listed, not qualified |

- Models load from a bundle's `openvino_ir` or `onnx` artifact and need a
  WordPiece `tokenizer.json`; other tokenizer kinds are rejected (use the
  core tokenizer and `turbo_session_write_tokens`).
- Each session compiles the graph at its fixed `[max_batch, max_seq]` with
  the post-processing fused in: pooling and L2 for embedders, sigmoid for
  rerankers, softmax for classifiers, per-token softmax for token classifiers.
- Options honored: `truncate`, `max_tokens`, `prompt_role`, rerank `top_n`
  and `return_sorted`, token-classification `aggregation`. Options not
  honored (`normalize`, `pooling`, `output_dim`, `output_dtype`,
  `raw_scores`) are rejected with `TURBO_E_UNSUPPORTED_OPTION` and the
  field index, because the graph fixes them at compile time. `raw_scores`
  is the one exception: a bundle that declares no activation already
  produces logits, so it is accepted there. A value that is not one of the
  `TURBO_TRUNCATE_*`, `TURBO_PROMPT_*`, or `TURBO_AGGREGATE_*` constants
  this build knows is `TURBO_E_INVALID_ENUM` with the field index, never a
  default.
- Pair truncation (`turbo_session_write_pairs`): `MODEL` is the tokenizer's
  longest-first rule, `RIGHT` keeps the query whole and truncates the
  document, `NONE` fails instead of dropping tokens, and `LEFT` is refused
  (field 2) because it would drop the `[CLS]` and the query.
- Token-classification spans are word-aligned. A word that truncation cut in
  half, at either end of the row, is dropped rather than clipped: its label
  would otherwise come from a fragment. A group's score is the mean of its
  word scores, matching the CUDA provider.
- Caller memory is imported without a copy (`TURBO_CAP_HOST_PTR_IMPORT`,
  `TURBO_HANDLE_HOST_PTR` as `TURBO_PLACE_HOST`); the caller keeps ownership
  and the memory must outlive the buffer handle.
- On GPU the result stays in a device buffer (`TURBO_PLACE_DEVICE`,
  exportable as `TURBO_HANDLE_CL_MEM`) until `turbo_result_read`. Token
  rows are uploaded with one OpenCL write per input and counted in
  `h2d_bytes`; reads are counted in `d2h_bytes`. Tokenization is host work,
  so `fully_accelerated` is 0.

## Build

Needs an OpenVINO archive distribution (2026.3 or later), OpenCL headers
and loader, CMake 3.20+, and a C++17 compiler.

```bash
OV=/path/to/openvino_genai_ubuntu26_2026.3.1.0_x86_64
cmake -S providers/openvino -B build/openvino -DOpenVINO_DIR=$OV/runtime/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/openvino -j
```

The library's RUNPATH points at the OpenVINO runtime it was built against;
otherwise set `LD_LIBRARY_PATH` to `$OV/runtime/lib/intel64` and the TBB
directory under `$OV/runtime/3rdparty`.

## Run the tests

Two suites cover this provider. The Rust live tests are provider-neutral and
run against real bundles:

```bash
export LD_LIBRARY_PATH=$OV/runtime/lib/intel64:$OV/runtime/3rdparty/tbb/lib
TURBO_LIVE_LIB=$PWD/build/openvino/libturbo_provider_openvino.so \
TURBO_LIVE_PROVIDER=openvino \
TURBO_LIVE_BUNDLE=/path/to/bundles/minilm-onnx \
TURBO_LIVE_RERANK_BUNDLE=/path/to/bundles/rerank-onnx \
TURBO_LIVE_CLASSIFY_BUNDLE=/path/to/bundles/sst2-onnx \
TURBO_LIVE_NER_BUNDLE=/path/to/bundles/ner-onnx \
cargo test -p turbo-conformance --test live_embed --test live_tasks -- --test-threads=1 --nocapture
```

`tests/provider_test.cpp` builds alongside the library and drives the vtable
directly, which is the only way to reach what the core filters out first: a
caller's smaller `struct_size`, an unknown option enumeration, a `top_n`
above the row count, and the truncation cases where a word is cut in half. It
takes the same bundle variables and skips, saying so, when one is unset.

```bash
TURBO_LIVE_BUNDLE=... TURBO_LIVE_RERANK_BUNDLE=... TURBO_LIVE_NER_BUNDLE=... \
./build/openvino/turbo_provider_openvino_test
```

Set `TURBO_PROVIDER_LIB` to point the same binary at another build, and
`TURBO_LIVE_ORDINAL` to pick a device other than the CPU. Pass a substring as
the first argument to run one test.

Receipts from real runs are under `testdata/receipts/turbo/`.
