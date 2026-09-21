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
  field index, because the graph fixes them at compile time.
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

## Run the live test

```bash
cargo run -p turbo-bundle -- import --source /path/to/all-MiniLM-L6-v2 \
    --output /path/to/bundles/minilm-onnx --license Apache-2.0 \
    --artifact onnx=/path/to/all-MiniLM-L6-v2/onnx/model.onnx --max-batch 32
TURBO_OPENVINO_LIB=$PWD/build/openvino/libturbo_provider_openvino.so \
TURBO_OPENVINO_BUNDLE=/path/to/bundles/minilm-onnx \
TURBO_OPENVINO_ORDINAL=0 \
cargo test -p turbo-conformance --test openvino_live -- --nocapture --test-threads=1
```

Receipts from real runs are under `testdata/receipts/turbo/`.
