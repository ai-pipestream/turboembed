# Providers

A provider is a set of devices plus the ability to run tasks on them, reached
through the plugin vtable in `include/turbo/turbo_provider.h`. Four
providers exist in this tree today: `mock` and `static` (Rust, built into
`libturbo`) and `openvino` (C++) and `cuda` (Rust), each built separately
and loaded as a plugin library. Every other row below is planned; see
`PLAN.md` sections 7 and 10 for the full hardware/runtime detail and
milestone gates this table summarizes.

## Provider table

| provider | status | hardware / machine | runtime | lowest layer used |
|---|---|---|---|---|
| `mock` | supported for contract testing only | any host; two synthetic devices (CPU ordinal 0, Accel ordinal 1) | none (pure Rust) | deterministic hash-derived math on host; serves only bundles with a `mock` artifact |
| `static` | EXPERIMENTAL | any CPU; explicit selection only | none (pure Rust, model2vec-style static token embeddings) | table lookup + mean + L2 on host; one capability cell `EMBED x TEXT x CPU`; tokenizes through the Hugging Face `tokenizers` crate |
| `cpu` | PLANNED (folded into the CUDA/ORT provider, P3) | any; explicit selection only, never `AUTO` | ORT 1.30 CPU EP; ggml CPU for GGUF | host arena, write-through tokens |
| `openvino` | EXPERIMENTAL (GPU and CPU); NPU listed, not qualified | `krick-1` Battlemage B70 (GPU), any CPU; Intel NPU when available | OpenVINO 2026.3.1 | `ov::Core` compiled model with pooling/normalization/post-processing fused into the graph; `cl_mem` remote tensors on GPU; native WordPiece (`native/wordpiece/`) |
| `cuda` | EXPERIMENTAL on `krick` (x86_64); Jetson (`nano1`, aarch64) not started | `krick` RTX 4080 SUPER (x86_64, landed); `nano1` Orin Nano Super (aarch64, planned) | ONNX Runtime 1.28 CUDA execution provider (`ort` crate prebuilt CUDA 13 bundle) | IoBinding on pinned/device arena; device kernels (`kernels.cu`) for mean/CLS/last pooling, L2 normalization, sigmoid, and softmax |
| `metal` | PLANNED (P4) | Apple M2 | MLX 0.32 via mlx-swift 0.31 | MLX arrays over Metal shared buffers, no-copy construction; pooling and L2 as MLX ops |
| `hailo` | PLANNED (P5) | two Pis with Hailo-8; one Pi with Hailo-10H; x86_64 hosts with a PCIe Hailo-8 card | HailoRT 4.24.0 / 5.4.0 | `VDevice` + `InferModel` with `dma_map` zero-copy on page-aligned arena rows |
| `ggml` | PLANNED (P6) | every machine | llama.cpp v0.4.1 / ggml 0.24 (CUDA, SYCL, Metal, CPU backends) | `ggml_backend_dev` registry; `llama_batch` decode; embeddings copied once from `llama_get_embeddings_seq` |
| `hailo` GenAI | PLANNED (P6) | Hailo-10H Pi | `hailort::genai::LLM` (HailoRT 5.4) | native LLM on the NPU with its own sampler |

`PLAN.md` section 7 lists the CUDA provider's planned runtime as ORT 1.30;
the landed build pins ONNX Runtime 1.28 through the `ort` crate's prebuilt
CUDA 13 bundle (`providers/cuda/Cargo.toml`, `providers/cuda/README.md`).

`mock` (`crates/turbo-core/src/mock.rs`) and `static` (`providers/static`)
are both built into `libturbo` (`turbo::builtin_providers()`) and are also
packaged as separately loadable libraries under `providers/mock` and
`providers/static` for exercising the plugin-loading path against the same
code the core links in. `openvino` (`providers/openvino`) only exists as a
loadable library, `libturbo_provider_openvino.so`, built with CMake against
an OpenVINO SDK; see `providers/openvino/README.md` for the build and the
live-test commands, and `testdata/receipts/turbo/openvino-minilm-2026-09-21.json`
and `openvino-tasks-2026-09-21.json` for the conformance and precision
receipts behind its `EXPERIMENTAL` status. `cuda` (`providers/cuda`) also
only exists as a loadable library, `libturbo_provider_cuda.so`, built with
`cargo build -p turbo-provider-cuda` (it is a Rust provider using
`turbo_core::export_provider!`, like `mock` and `static`, but is not part of
`turbo::builtin_providers()` because it needs a CUDA toolkit to build); see
"The CUDA provider" below, `providers/cuda/README.md` for the build and the
live-test commands, and `testdata/receipts/turbo/cuda-2026-09-21.json` for
the receipt behind its `EXPERIMENTAL` status. All four report a status of
`EXPERIMENTAL` or better only for cells with a receipt; none has a
matched-native benchmark receipt yet, so none is `SUPPORTED`
(`PLAN.md` section 2, item 7).

`static`, `cpu`, and `openvino`'s CPU device are explicit-selection-only:
`AUTO` never returns a CPU-kind device (`PLAN.md` section 2, item 4). On
OpenVINO, GPU device ordinals come first (`GPU.0`, `GPU.1`, ...), then CPU,
then NPU (`providers/openvino/src/provider.cpp`).

`docs/c-api.md`'s "Which providers set which bits" table is the full bit
matrix for all four; in outline: `mock` (`MOCK_CAPS`,
`crates/turbo-core/src/mock.rs`) sets the most bits, including every
`GENERATE` option bit (it is the only provider offering `GENERATE`) and
`OPT_NORMALIZE`/`OPT_RAW_SCORES`, but not `DEVICE_RESULT`,
`OPT_POOLING_OVERRIDE`, or `OPT_OUTPUT_DTYPE`. `static` (`STATIC_CAPS`,
`providers/static/src/lib.rs`) sets only the `EMBED`-relevant bits it needs
for its one capability cell (no `OPT_TOP_N`/`OPT_AGGREGATION`/`OPT_RAW_SCORES`,
since it offers neither `RERANK` nor `CLASSIFY`). `openvino`
(`kCapsCommon`/`caps_of`, `providers/openvino/src/provider.cpp`) sets
`OPT_TRUNCATE`, `OPT_MAX_TOKENS`, `OPT_PROMPT_ROLE`, `OPT_TOP_N`,
`OPT_AGGREGATION`, and `DETERMINISTIC` on every non-NPU device, plus
`DEVICE_RESULT` only on GPU/iGPU; it sets neither `OPT_NORMALIZE` nor
`OPT_POOLING_OVERRIDE` (pooling and normalization always follow the bundle
contract) nor `DEVICE_POSTPROCESS`. `cuda` (`CUDA_CAPS`,
`providers/cuda/src/lib.rs`) is the only provider setting
`OPT_POOLING_OVERRIDE` and `DEVICE_POSTPROCESS`; see "The CUDA provider"
below for its full list. The NPU device OpenVINO enumerates reports `caps =
0` (listed, not qualified).

## Writing a provider

There are two routes to the same vtable
(`turbo_provider_vtbl` in `include/turbo/turbo_provider.h`); either is
loaded the same way, with `turbo_runtime_load_provider` or
`turbo_runtime_desc.provider_paths`.

### Rust route: implement the core traits

From `crates/turbo-core/src/provider.rs` (see also `AGENTS.md`'s "Adding a
provider" section):

- `Provider`: `id()`, `version()`, `devices()` (enumerate hardware; a probe
  failure is recorded and logged, not a panic), `capability(ordinal, task,
  modality)`, `can_run(ordinal, bundle, task, modality)`, `create_context`.
- `ProviderContext`: `alloc`, `import`, `load_model`.
- `ProviderModel`: `info()`, `create_session`, and `create_generation` for
  generative models (default: `TURBO_E_UNSUPPORTED_TASK`).
- `ProviderSession`: `write_text`, `write_tokens`, `write_pairs`,
  `write_text_classify`, `bind`, `run`, `stats`. A provider only overrides
  the write methods for the tasks its capability matrix actually offers;
  every other method's default already fails loudly.
- `ProviderGeneration`: `prompt`, `prompt_tokens`, `step`, `cancel`.

Export the implementation with `turbo_core::export_provider!(c"id", c"version",
|| Arc::new(YourProvider::new()))` (`crates/turbo-core/src/plugin_export.rs`);
this defines the `#[no_mangle] turbo_provider_get` symbol, builds the vtable,
and wraps every entry point in `catch_unwind` so a Rust panic cannot cross
the plugin boundary. `providers/mock`, `providers/static`, and
`providers/cuda` are the examples; each is a crate whose `lib.rs` ends with
one `export_provider!` call.

### C/C++ route: implement the vtable directly

A provider in another language implements `turbo_provider_vtbl` and exports
`turbo_provider_get(uint32_t core_abi_version)` itself, returning `NULL` for
an unsupported ABI version. `providers/openvino/src/provider.cpp` is the
example: it builds the vtable as a `static const turbo_provider_vtbl`, wraps
every entry point in a `try`/`catch` that converts C++ exceptions
(`turbo_ov::Failure`, carrying a status code and field index) into
`turbo_error`, and never lets an exception cross into the caller.

Ownership rules from `turbo_provider.h`'s header comment, binding on both
routes:

- Provider functions must never unwind or throw across the plugin boundary;
  every entry point returns a status code and fills the caller's
  `turbo_error` instead.
- `struct_size` versioning applies inside the vtable's structs
  (`turbo_provider_buffer`, `turbo_provider_output`, `turbo_provider_result`)
  exactly as it does at the public API: read only fields below the size the
  core declared.
- All vtable function pointers are required except those the header marks
  optional (`buffer_import`, `buffer_export`, `generation_create` — `NULL`
  means "the provider does not offer this"); the core rejects a vtable with
  a `NULL` required entry at load time with `TURBO_E_PROVIDER_LOAD`.
- A `turbo_provider_buffer` or `turbo_provider_output` returned from
  `buffer_alloc`/`buffer_import`/`session_run` is borrowed: the provider
  owns the underlying memory until the core calls `buffer_release`, and
  `session_run`'s output arrays are borrowed only until the next `run` or
  `session_release` on that session, never released by the core directly.
- `context_release`, `model_release`, `session_release`, `buffer_release`,
  `generation_release`, and `generation_cancel` return `void` and must
  accept whatever the corresponding `_create`/`_alloc`/`_load` call handed
  back exactly once; they are not passed `NULL` by the core (the core's own
  handles are never null when a release is due).
- The provider's own state (the vtable's `void *state` and every handle it
  hands back as `void *`) must be safe to use from any thread for the
  provider-level functions (`device_count`, `device_info`, `capability`,
  `can_run`, `context_create`); session and generation handles follow the
  single-owner rule the core already enforces (one call in flight at a
  time), so a provider does not need its own locking for those beyond not
  racing itself.

The provider vtable is defined at task granularity, not graph granularity
(`PLAN.md` section 4.7): a provider is free to run `embed`, `rerank`,
`classify`, `token_classify`, `generate`, and `run` as one fused device
pipeline. `turbo-core`'s own tokenize-then-run-then-pool sequence is a
fallback for stages a provider declines to implement, never a hidden
default a provider's own fused path is measured against. The OpenVINO
provider is the working example: each session compiles one graph with
pooling and L2 fused in for embedders, sigmoid for rerankers, softmax for
classifiers, and per-token softmax for token classifiers
(`providers/openvino/README.md`), and only tokenization stays on the host
(`TURBO_STAGE_HOST`) because WordPiece has no GPU kernel in this tree.

## The CUDA provider

`providers/cuda` (`turbo-provider-cuda`) runs `EMBED`, `RERANK`, `CLASSIFY`,
and `TOKEN_CLASSIFY` on `TEXT` bundles with an `onnx` artifact and a BERT
WordPiece tokenizer, through the ONNX Runtime CUDA execution provider
(`providers/cuda/src/lib.rs`). One device per CUDA ordinal, kind GPU; the
CPU execution provider is never registered, so a missing CUDA library fails
context creation (`TURBO_E_DEVICE_UNAVAILABLE`) rather than falling back to
the host.

Data path: token rows are tokenized natively on the host
(`providers/cuda/src/wordpiece.rs`) into pinned staging, compacted to the
longest live row, and copied to the session's own device input buffers in
one H2D copy per input tensor (`input_ids`, `attention_mask`, optionally
`token_type_ids`). The model runs through IoBinding with a device-bound
output. Pooling (mean, CLS, or last) with optional L2 normalization and
`output_dim` truncation, sigmoid (rerank/classify raw scores), and per-row
softmax (classify/token-classify) are the provider's own kernels
(`providers/cuda/src/kernels.cu`), launched on the session's stream directly
from the model output; token-classify span aggregation runs on the host
(word-aligned, mean score across a word's sub-tokens). Results stay on the
device and are exported as `TURBO_HANDLE_CUDA_PTR`
(`TURBO_CAP_DEVICE_RESULT`); a host read is explicit and its bytes are
counted in `d2h_bytes`.

Stage placement (`stage_placement` in `turbo_model_info`): `TOKENIZE` is
`HOST`; `ENCODE` is `DEVICE`; `POOL` and `NORMALIZE` are `DEVICE` for
embedding models and unused otherwise; `POSTPROCESS` is `DEVICE` for rerank
and classify (the activation kernel) and `HOST` for token-classify (span
aggregation reads the device output back once). `fully_accelerated` is
therefore `0`: tokenization, and token-classify aggregation, never move off
the host in this build.

Capability bits every CUDA device reports (`CUDA_CAPS`,
`providers/cuda/src/lib.rs`): `HOST_PTR_IMPORT`, `DEVICE_RESULT`,
`DYNAMIC_SHAPE`, `WEIGHT_SHARING`, `DEVICE_POSTPROCESS`, and
`OPT_TRUNCATE`, `OPT_MAX_TOKENS`, `OPT_PROMPT_ROLE`, `OPT_NORMALIZE`,
`OPT_POOLING_OVERRIDE`, `OPT_OUTPUT_DIM`, `OPT_TOP_N`, `OPT_AGGREGATION`,
`OPT_RAW_SCORES`. Options honored: `truncate` (including `LEFT` on
`write_text`), `max_tokens`, `prompt_role`, `normalize`, `pooling` override,
`output_dim`, `top_n` and `return_sorted` for rerank, `aggregation` for
token-classify, and `raw_scores` for rerank and classify (`TURBO_CAP_OPT_RAW_SCORES`,
`crates/turbo-core/src/handles.rs` gates it centrally the same way for every
provider). Options rejected with `TURBO_E_UNSUPPORTED_OPTION` and the field
index: `output_dtype` values other than the model's own `dtype_used` (always
`F32` for this provider; no `OPT_OUTPUT_DTYPE` bit), and `truncate = LEFT`
on `write_pairs` (rerank pairs only support `NONE` and the model's
longest-first/query-priority policies). `max_tokens` above the session's
`max_seq` is `TURBO_E_CAPACITY` and a `prompt_role` the bundle has no prefix
for is `TURBO_E_INVALID_ARGUMENT`, the same unconditional checks every
provider gets from `turbo-core` (`docs/c-api.md`'s Session section).

Runtime dependencies: `libonnxruntime_providers_cuda.so` and
`libonnxruntime_providers_shared.so` next to the provider library, and the
CUDA 13 user-space libraries they link (cuBLAS, cuBLASLt, cuDNN 9, NVRTC,
the CUDA 13 runtime). These are found through the loader's normal search, or
preloaded from a directory named by the context option `cuda_lib_dir` or the
environment variable `TURBO_CUDA_LIB_DIR`; a library that never loads is
`TURBO_E_DEVICE_UNAVAILABLE` at context creation. See
`providers/cuda/README.md` for the build-time requirements (a CUDA toolkit
with `nvcc`) and the live-test command.

Build feature: `download` (on by default) has the `ort` crate fetch the
prebuilt ONNX Runtime 1.28 CUDA 13 bundle for `x86_64-unknown-linux-gnu` at
build time (network access needed on the first build). Building with
`--no-default-features` turns this off and instead links a local ONNX
Runtime through `ORT_LIB_LOCATION`; this is the Jetson (`aarch64`) path,
since there is no prebuilt CUDA bundle for that target, and it also avoids
the downloader's OpenSSL dependency. See `providers/cuda/README.md` for the
exact flags (`nano1` links ONNX Runtime 1.24.0 this way, `TURBO_CUDA_ARCHS=87`).

Status: `EXPERIMENTAL` on `krick` (RTX 4080 SUPER, x86_64) for all four
tasks; precision matches the FP32 reference vectors at cosine 1.000
(`testdata/receipts/turbo/cuda-2026-09-21.json`). Jetson (`nano1`, aarch64)
is not started (`PLAN.md` section 10, P3): there is no prebuilt
`aarch64-unknown-linux-gnu` ONNX Runtime CUDA bundle, so the board needs a
source build first. No cell is `SUPPORTED` yet: the matched-native benchmark
receipt is missing, same as `openvino` and `static`.

## Honesty rules

These are `PLAN.md` section 2's principles, applied specifically to what a
provider reports:

- **Stage placement.** `ModelInfo::stages` (`stage_placement[TOKENIZE|
  ENCODE|POOL|NORMALIZE|POSTPROCESS]`) must reflect where each stage
  actually ran (`HOST`, `DEVICE`, or `FUSED`), not where the provider
  intends it to run eventually. `fully_accelerated` is `1` only if every
  applicable stage is off the host.
- **Capability status.** `Provider::capability` returns one of
  `UNSUPPORTED`, `PLANNED`, `EXPERIMENTAL`, `SUPPORTED`. A cell is marked
  `SUPPORTED` only once it carries a conformance receipt, a precision
  receipt (cosine floor and max absolute error against a reference dtype),
  and a matched-native benchmark from a named machine (`PLAN.md` section 2,
  item 7, and section 11). `PLANNED` and `EXPERIMENTAL` are honest,
  non-blocking states to use instead of overclaiming; the `mock` provider's
  own capability cells return `SUPPORTED` because its outputs really are a
  deterministic, fully-tested function of the input, not because
  `SUPPORTED` is a convenient default.
- **Capability bits gate options, not silently ignore them.** Every
  `TURBO_CAP_OPT_*` bit either matches what the provider actually honors, or
  the corresponding option field must fail with
  `TURBO_E_UNSUPPORTED_OPTION` (`turbo-core` enforces the rejection side of
  this centrally in `Model::validate_*`; the provider only needs to
  advertise its bits honestly and implement what it advertises).
- **Device probing failures are visible, not swallowed.** `Runtime::register`
  records a provider's device-probe failure as a `ProviderFailure` and logs
  it; the provider stays registered with zero devices rather than the
  runtime silently pretending the provider does not exist.

See `docs/architecture.md` for how the capability matrix, memory model, and
provider contract fit together, and `docs/testing.md` for how to run the
conformance suite against a new provider (`TURBO_CONFORMANCE_PROVIDER_PATHS`
and friends).
