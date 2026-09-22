# Providers

A provider is a set of devices plus the ability to run tasks on them, reached
through the plugin vtable in `include/turbo/turbo_provider.h`. Five
providers exist in this tree today: `mock` and `static` (Rust, built into
`libturbo`) and `openvino` (C++), `cuda` (Rust), and `ggml` (Rust), each
(other than `mock`/`static`) built separately and loaded as a plugin
library. Every other row below is planned; see `PLAN.md` sections 7 and 10
for the full hardware/runtime detail and milestone gates this table
summarizes.

## Provider table

| provider | status | hardware / machine | runtime | lowest layer used |
|---|---|---|---|---|
| `mock` | supported for contract testing only | any host; two synthetic devices (CPU ordinal 0, Accel ordinal 1) | none (pure Rust) | deterministic hash-derived math on host; serves only bundles with a `mock` artifact |
| `static` | EXPERIMENTAL | any CPU; explicit selection only | none (pure Rust, model2vec-style static token embeddings) | table lookup + mean + L2 on host; one capability cell `EMBED x TEXT x CPU`; tokenizes through the Hugging Face `tokenizers` crate |
| `cpu` | PLANNED (folded into the CUDA/ORT provider, P3) | any; explicit selection only, never `AUTO` | ORT 1.30 CPU EP; ggml CPU for GGUF | host arena, write-through tokens |
| `openvino` | EXPERIMENTAL (GPU and CPU); NPU listed, not qualified | `krick-1` Battlemage B70 (GPU), any CPU; Intel NPU when available | OpenVINO 2026.3.1 | `ov::Core` compiled model with pooling/normalization/post-processing fused into the graph; `cl_mem` remote tensors on GPU; native WordPiece (`native/wordpiece/`) |
| `cuda` | EXPERIMENTAL on `krick` (x86_64); embedding landed on `nano1` (aarch64), task suite still being verified there | `krick` RTX 4080 SUPER (x86_64, landed); `nano1` Orin Nano Super (aarch64, embedding landed) | ONNX Runtime 1.28 CUDA execution provider on `krick` (`ort` crate prebuilt CUDA 13 bundle); ONNX Runtime 1.24.0 linked dynamically (`ORT_LIB_LOCATION`, `--no-default-features`) on `nano1` | IoBinding on pinned/device arena; device kernels (`kernels.cu`) for mean/CLS/last pooling, L2 normalization, sigmoid, and softmax |
| `metal` | PLANNED (P4) | Apple M2 | MLX 0.32 via mlx-swift 0.31 | MLX arrays over Metal shared buffers, no-copy construction; pooling and L2 as MLX ops |
| `hailo` | EXPERIMENTAL (Hailo-8) on `pi5ai1` and `cm5ai1`; Hailo-8L untested; Hailo-10H open | two Pis with Hailo-8 (landed); Hailo-8L boards; one Pi with Hailo-10H (needs a DFC 5 HEF); x86_64 hosts with a PCIe Hailo-8 card | HailoRT 4.23.0 (`hailo-all`) | `hailo_vdevice` bound to one device with the scheduler on; the HEF's fixed-shape encoder body through f32 vstreams; host WordPiece, word-embedding gather from the `hailo_tables` artifact, pooling, L2; INT8 compute reported in the capability cell with the measured cosine floor |
| `ggml` | EXPERIMENTAL (GPU and CPU) on `krick`; EXPERIMENTAL (Metal GPU) on `krickert-mac` | every machine; landed on `krick` (RTX 4080 SUPER and CPU) and `krickert-mac` (Apple M2, Metal backend) | llama.cpp through the `llama-cpp-2` binding, built from source with cmake (CUDA backend on `krick`; Metal backend on `krickert-mac`; CPU backend everywhere, including CI) | `ggml_backend_dev` registry; `llama_batch` decode, one token per `step`; a generation owns its own `llama_context` (KV cache) |
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
the receipt behind its `EXPERIMENTAL` status. `ggml` (`providers/ggml`) also
only exists as a loadable library, `libturbo_provider_ggml.so`, built with
`cargo build -p turbo-provider-ggml` (also a Rust provider using
`turbo_core::export_provider!`, but not part of `turbo::builtin_providers()`
because it builds llama.cpp from source with cmake); unlike `cuda`, it *is*
a workspace member with no `--exclude` in the standard build/test/clippy
commands, because its default (CPU-only) build needs no GPU toolkit. See
"The ggml provider" below, `providers/ggml/README.md` for the build and the
live-test command, and `testdata/receipts/turbo/ggml-2026-09-21.json` for
the receipt behind its `EXPERIMENTAL` status. All five report a status of
`EXPERIMENTAL` or better only for cells with a receipt; none has a
matched-native benchmark receipt yet, so none is `SUPPORTED`
(`PLAN.md` section 2, item 7).

`static`, `cpu`, `openvino`'s CPU device, and `ggml`'s CPU device are
explicit-selection-only: `AUTO` never returns a CPU-kind device (`PLAN.md`
section 2, item 4). On OpenVINO, GPU device ordinals come first (`GPU.0`,
`GPU.1`, ...), then CPU, then NPU (`providers/openvino/src/provider.cpp`).

`docs/c-api.md`'s "Which providers set which bits" table is the full bit
matrix for all five; in outline: `mock` (`MOCK_CAPS`,
`crates/turbo-core/src/mock.rs`) sets the most bits, including every
`GENERATE` option bit it offers and `OPT_NORMALIZE`/`OPT_RAW_SCORES`, but
not `DEVICE_RESULT`, `OPT_POOLING_OVERRIDE`, or `OPT_OUTPUT_DTYPE`. `static`
(`STATIC_CAPS`, `providers/static/src/lib.rs`) sets only the `EMBED`-relevant
bits it needs for its one capability cell (no
`OPT_TOP_N`/`OPT_AGGREGATION`/`OPT_RAW_SCORES`, since it offers neither
`RERANK` nor `CLASSIFY`). `openvino` (`kCapsCommon`/`caps_of`,
`providers/openvino/src/provider.cpp`) sets `OPT_TRUNCATE`, `OPT_MAX_TOKENS`,
`OPT_PROMPT_ROLE`, `OPT_TOP_N`, `OPT_AGGREGATION`, `DETERMINISTIC`, and
`HOST_PTR_IMPORT` on every non-NPU device (host-pointer import is
implemented the same way for the CPU device as for GPU, so both advertise
the bit), plus `DEVICE_RESULT` only on GPU/iGPU; it sets neither
`OPT_NORMALIZE` nor `OPT_POOLING_OVERRIDE` (pooling and normalization always
follow the bundle contract) nor `DEVICE_POSTPROCESS`. `cuda` (`CUDA_CAPS`,
`providers/cuda/src/lib.rs`) is the only provider setting
`OPT_POOLING_OVERRIDE` and `DEVICE_POSTPROCESS`; see "The CUDA provider"
below for its full list. `ggml` (`GGML_CAPS`, `providers/ggml/src/lib.rs`)
sets `DYNAMIC_SHAPE`, `WEIGHT_SHARING`, and every `OPT_GEN_*` bit except
`OPT_GEN_N` and `OPT_GEN_TOOLS` (`n_sequences > 1` and `tools` are refused
naming the field, gated by the clear bit like every other capability
check); see "The ggml provider" below. The NPU device OpenVINO enumerates
reports `caps = 0` (listed, not qualified).

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

## The OpenVINO provider

`providers/openvino` runs `EMBED`, `RERANK`, `CLASSIFY`, and
`TOKEN_CLASSIFY` on `TEXT` bundles with an `openvino_ir` or `onnx` artifact
and a WordPiece tokenizer, on its GPU and CPU devices (the NPU device is
enumerated but offers no capability cells). Each session compiles one
`ov::Core` graph at its fixed `[max_batch, max_seq]` with post-processing
fused in: pooling and L2 for embedders, sigmoid for rerankers, softmax for
classifiers, per-token softmax for token classifiers.

The review pass tracked in `docs/reviews/2026-09-21-p0-p2.md` closed several
gaps in how the provider reports and honors its options; the current
behavior, also in `providers/openvino/README.md`:

- **Host-pointer import on every device.** `TURBO_CAP_HOST_PTR_IMPORT` is
  now set on both GPU and CPU devices (`caps_of` in `provider.cpp`): caller
  memory is wrapped without a copy on either kind, not only on GPU as
  before.
- **Pair truncation maps one to one.** `turbo_session_write_pairs`'
  `truncate` field: `MODEL` is the tokenizer's longest-first rule, `RIGHT`
  keeps the query whole and truncates the document, `NONE` fails instead of
  silently dropping tokens, and `LEFT` is refused with
  `TURBO_E_UNSUPPORTED_OPTION` naming the field (it would drop the `[CLS]`
  token and the query).
- **Unknown enum values are `TURBO_E_INVALID_ENUM`.** A `truncate`,
  `prompt_role`, or `aggregation` value that is not one of the
  `TURBO_TRUNCATE_*`/`TURBO_PROMPT_*`/`TURBO_AGGREGATE_*` constants this
  build knows fails that way, naming the field, rather than falling back to
  a default.
- **Span aggregation.** Token-classification spans are word-aligned; a word
  that truncation cut in half at either end of the row is dropped rather
  than clipped (aggregation now stops at the row's live token count instead
  of reading past it), and a group's score is the mean of its word scores,
  matching the CUDA provider's `aggregate_spans` (previously the minimum,
  and indistinguishable from `FIRST` since spans here are already
  word-aligned).
- **`raw_scores` accepted when no activation is fused, at the vtable only.**
  `normalize`, `pooling`, `output_dim`, and `output_dtype` are always
  rejected with `TURBO_E_UNSUPPORTED_OPTION` because the graph fixes them at
  compile time; `provider.cpp`'s rerank/classify/token-classify write paths
  apply the same rule to `raw_scores` except when the bundle's
  `contract.activation` is `none`, in which case the compiled graph already
  produces logits and the provider accepts it
  (`require(o.raw_scores == 0 || S.model->activation == "none", ...)`).
  No `openvino` device ever sets `TURBO_CAP_OPT_RAW_SCORES` (`caps_of` never
  includes it; `docs/c-api.md`'s bit table correctly shows it clear), and
  `Model::validate_rerank`/`validate_classify`
  (`crates/turbo-core/src/handles.rs`) reject `raw_scores = true` centrally,
  for every provider, whenever that bit is clear — before the provider is
  even called. In this tree that makes the provider's own acceptance branch
  unreachable through `turbo_session_write_pairs`/
  `turbo_session_write_text_classify` (the C API) or the equivalent safe
  Rust API call, for any bundle: only code that drives
  `turbo_provider_vtbl` directly, bypassing `turbo-core`'s option gate —
  `providers/openvino/tests/provider_test.cpp` — can observe it.
  `crates/turbo-conformance/tests/live_tasks.rs`'s
  `live_rerank_raw_scores_preserve_the_activated_ranking` and the classify
  equivalent, run through the normal API, therefore accept either outcome
  (honored or `TURBO_E_UNSUPPORTED_OPTION` naming `raw_scores`) and pass by
  taking the rejected branch against this provider.
- **`top_n`** is clamped to the row count before the sorted output is
  published, so no index the sort never touched is exposed.
- **CPU vendor reporting.** The CPU device reports the vendor actually read
  from the host CPU (`GenuineIntel`/`AuthenticAMD`, else `unknown`) instead
  of always reporting Intel.
- **Results and allocated buffers keep the caller's declared `struct_size`**
  (`x_session_run`/`x_buffer_alloc` copy through `write_sized`), and release
  entry points delete through a `noexcept` helper so a throwing `clRelease`
  during teardown cannot unwind out of a `void` vtable slot.

`providers/openvino/tests/provider_test.cpp` is a vtable-level test binary
built alongside the library (`providers/openvino/CMakeLists.txt`) that
drives `turbo_provider_vtbl` directly instead of going through
`turbo-core`, because several of the fixes above (the `struct_size`
clamping, the word-cut-in-half truncation cases, the unknown-enum checks)
are things `turbo-core` already filters out before a provider ever sees
them through the public API; see `docs/testing.md`.

## The CUDA provider

`providers/cuda` (`turbo-provider-cuda`) runs `EMBED`, `RERANK`, `CLASSIFY`,
and `TOKEN_CLASSIFY` on `TEXT` bundles with an `onnx` artifact and a BERT
WordPiece tokenizer, through the ONNX Runtime CUDA execution provider
(`providers/cuda/src/lib.rs`). One device per CUDA ordinal, kind GPU; the
CPU execution provider is never registered, so a missing CUDA library fails
context creation (`TURBO_E_DEVICE_UNAVAILABLE`) rather than falling back to
the host.

Device enumeration (`providers/cuda/src/cuda.rs`) reads compute capability
through `cudaDeviceGetAttribute` (`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_
{MAJOR,MINOR}`) and the device name through the driver library's
`cuDeviceGetName` (`libcuda.so.1`, loaded with `dlopen`/`dlsym`), not the
ONNX Runtime's `cudaGetDeviceProperties`. That function is re-versioned with
every `cudaDeviceProp` layout change, and CUDA 13 no longer exports the
`_v2` symbol the CUDA 12 headers name it by, so the previous approach failed
to load the provider at all on the Jetson; the attribute and driver APIs
have been stable across CUDA toolkit majors. `device_info.runtime_version`
reports `onnxruntime <version> / cudart <version>`, where the ONNX Runtime
half comes from `ort::info()` — the version of the ONNX Runtime library
actually linked or loaded, which differs from the crate's own version and,
on a build linking a local runtime through `ORT_LIB_LOCATION` (Jetson), is
not the same string as the prebuilt-download builds report.

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
(`testdata/receipts/turbo/cuda-2026-09-21.json`). On Jetson `nano1`
(aarch64, JetPack R39 rev 2.0, CUDA 13.2, ONNX Runtime 1.24.0 linked
dynamically through `ORT_LIB_LOCATION` and `--no-default-features`) the
device enumeration fix above unblocked loading the provider, and all 12
live embedding tests pass at cosine 1.000; there is no committed receipt
for this machine yet, and the task suite (`live_tasks.rs`: rerank, classify,
token-classify) is still being verified there (`PLAN.md` section 10, P3).
No cell is `SUPPORTED` yet: the matched-native benchmark receipt is
missing, same as `openvino` and `static`.

## The ggml provider

`providers/ggml` (`turbo-provider-ggml`) runs `GENERATE` on `TEXT` bundles
with a `gguf` artifact, through llama.cpp via the `llama-cpp-2` binding
(`providers/ggml/src/lib.rs`), which builds llama.cpp from source with
cmake at compile time. One Turbo device per `ggml_backend_dev` the compiled
backend exposes (a GPU when built with the `cuda`, `metal`, or `vulkan`
feature) plus the CPU device, which `AUTO` never selects. A model is a
`llama_model` loaded with every layer on the selected device; a generation
owns its own `llama_context` (the KV cache) and produces one token per
`step`, so cancellation and backpressure are the caller's.

Sampling is llama.cpp's sampler chain, built from the request: logit bias,
repetition penalties, top-k, top-p, min-p, temperature, then a seeded
distribution sample or greedy. Options honored (each behind its own
capability bit, `GGML_CAPS`): `temperature`, `top_k`, `top_p`, `min_p`,
`repeat_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
`logprobs`, `stop` strings, `stop_tokens`, `min_new_tokens` (suppresses the
end-of-sequence token), `echo`, `seed`, and `structured_kind = GRAMMAR`
with a GBNF grammar constraining output. Options refused with
`TURBO_E_UNSUPPORTED_OPTION` naming the field: `structured_kind =
JSON_SCHEMA` (GBNF grammars only), `n_sequences > 1`, and `tools`.
`max_new_tokens = 0` means 512. The chat template applied to
`turbo_generation_prompt`'s messages is the bundle's
`tokenizer.chat_template` when present, else the GGUF metadata's; prompt
roles and special tokens are the model's own vocabulary, not a
provider-side table.

Bundle contract: a `gguf` artifact; `contract.max_seq` is the context
length (must not exceed the model's training context). Embeddings through
GGUF, tokenize/detokenize for GGUF vocabularies, and JSON-schema
constrained output are not offered yet (`PLAN.md` section 7's secondary
GGUF-embedding path).

Build requirements: a C/C++ toolchain and cmake for llama.cpp's own build
(the default CPU-only build needs nothing else, which is why `ggml` is a
plain workspace member with no `--exclude`, unlike `cuda`).
`.cargo/config.toml` sets `CMAKE_POSITION_INDEPENDENT_CODE=ON` for every
build in the workspace, because llama.cpp's objects end up linked into a
shared library (`libturbo_provider_ggml.so`) and non-PIC objects cannot be.
The `cuda` feature additionally needs a CUDA toolkit whose `nvcc` accepts
the host compiler — on `krick`, `CUDAHOSTCXX=/usr/bin/g++-13` points nvcc
12.4 at GCC 13 (the machine's default GCC 15 is not accepted) — and a CUDA
library directory the `llama-cpp-sys-2` build can find via
`CUDA_LIBRARY_PATH` (on `krick`, a directory whose `lib64` links into
`/usr/lib/x86_64-linux-gnu`, where the runtime actually lives). CI builds
the CPU backend only.

Status: `EXPERIMENTAL` for every cell. `crates/turbo-conformance/
tests/live_generate.rs`'s seven checks pass through the safe API with
Qwen2.5-0.5B-Instruct (Q8_0) on three machines: on `krick`, both the RTX
4080 SUPER (CUDA device, `cuda` feature) and the CPU, where greedy decoding
is bit-reproducible across runs; and on `krickert-mac` (Apple M2), the
Metal device (`metal` feature) — llama.cpp's own Metal backend, not the
separate MLX-based `metal` provider `PLAN.md` section 10 (P4) scopes
(`testdata/receipts/turbo/ggml-2026-09-21.json`). Not yet: MLX generation
through the dedicated `metal` provider and Hailo-10H generation (`PLAN.md`
section 10, P6), GGUF embeddings, tokenize/detokenize for GGUF
vocabularies, JSON-schema constrained output, and the throughput receipts a
`SUPPORTED` status needs.

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
