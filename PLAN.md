# Turbo: one native inference and embedding library

Plan date: 2026-09-21. This document supersedes `ROADMAP.md` (2026-09-13) as the
governing plan. It is written to be executed in one pass, milestone by
milestone, by an agent or a contributor who has not seen the prior code.
Research inputs (runtime versions, API designs, binding technology) were
verified against primary sources on the plan date and are cited inline.

## 1. Verdict on the current code and what we keep

The repo is a proof of concept whose layers overclaimed. Two independent
reviews on 2026-09-21 agreed on the shape of the problem:

- The "common interface" exists at the symbol level (`include/turboembed.h`)
  but not at the semantic level. Per-call options mean four different things
  across ORT, OpenVINO, MLX, and Hailo; default pooling for one alias comes
  from three unrelated sources; lifetime rules differ by language.
- The "lowest, fastest layer" holds on NVIDIA only. Intel and Apple pull the
  full `[batch, seq, hidden]` activation to the CPU to pool it.
- The fast path that does exist on Intel lives in a second, vendor-specific
  ABI (`include/turboembed_prepared.h`), and that is the only ABI Java can
  reach.
- Generation is not behind any native contract; it exists only inside the
  Rust gRPC server.

We refactor by extraction, not by restart. The contract is new. The
internals that were measured to work are moved behind it from the
`poc-2026-09-21` tag.

| Keep (move behind the new contract) | Why |
|---|---|
| `native/wordpiece` | Gated loader (exact WordPiece + BertNormalizer match), HF-parity normalizer, write-through encode into caller rows |
| `native/turbo_buffer` arenas (CPU, CUDA pinned/device, Level Zero USM, Metal shared) | Working placements with allocation counters |
| `native/turboembed/src/pool_cuda.cu` and the ORT CUDA IoBinding path in `crates/turboembed/src/ort_cuda.rs` | Device-side mean+L2, `d2h_hidden_bytes == 0`, refcounted external allocator |
| `native/turboembed/sdk/prepared.cpp` | Fused mean+L2 OpenVINO graph, bundle verifier, OpenCL result lease, context/model/slot/result handle model |
| `native/turboembed/src/hailo.cpp` | Host gather + NPU encoder body + host pooling split, proven on Hailo-8 |
| `swift/Sources/TurboEmbed`, `MlxEngine`, `MetalArena` | MLX provider and Metal shared arena (pooling must move onto MLX ops) |
| `crates/turboembed/src/chunker.rs` | Byte-offset chunk planner |
| `crates/fetch`, `models/manifests/*.json` | Hash-pinned provisioning |
| `proto/`, `crates/protocol`, `crates/server` | Wire contract and server plumbing, rebuilt on the new ABI in P9 |
| `crates/backend-llamacpp` | Becomes the ggml generation provider |
| Test fixtures, goldens, receipts under `testdata/` | Reference vectors and hardware history |

| Discard | Why |
|---|---|
| `include/turboembed.h` and `include/turboembed_prepared.h` as public surfaces | Replaced by one versioned header set; a compat shim is optional and last |
| `native/turboembed/src/stub.cpp` dispatcher | `#ifdef` dispatch per feature is the source of the drift |
| `TURBOEMBED_WORKSPACE_ROOT` baked into binaries | Models come from explicit bundle paths |
| Thread-local last-error, three pooling defaults, per-provider option branches | Replaced by caller-owned errors and a single bundle contract |
| `crates/backend-apple`, `crates/arch-apple`, `native/mlx-engine`, `crates/backend-trtllm` | Stubs and legacy |
| Cargo feature flags as the way to pick a provider | Providers become runtime-loaded plugins; one core binary |

## 2. Principles (non-negotiable)

1. One contract. Every provider implements the same header set. Vendor
   differences are expressed through capability reporting, never through a
   second API.
2. Honest capabilities. Every option and every placement has a capability
   bit. A provider honors it exactly or reports the bit clear and the call
   fails with `TURBO_E_UNSUPPORTED` naming the field. Nothing is silently
   ignored, capped, or substituted.
3. Lowest layer per device. Tokenization writes into the buffers the model
   reads. Pooling and normalization run on the device that produced the
   hidden state. Results stay resident until the caller asks for a host copy.
   Where a device cannot do this (Hailo-8 has no pooling on the NPU), the
   provider reports `fully_accelerated = 0` and says which stages run on the
   host.
4. Device policy. `AUTO` means the host's best accelerator. CPU runs only when
   selected. An absent device is an error, never a fallback. The mock
   provider serves only the mock model.
5. Ownership is executable. Child handles retain parents. Releasing a parent
   never invalidates a child. Bindings enforce this with types, not comments.
6. Per-model truth lives in the bundle. Pooling, normalization, sequence
   limit, prefixes, dimension, dtype, and tokenizer identity are derived once
   at import from the model's own files and frozen with hashes.
7. Measured, not claimed. Every capability cell a provider marks
   `SUPPORTED` ships a conformance receipt, a precision receipt (variance
   against the reference), and a matched-native benchmark from a named
   machine. `EXPERIMENTAL` and `PLANNED` are honest states, not failures.
8. No one-offs. A consumer need (OpenNLP token classification, KServe) is met
   by the common surface. If the surface cannot express it, the surface is
   extended for everyone.

## 3. Product definition

- Library name: `turbo`. C prefix `turbo_`, shared library `libturbo` plus
  one `libturbo_provider_<name>` per provider. TurboEmbed stays the project
  and repository name. Inferstream stays the server.
- Deliverables, in order of delivery: C ABI + Rust crate; providers for
  OpenVINO, CUDA (x86_64 and Jetson aarch64), Metal, Hailo-8/8L, Hailo-10H,
  and ggml generation; Java (JDK 25 FFM) and Swift packages; SDK archives per
  platform; Inferstream server on the new ABI; Android JNI; GraalVM
  native-image validation for OpenNLP.
- Tasks: `EMBED` (dense; sparse and multi-vector reserved), `RERANK`
  (cross-encoder), `CLASSIFY` (sequence classification with labels),
  `TOKEN_CLASSIFY` (per-token labels with span aggregation, the OpenNLP
  NER/POS shape), `GENERATE` (causal LM, streaming), `TOKENIZE` and
  `DETOKENIZE`, `RUN` (generic named-tensor execution for any ONNX/IR/GGUF
  model, the KServe `ModelInfer` shape), and `CHUNK` (host utility).
- Scope of v1 is text. The modality axis (`TEXT, AUDIO, IMAGE, VIDEO`) is
  in the ABI from P0 so audio and video providers (Whisper on Hailo-10H,
  audio or image encoders on CUDA and OpenVINO) are added as capability
  cells right after v1 without an ABI change. `RUN` accepts image or audio
  tensors a caller has already prepared; decoders and resizers come with
  those modalities.
- Consumers this plan must not break: an OpenNLP provider on GraalVM
  native-image, and a KServe-compatible server. Both are reached through the
  same header.

## 4. Architecture

### 4.1 Layers

```
 bindings:   Rust crate | Java (FFM, later JNI) | Swift package | C/C++
 ------------------------------------------------------------------------
 libturbo:   runtime + provider registry | device discovery | buffers
             bundles + tokenizers + chunker | sessions | results | errors
 ------------------------------------------------------------------------
 providers:  cpu | cuda | openvino | metal | hailo | ggml   (dlopen plugins)
 ------------------------------------------------------------------------
 runtimes:   ORT 1.30 | OpenVINO 2026.4 | MLX 0.32 | HailoRT 4.24 / 5.4 | llama.cpp 0.4
```

The core is a Rust `cdylib` that also compiles the shared C++ (`wordpiece`,
`turbo_buffer`, pooling kernels). Providers are separate shared libraries
that export one symbol, `turbo_provider_get`, returning a versioned vtable.
The core loads providers from a search path and from explicit
`turbo_runtime_load_provider(path)` calls. A binary without a provider
library for a device reports that device absent. This replaces Cargo
features as the way to choose hardware and follows the ggml
`ggml_backend_load_all` and Windows ML `RegisterExecutionProviderLibrary`
distribution shape (ggml-backend.h; ORT 1.30 plugin EP docs).

### 4.2 Object model

```
turbo_runtime      library instance; owns the provider registry; thread-safe
  turbo_device     enumerated hardware; immutable info + capabilities
    turbo_context  device + memory domain; owns allocators and queues
      turbo_buffer typed memory in a placement; import/export native handles
      turbo_model  loaded bundle on a context; immutable contract + placement
        turbo_session  execution workspace; fixed max shape; one in-flight op
          turbo_result   leased output of the last op; explicit host read
        turbo_generation streaming generation state; pull-style iterator
  turbo_tokenizer  from a bundle; encode into caller buffers, decode, chat template
```

Every handle is opaque and reference counted internally. `turbo_*_release`
on a parent does not invalidate a child. A session accepts one operation at
a time and returns `TURBO_E_BUSY` otherwise. Distinct sessions on one model
run concurrently. Weight sharing across sessions is a provider capability.

### 4.3 Error and versioning model

- One `int32_t` status space, graded so callers can branch without parsing
  text: `0` ok; `0x1xx` argument/contract; `0x2xx` unsupported on this
  device/model/option; `0x3xx` resource (`OUT_OF_MEMORY`, `BUSY`,
  `OVERLOADED`); `0x4xx` device/runtime failure; `0x5xx` bundle integrity;
  `0x6xx` internal. Modeled on ExecuTorch's graded `Error` and TEI's
  424/429/422 split.
- Caller-owned error struct, no allocation on the error path, no shared
  state: `turbo_error { uint32_t struct_size; int32_t code; uint32_t field;
  char message[496]; }`. `field` names the offending option for `0x2xx`.
- Every public struct begins with `uint32_t struct_size`. The library reads
  only fields below the size the caller declared, and accepts a size only
  when it is the end of a field the struct has ever had (a per-struct table,
  `crates/turbo-abi/src/versioned.rs`, generated from the field lists by
  `scripts/gen-versioned.py` and checked by `scripts/gen-versioned.py
  --check`); any other size, including one that ends inside a field, is
  `TURBO_E_INVALID_STRUCT_SIZE`. Appended fields must have a
  zero value that means "old behavior". Enumerations in ABI position are
  `uint32_t` with named constants, never C `enum`. An optional `const void
  *next` chain is reserved for rare vendor imports (CUDA stream, `cl_mem`,
  `MTLBuffer`, Level Zero handle, DMA-BUF fd). This is the Linux-kernel
  extensible-struct rule; `pNext` is not used on per-call structs.
- `TURBO_ABI_VERSION` is a single integer. The header set is generated by
  cbindgen 0.29 from `#[repr(C)]` Rust declarations, committed, and checked
  by a parity test; the committed header is the artifact consumers see.

### 4.4 Capability model

Capabilities are a matrix, not a bitset alone. The axes are task
(`EMBED, RERANK, CLASSIFY, TOKEN_CLASSIFY, GENERATE, TOKENIZE, RUN`),
modality (`TEXT, AUDIO, IMAGE, VIDEO`; v1 fills in `TEXT` only, the others
are declared so audio and video are new cells later, not a new ABI), and
device. Each cell is a `turbo_capability { struct_size; status; dtype;
precision; determinism; notes[] }` where `status` is `SUPPORTED |
EXPERIMENTAL | PLANNED | UNSUPPORTED` and `precision` is the measured
variance against the reference from the qualification receipt (cosine
floor, max absolute error, reference dtype). A provider that offers one
cell (a static-embedding provider offering `EMBED x TEXT x CPU`) is a
complete, valid provider. `turbo_device_capability(device, task, modality,
&cell)` reads a cell; the same matrix is what `turbo_model_info` reports
after load for the loaded model.

Two further levels, both required:

1. Static: `turbo_device_info` carries `kind` (`CPU`, `GPU`, `IGPU`, `NPU`,
   `ACCEL`), vendor, name, provider id, ordinal, memory totals, runtime and
   driver versions, and a `uint64_t caps` bitset: `ASYNC`, `HOST_PTR_IMPORT`,
   `DEVICE_RESULT`, `EXTERNAL_QUEUE`, `DMABUF`, `UNIFIED_MEMORY`,
   `DYNAMIC_SHAPE`, `WEIGHT_SHARING`, plus one `OPT_*` bit per option field
   (`OPT_TRUNCATE`, `OPT_MAX_TOKENS`, `OPT_PROMPT_ROLE`, `OPT_NORMALIZE`,
   `OPT_POOLING_OVERRIDE`, `OPT_OUTPUT_DIM`, `OPT_OUTPUT_DTYPE`, and for
   generation `OPT_GEN_STRUCTURED`, `OPT_GEN_TOOLS`, `OPT_GEN_N`,
   `OPT_GEN_LOGIT_BIAS`, `OPT_GEN_PENALTIES`), plus `DEVICE_TOKENIZE` and
   `DEVICE_POSTPROCESS`. Modeled on
   `ggml_backend_dev_props.caps` and LiteRT's accelerator bitmask.
2. Per request: `turbo_can_run(device, bundle_or_model, task, options,
   &reason)` answers "this model, these options, on this device" before any
   allocation. Modeled on `ggml_backend_dev_supports_op`.

After load, `turbo_model_info` reports what actually happened:
`placement` (which stages run on device vs host), `fully_accelerated`,
`dtype_used`, `max_batch`, `max_seq`, and the frozen contract (below).

### 4.5 Memory model

`turbo_buffer_desc { struct_size; context; placement (HOST, PINNED, DEVICE,
SHARED); dtype; ndim; shape[8]; strides[8]; bytes; next }`. The dtype set
is the KServe Open Inference Protocol set plus BF16: `BOOL, U8, U16, U32,
U64, I8, I16, I32, I64, F16, BF16, F32, F64, BYTES` (BYTES is a
length-prefixed string tensor for text-input graphs). A provider reports
which dtypes it can hold in each placement.

- `turbo_buffer_alloc` from a context; `turbo_buffer_import` wraps caller
  memory (host pointer, CUDA device pointer, `cl_mem`, USM, `MTLBuffer`,
  Level Zero handle, DMA-BUF fd, via `next`); `turbo_buffer_export` returns
  the native handle for a downstream GPU stage; `turbo_buffer_map/unmap` for
  host access where the placement allows it.
- Sessions preallocate inputs, workspace, and outputs at creation for the
  declared max shape. `turbo_session_write_tokens` copies caller token rows
  into the bound input buffers; `turbo_session_write_text` tokenizes directly
  into them. `turbo_session_run` never allocates after warmup; the
  conformance suite asserts `allocs_per_run == 0` on the provider's own
  counter and reports provider-internal allocations separately.
- Results lease the session's output buffer. `turbo_result_read` is an
  explicit blocking copy to host memory. `turbo_result_buffer` returns the
  device buffer for zero-copy consumption. Releasing the result returns the
  lease; the session then accepts the next operation.
- Synchronous in v1. `CAP_ASYNC` providers additionally expose
  `turbo_session_submit` + `turbo_fence_wait`; external queue import is
  `CAP_EXTERNAL_QUEUE`.

### 4.6 Threading model

Runtime, device, and model handles are immutable after creation and may be
used from any thread. Context is thread-safe for allocation. Session is
single-owner: the caller serializes; a concurrent call returns `BUSY`
without corrupting state. Callbacks are invoked on the calling thread, never
from provider worker threads, and must not call back into the same session.
Bindings encode this (Rust sessions are `Send + Sync` handles whose calls are single-owner at run time; Java
`Arena.ofConfined` per call with `synchronized` handle methods; Swift
`final class` with an internal lock).

### 4.7 Provider contract: task-level, fused pipelines preferred

The provider vtable is defined at task granularity, not graph granularity.
A provider implements `embed`, `rerank`, `classify`, `token_classify`,
`generate`, and `run` against the model and session handles, and is free to
execute each as one fused device pipeline. The core's own sequence
(tokenize on host into arena rows, run the graph, pool, normalize) is the
fallback path a provider uses only for stages it declines to implement,
and a provider says which stages it took by reporting per-stage placement
in `turbo_model_info` (`stage_placement[TOKENIZE|ENCODE|POOL|NORMALIZE|
POSTPROCESS]` = `DEVICE | HOST | FUSED`). This is the gRPC rule: the header
is the contract; how a provider honors it is the provider's business, and a
faster, less uniform implementation wins over a uniform, slower one.

Consequences:

- `turbo_session_write_text` may upload raw UTF-8 into a device buffer when
  the provider reports `CAP_DEVICE_TOKENIZE`; otherwise the core tokenizes
  into the bound input rows. GPU tokenization is a measured experiment in
  the CUDA provider (P3 stretch), not a v1 requirement; the win is CPU
  offload at high throughput, not latency at batch one.
- OpenVINO compiles tokenizer (CPU device), encoder, pooling, and
  normalization as one unit; the tokenizer stage is reported `HOST`
  because string ops have no GPU kernels.
- Post-processing for `CLASSIFY` and `TOKEN_CLASSIFY` (softmax, argmax,
  span aggregation with the bundle's label set) runs on the device where the
  provider can express it and on the host otherwise, and is reported.
- The core never branches on hardware. Vendor differences live only inside
  provider libraries; the `#ifdef` dispatch of the PoC does not return.

## 5. The C ABI (representative declarations)

The full header is produced in P0. These are the shapes that decide the
design; names are final unless P0 finds a conflict.

```c
/* turbo/turbo.h  (generated; TURBO_ABI_VERSION 2) */
typedef struct turbo_error { uint32_t struct_size; int32_t code; uint32_t field; char message[496]; } turbo_error;

int32_t turbo_runtime_create(const turbo_runtime_desc *, turbo_runtime **, turbo_error *);
int32_t turbo_runtime_load_provider(turbo_runtime *, const char *path, turbo_error *);
int32_t turbo_runtime_device_count(turbo_runtime *, uint32_t *);
int32_t turbo_runtime_device_info(turbo_runtime *, uint32_t index, turbo_device_info *, turbo_error *);
int32_t turbo_runtime_select_device(turbo_runtime *, const turbo_device_selector *, uint32_t *index, turbo_error *);
/* selector: { struct_size; policy AUTO|EXPLICIT; kind mask; vendor; provider_id; ordinal } */

int32_t turbo_context_create(turbo_runtime *, uint32_t device_index, const turbo_context_desc *, turbo_context **, turbo_error *);
int32_t turbo_buffer_alloc(turbo_context *, const turbo_buffer_desc *, turbo_buffer **, turbo_error *);
int32_t turbo_buffer_import(turbo_context *, const turbo_buffer_desc *, turbo_buffer **, turbo_error *);
int32_t turbo_buffer_export(turbo_buffer *, uint32_t handle_kind, turbo_native_handle *, turbo_error *);

int32_t turbo_can_run(turbo_runtime *, uint32_t device_index, const char *bundle_path, uint64_t len, uint32_t task, const void *options, turbo_error *);
int32_t turbo_model_load(turbo_context *, const char *bundle_path, uint64_t len, const turbo_model_desc *, turbo_model **, turbo_error *);
int32_t turbo_model_info(turbo_model *, turbo_model_info *, turbo_error *);
/* model_info: { struct_size; task; kind EMBEDDING|RERANKER|GENERATIVE|GENERIC; dim; n_labels; pooling; normalize;
     max_seq; max_batch; dtype_used; stage_placement[5]; fully_accelerated; prefix_query[]; prefix_document[];
     n_labels; labels (via turbo_model_label(i));
     model_id[]; revision[]; tokenizer_sha256[]; n_inputs; n_outputs; provider_id[] } */
int32_t turbo_model_io_info(turbo_model *, uint32_t index, uint32_t direction, turbo_tensor_info *, turbo_error *);

int32_t turbo_session_create(turbo_model *, const turbo_session_desc *, turbo_session **, turbo_error *);
/* session_desc: { struct_size; max_batch; max_seq; n_options; const turbo_kv *options; next }  -- options are
   provider knobs as key/value strings (tensorrt=1, cuda_graph=1, npu_tiles=2), validated, unknown keys rejected */
int32_t turbo_session_write_text(turbo_session *, const turbo_text *texts, uint32_t count, const turbo_embed_options *, turbo_error *);
int32_t turbo_session_write_tokens(turbo_session *, const turbo_token_batch *, turbo_error *);
int32_t turbo_session_bind(turbo_session *, const char *io_name, turbo_buffer *, turbo_error *);   /* RUN task */
int32_t turbo_session_run(turbo_session *, const turbo_run_options *, turbo_result **, turbo_error *);
/* run_options: { struct_size; n_params; const turbo_kv *params }  -- the KServe per-request parameters map */
/* classification: same session; result carries scores[n][n_labels] (CLASSIFY) or per-token labels + aggregated spans
   {start_byte, end_byte, label, score} (TOKEN_CLASSIFY); labels come from the bundle contract */
int32_t turbo_session_write_text_classify(turbo_session *, const turbo_text *, uint32_t n, const turbo_classify_options *, turbo_error *);
int32_t turbo_session_stats(turbo_session *, turbo_session_stats *, turbo_error *);

int32_t turbo_result_info(turbo_result *, turbo_result_info *, turbo_error *);
int32_t turbo_result_buffer(turbo_result *, uint32_t index, turbo_buffer **, turbo_error *);
int32_t turbo_result_read(turbo_result *, uint32_t index, void *dst, uint64_t capacity_bytes, turbo_error *);
void    turbo_result_release(turbo_result *);

/* embed options (per call). Each field maps to a capability bit; MODEL means "use the bundle contract". */
typedef struct turbo_embed_options {
    uint32_t struct_size;
    uint32_t truncate;     /* MODEL | NONE | RIGHT | LEFT */
    uint32_t max_tokens;   /* 0 = MODEL */
    uint32_t prompt_role;  /* NONE | QUERY | DOCUMENT */
    uint32_t normalize;    /* MODEL | NONE | L2 */
    uint32_t pooling;      /* MODEL | MEAN | CLS | LAST  (requires OPT_POOLING_OVERRIDE) */
    uint32_t output_dim;   /* 0 = MODEL; Matryoshka truncation requires OPT_OUTPUT_DIM */
    uint32_t output_dtype; /* F32 | F16 | I8 (requires OPT_OUTPUT_DTYPE) */
} turbo_embed_options;

/* rerank: query + documents through the same session; result is scores[n] in input order plus optional sorted index */
int32_t turbo_session_write_pairs(turbo_session *, const turbo_text *query, const turbo_text *docs, uint32_t n, const turbo_rerank_options *, turbo_error *);

/* generation: pull-style iterator is primary; works without upcall stubs in every binding; cancel is a call, not a return value.
   A push form, turbo_generate(model, desc, messages, n, callback, user_data, error), is declared from P0 and returns
   TURBO_E_NOT_IMPLEMENTED until P6 lands it as a loop over turbo_generation_step; the callback returns CONTINUE | STOP. */
int32_t turbo_generation_create(turbo_model *, const turbo_generate_desc *, turbo_generation **, turbo_error *);
int32_t turbo_generation_prompt(turbo_generation *, const turbo_message *messages, uint32_t n, turbo_error *); /* applies chat template */
int32_t turbo_generation_prompt_tokens(turbo_generation *, const int32_t *ids, uint32_t n, turbo_error *);
int32_t turbo_generation_step(turbo_generation *, turbo_generation_chunk *, turbo_error *);  /* token ids, text piece, logprobs, done, finish_reason */
int32_t turbo_generation_cancel(turbo_generation *);
void    turbo_generation_release(turbo_generation *);
/* generate_desc: { struct_size; max_new_tokens; min_new_tokens; n_sequences; temperature; top_k; top_p; min_p;
   repeat_penalty; presence_penalty; frequency_penalty; seed; n_stop; stop[]; n_logit_bias; logit_bias[]; logprobs; echo;
   structured: { kind NONE | JSON_SCHEMA | GRAMMAR; text }; n_tools; tools[] (JSON); n_options; options[] }
   Each of structured output, tools, n_sequences > 1, logit_bias, and penalties is capability-gated (OPT_GEN_*). */

int32_t turbo_tokenizer_create(turbo_runtime *, const char *bundle_path, uint64_t len, turbo_tokenizer **, turbo_error *);
int32_t turbo_tokenizer_encode(turbo_tokenizer *, const turbo_text *, uint32_t n, const turbo_encode_options *, turbo_token_batch *out, turbo_error *);
int32_t turbo_tokenizer_decode(turbo_tokenizer *, const int32_t *ids, uint32_t n, char *dst, uint64_t cap, uint64_t *written, turbo_error *);
int32_t turbo_tokenizer_count(turbo_tokenizer *, const turbo_text *, uint32_t *n_tokens, turbo_error *);

int32_t turbo_chunk_plan(const turbo_chunk_desc *, const turbo_text *, turbo_tokenizer *, turbo_chunk_plan **, turbo_error *);
```

Convenience wrappers (`turbo_embed(context, bundle, texts, n, opts, float *out)`)
are provided in the header as thin sequences of the calls above and are
measured against them so their overhead is known.

## 6. Bundles and the per-model contract

A bundle is a directory with `bundle.json` (version 2) written last:

- identity: `model_id`, `revision`, `source_sha256`, `license`, `task`,
  `kind`, `family` (bert, xlm-roberta, mpnet, qwen3, llama, ...).
- tokenizer: files with hashes, `tokenizer_kind` (wordpiece, bpe, unigram,
  sentencepiece, gguf-vocab), `chat_template` for generative models.
- contract: for classifiers `labels[]`, `id2label`, `activation` (softmax,
  sigmoid, none), and for token classifiers `aggregation` (none, simple,
  first, max) with the tagging scheme (BIO, BILOU); for embedders
  `pooling` (mean, cls, last, mean_sqrt_len, weighted_mean),
  `normalize` (none, l2), `max_seq`, `dim`, `truncate_dim` list for
  Matryoshka, `prompts.query`, `prompts.document`, `similarity_fn`, `dtype`.
- artifacts, one entry per format, each with hash and provenance: `onnx`,
  `openvino_ir`, `gguf`, `hef` (+ `embedding_tables.bin`, arch tag
  `hailo8|hailo8l|hailo10h`), `mlx_safetensors`, `tensorrt_plan` (SM- and
  version-locked, optional cache only).
- limits: `max_batch`, fixed-shape flag for NPUs.

`turbo-bundle import` derives the contract from the model's own files:
`modules.json`, `1_Pooling/config.json` in both the sentence-transformers v6
schema (`pooling_mode`, `embedding_dimension`) and the legacy six-boolean
schema with its documented precedence and `mean` as the no-flag default,
`config_sentence_transformers.json` prompts, `sentence_bert_config.json`
`max_seq_length` when present, GGUF metadata for generative models. The
importer refuses ambiguous inputs rather than guessing. Providers read the
contract; they never infer pooling from an alias.

The existing fetch manifests remain the hash-pinned source list. A catalog
(alias to bundle path) exists only in the server layer.

## 7. Providers and the lowest layer on each device

| provider | hardware / machine | runtime (version, license) | lowest layer used | salvaged from | known limits to report |
|---|---|---|---|---|---|
| `static` | any CPU | none (pure Rust/C++; model2vec-style static token embeddings, Apache-2 models) | table lookup + mean + L2 on host; one capability cell `EMBED x TEXT x CPU` | new | no context, no attention; documented quality gap vs transformer embedders; first real conformance target after mock |
| `cpu` | any; explicit only | ORT 1.30 CPU EP; ggml CPU for GGUF (MIT) | host arena, write-through tokens | `ort_cuda.rs` CPU path, `backend-llamacpp` | none; never AUTO |
| `cuda` | `krick` RTX 4080 (x86_64); `nano1` Orin Nano Super (aarch64) | ORT 1.30 CUDA EP (CUDA 13 build for x86; JetPack 7.2.1 with CUDA 13.2 / TensorRT 10.16 on Jetson, using the `sbsa/cu130` ORT wheel if it carries sm_87 kernels, otherwise a pinned ORT source build on the board); TensorRT EP via session option; llama.cpp CUDA arch 87/89 | IoBinding on pinned/device arena, `user_compute_stream` import, `gpu_external_alloc` pool, device mean+L2 kernel, ORT 1.30 `CreateSyncStreamForEpDevice` for external queues | `ort_cuda.rs`, `ort_allocator.rs`, `pool_cuda.cu`, `turbo_buffer/cuda.cpp` | TensorRT plan cache is SM-locked; Jetson wheels unverified for sm_87 |
| `openvino` | `krick-1` Battlemage B70; Intel NPU when the Core Ultra host arrives | OpenVINO 2026.4.0 (Apache-2) | `ov::Core` compiled model with mean+L2 fused into the graph; `ClContext` USM/`cl_mem` remote tensors on GPU; `ZeroContext` remote tensors on NPU; explicit `"CPU"` | `prepared.cpp` (graph fusion, OpenCL lease), `turbo_buffer/ze.cpp`, `wordpiece` | NPU is static-shape only (fixed `max_seq`, batch from bundle); OpenVINO GenAI RAG pipelines are not used on the hot path because they allocate their own outputs |
| `metal` | Apple M2 | Metal directly (Objective-C++; MSL kernels compiled at load; no MLX, no Xcode) | shared `MTLBuffer`s for tokens, weights, scratch and results; encoder, pooling, L2 and the reranker head as kernels; results resident in unified memory as `TURBO_PLACE_SHARED` | `providers/metal` (landed 2026-09-22; the PoC kernels from `native/turborerank`) | no CPU device (the CPU is reached through `ggml` or OpenVINO); WordPiece on the host; F32 weights only |
| `hailo` | two Pis with Hailo-8 (HailoRT 4.24.0, `hailo8` branch); one Pi with Hailo-10H 8 GB (HailoRT 5.4.0); also x86_64 hosts with a PCIe Hailo-8 card | HailoRT (MIT); DFC is proprietary and used offline only | `VDevice` + `InferModel` + `ConfiguredInferModel::Bindings` with `dma_map` on page-aligned arena rows; async `run_async` behind `CAP_ASYNC` | `hailo.cpp` split pipeline | encoder body only on NPU: host gather and host pooling reported as `fully_accelerated = 0`; batch 1, fixed seq 128; HEF locked to chip and HailoRT line; Hailo-10H embedding HEF needs a DFC 5 compile (open) |
| `ggml` | every machine | llama.cpp v0.4.1 / ggml 0.24 (MIT) with CUDA, SYCL, Metal, CPU backends | `ggml_backend_dev` registry for device identity; `llama_batch` decode; embeddings copied once from `llama_get_embeddings_seq` into the result buffer (no caller-owned output in llama.h); KV cache owned by the generation handle | `backend-llamacpp` | generation is the primary use; GGUF embeddings are a secondary path with `pooling_type` from the bundle |
| `hailo` GenAI | Hailo-10H Pi | `hailort::genai::LLM` in HailoRT 5.4 | native LLM on the NPU with its own sampler | new | model set limited to the Hailo GenAI zoo (Qwen2.5/3 1.5B, Llama 3.2 1B); ~8 to 10 tok/s |

Hailo-8 is not ARM-only: HailoRT ships x86_64 packages and the PCIe/M.2
card is supported on Ubuntu x86_64. The provider builds for both.

## 8. Bindings

- **Rust** (`crates/turbo`): the safe API. Types enforce lifetimes
  (`Result` borrows `Session`, `Session` borrows `Model`, all `Arc`-retained),
  single-owner sessions enforced at run time with `BUSY`, callback reentry rejected, panics
  caught at the boundary. The C ABI is exported from `crates/turbo-abi`.
- **C/C++**: the header set plus a CMake package, `-fvisibility=hidden`,
  a version script exporting only `turbo_*`, SONAME `libturbo.so.2`,
  `abidiff` gate in CI.
- **Java desktop** (`bindings/java/turbo-api`, `turbo-ffm`): JDK 25 LTS
  floor; jextract `25-jextract+2-4` used at build time only, output vendored.
  `Arena.ofConfined` per call, `Arena.ofShared` for model handles used from
  pools, `MemorySegment.reinterpret` with cleanup for native-owned memory,
  upcalls only to static methods, `Enable-Native-Access` in the manifest, CI
  runs under `--illegal-native-access=deny`. `turbo-api` is the boundary the
  Android JNI adapter implements later. Java package `ai.pipestream.turbo`;
  Maven group `ai.pipestream`, artifacts `turbo-api`, `turbo-ffm`, later
  `turbo-android`, natives per platform classifier plus `turbo-native-auto`.
  The old `ai.pipestream.turboembed` package is not carried forward.
- **Swift** (`bindings/swift`): SwiftPM package with a C module map over the
  header; `final class` wrappers with `deinit` release; callbacks as
  file-level `@convention(c)` functions with `Unmanaged` user data; shipped
  as an xcframework `binaryTarget` with headers nested under
  `Headers/Turbo/`. The Metal provider is a Swift dynamic library exporting
  `turbo_provider_get` via `@c` (SE-0495, Swift 6.3).
- **Android** (P10): JNI shim over the same header, NDK r30, `arm64-v8a` and
  `x86_64`, 16 KB page alignment, minSdk 24, targetSdk 36. FFM does not exist
  on Android. GPU on device is a later capability (LiteRT-Next or Vulkan).
- **GraalVM native-image** (P10): GraalVM 25.2 supports FFM downcalls and
  upcalls on linux-x64, linux-aarch64, and macos-aarch64. Foreign calls are
  registered through `reachability-metadata.json`; handles are not created in
  build-time-initialized statics. This is the OpenNLP path.
- UniFFI is not used: its per-call `RustBuffer` copies, JNA-based Kotlin, and
  lack of a streaming primitive do not fit a hot path, and it cannot host a
  Swift-implemented provider.

## 9. Repository layout after the refactor

```
include/turbo/           generated, committed headers (turbo.h, turbo_provider.h)
crates/turbo-abi/        #[repr(C)] types, cbindgen config, extern "C" exports
crates/turbo-core/       registry, device discovery, buffers, bundles, tokenizers, chunker, sessions
crates/turbo/            safe Rust API
crates/turbo-conformance/ provider-agnostic contract suite (runs against any provider)
crates/turbo-bench/      matched-native benchmark harness and receipt writer
providers/cpu/  providers/cuda/  providers/openvino/  providers/hailo/  providers/ggml/
providers/metal/         Objective-C++ provider over Metal directly (make + clang++)
native/                  shared C++: wordpiece, turbo_buffer, pooling kernels
bindings/java/  bindings/swift/  bindings/android/
tools/turbo-bundle/      import, verify, fetch (from crates/fetch)
server/                  Inferstream: OIP v2 over gRPC and REST, OpenAI-shaped routes (landed 2026-09-22)
docs/  testdata/  scripts/
```

The proof-of-concept tree is tagged `poc-2026-09-21` at the start of P0 and
then removed from the working tree. Salvage pulls files from that tag, so
each move is reviewable against a fixed reference and the new tree starts
clean. Receipts, fixtures, manifests, and `proto/` stay in place.

## 10. Milestones and acceptance gates

Each milestone is landed when its scoped changes are merged, the listed
gates pass on the named machine, and a dated receipt is committed under
`testdata/receipts/`. Local runs, hosted CI, and device runs are recorded
separately.

### P0 Contract, mock provider, conformance suite, Rust API
Deliver `include/turbo/*.h` generated from `turbo-abi`, the `mock` provider
(deterministic 8-d vectors, only for the `mock` bundle), `turbo-core` with
runtime, device enumeration, contexts, host buffers, sessions, results,
errors, and the safe Rust crate. Deliver the conformance suite with these
groups: contract (empty text, embedded NUL, invalid UTF-8 rejected, unknown
`struct_size`, unknown option constants, oversized shapes), lifetime (release
order in every permutation, results outliving sessions and models, two
contexts), capability honesty (every `OPT_*` bit either passes its honor test
or its rejection test), device policy (AUTO never CPU, explicit CPU, absent
device fails), threading (two sessions concurrently, BUSY on overlap,
reentry rejected), allocation (`allocs_per_run == 0` after warmup where the
provider claims it).
Gate: suite passes on mock; header parity test; `cargo test --locked
--workspace` and clippy `-D warnings`; Linux CI. Machine: any.

### P1 Core services
Buffers for all placements (salvage `turbo_buffer`), bundle format v2 with
importer and verifier, the `static` embedding provider as the first non-mock
provider (one capability cell, full conformance, reference goldens from the
model2vec reference implementation), tokenizers (native WordPiece with the gated loader;
HF `tokenizers` Rust crate for BPE/Unigram; GGUF vocab via llama.cpp), chunk
planner, `turbo_can_run`, provider plugin loading with vtable versioning.
Gate: `static` provider passes the full suite and ships a precision
receipt; token-ID parity against HF `tokenizers` for MiniLM, BGE, E5, XLM-R
across ASCII, CJK, accents, emoji, long words; bundle importer round-trips
both sentence-transformers schemas; conformance still green on mock.
Machine: any.

### P2 OpenVINO provider (first GPU baseline)
Embed, rerank, classify, and token-classify on GPU with the fused graph and
remote tensors; explicit CPU; generic `RUN` task for named-tensor models
including BYTES inputs; device-resident results with OpenCL export. NPU path compiled behind the same code with static
shapes, validated when hardware arrives.
Gate: conformance green on GPU and CPU; MiniLM and BGE goldens within
tolerance; `d2h_hidden_bytes == 0`; matched-native overhead within the
budget set by the first benchmark run (recorded, then held); two-context
isolation; receipt from `krick-1`. This is the baseline all other providers
are compared against for correctness.

### P3 CUDA provider
x86_64 first (`krick`), then Jetson (`nano1`) on JetPack 7.2.1. The first
Jetson step is a runtime probe of the `sbsa/cu130` ORT wheel for sm_87
kernels; on `cudaErrorNoKernelImageForDevice` the pinned source build
(ORT 1.30, CUDA 13.2, TensorRT 10.16, sm_87) is used instead. IoBinding, stream
import, external allocator pool, device pooling kernel, TensorRT EP as a
session option, generic `RUN`, classify and token-classify with device
post-processing. Stretch, measured and reported separately: GPU WordPiece
behind `CAP_DEVICE_TOKENIZE`, compared against host tokenization at batch
1, 8, 32.
Gate: conformance green on both machines; goldens; `d2h_hidden_bytes == 0`;
two engines with interleaved create/run/destroy under the allocator pool;
receipts from both machines.

Status (2026-09-21): x86_64 has landed on `krick` (`providers/cuda/`).
Embed, rerank, classify, and token-classify run through the ONNX Runtime
CUDA execution provider with IoBinding and the provider's own device
kernels for pooling, L2 normalization, sigmoid, and softmax; results stay
on the device and are exported as `TURBO_HANDLE_CUDA_PTR`. Precision
matches the FP32 reference vectors at cosine 1.000
(`testdata/receipts/turbo/cuda-2026-09-21.json`). Every cell stays
`EXPERIMENTAL`: the matched-native benchmark and the two-engine
interleaving test above are not yet done. Jetson (`nano1`) has moved past
"not started": device enumeration originally used the runtime's
`cudaGetDeviceProperties_v2`, which CUDA 13 does not export under that
name, so the provider failed to load there; it now reads compute
capability through `cudaDeviceGetAttribute` and the device name through
the driver library's `cuDeviceGetName`, both stable across CUDA toolkit
majors (`providers/cuda/src/cuda.rs`). With that fixed, and building
`--no-default-features` against a dynamically linked ONNX Runtime 1.24.0
via `ORT_LIB_LOCATION` (`TURBO_CUDA_ARCHS=87`, JetPack R39 rev 2.0, CUDA
13.2), all 12 live embedding tests pass on `nano1` at cosine 1.000; there
is no committed receipt for that machine yet and the task suite (rerank,
classify, token-classify) is still being verified there. TensorRT EP,
`user_compute_stream` import, and the GPU WordPiece stretch goal are not
implemented on either machine yet.

### P4 Metal provider
Swift provider library exporting the plugin vtable; MLX arrays over the
shared arena without copies (pointer-verified); pooling and L2 on the GPU
stream; fixed create policy (HAILO and unknown constants rejected); result
struct laid out in C, not as a Swift struct.
Gate: conformance green on M2; goldens; no host copy of the hidden state
(measured); receipt.

Status (2026-09-22): landed as `providers/metal`, an Objective-C++
provider over Metal directly rather than the Swift-plus-MLX design above.
The change of route is deliberate: the plan's rule is the lowest, fastest
layer the platform allows, and on Apple silicon that is Metal itself, with
no MLX runtime in between and no dependency on Xcode (the provider builds
with `make` and the Command Line Tools' `clang++`, and compiles its Metal
Shading Language kernels at load time). The kernels are the ones the
2026-09-21 proof of concept validated (`native/turborerank`, tag
`poc-2026-09-21`) plus pooling, L2, the NSP pooler and the classifier
head, so every stage after WordPiece runs on the GPU and `turbo_model_info`
says so (`fully_accelerated = 0` for the tokenizer, every other stage
`DEVICE`). Weights come from an F32 `safetensors` artifact and the
architecture from the model's `config.json` (`hf_config` artifact),
cross-checked at load. Unified memory is reported honestly: the device is
`IGPU` with `DEVICE_RESULT`, `UNIFIED_MEMORY` and `HOST_PTR_IMPORT`; token
rows are written straight into shared `MTLBuffer`s, results are
`TURBO_PLACE_SHARED` (exportable as `TURBO_HANDLE_MTL_BUFFER` or a host
pointer), and `h2d_bytes` and `d2h_bytes` stay at zero, which the live
suite now checks for unified-memory devices instead of demanding an
upload. On `krickert-mac` (Apple M2, macOS 27): the 15 vtable tests
(`providers/metal/tests/provider_test.cpp`) pass; `live_embed` passes 16
of 16 at cosine 1.000 against the FP32 references with the STS ranking
gate; the rerank cases of `live_tasks` pass with
`cross-encoder/ms-marco-MiniLM-L-12-v2` (whose contract declares the
Identity activation, so scores are logits and the suite now follows the
bundle's `activation` rather than assuming a sigmoid). Receipt:
`testdata/receipts/turbo/metal-2026-09-22.json`; throughput under
`testdata/receipts/turbo/bench/metal-mac-*-2026-09-22.json`: 221 rows/s
at batch 32 by 32 tokens (5.5k tokens/s), 46 rows/s at 32 by 128, 15
rows/s at 32 by 256, and 23 documents/s reranking 32 documents at 128
tokens with the 12-layer cross-encoder; the whole batch is one set of
dispatches per layer with a 16 by 16 tiled matmul, which took the first
build from 21 to 221 rows/s. Not yet: a matmul on simdgroup matrix
operations (the M2 is two orders of magnitude below the RTX 4080 SUPER's
22k rows/s, and the elementwise and attention kernels are still one thread
per output), GPU-side WordPiece, F16 weights, classification heads with
more than one logit, and generation (which reaches the M2 through `ggml`'s
Metal backend).

### P5 Hailo provider
Hailo-8/8L on HailoRT 4.24 with `dma_map` zero-copy and async behind
`CAP_ASYNC`; Hailo-10H on HailoRT 5.4 for the same encoder split once a DFC 5
HEF exists (tracked as open); `fully_accelerated = 0` with stage placement
reported; x86_64 build for PCIe cards.
Gate: conformance green on both Hailo-8 Pis (capability tests assert the
honest limits); goldens within the INT8 tolerance recorded in the bundle;
receipt per board.

Status (2026-09-21): the `hailo` provider (`providers/hailo`, C++ over the
HailoRT 4.23 C API) serves `EMBED x TEXT` on `pi5ai1` and `cm5ai1`
(Hailo-8) with the Model Zoo INT8 MiniLM HEF: the HEF's fixed-shape encoder
body runs through f32 vstreams, WordPiece, the word-embedding gather (the
`hailo_tables` artifact), pooling, L2, and `output_dim` run on the host,
and `turbo_model_info` says so (`fully_accelerated = 0`, encode on the
device, everything else host). The capability cell reports `dtype = I8`,
`reference_dtype = F32`, and the measured `cosine_floor` (0.30); the live
suite now gates on the floor the cell states instead of a fixed 0.9995 and
adds a ranking gate (Spearman over `testdata/corpus/sts-pairs.jsonl`,
0.937 on the Hailo-8 against 0.944 for FP32). The 14 vtable tests
(`providers/hailo/tests/provider_test.cpp`) and the 14 live embedding
tests pass on both boards. Receipt:
`testdata/receipts/turbo/hailo-2026-09-21.json`. Not as planned: the
provider uses vstreams with HailoRT's scheduler rather than `InferModel`
with `dma_map` (the 4.23 packages on the Pis; `dma_map` zero-copy and
`CAP_ASYNC` stay open), and HailoRT 4.23 rather than 4.24. Not yet:
Hailo-8L (no board), Hailo-10H (needs a DFC 5 HEF), the x86_64 PCIe build,
and the x86_64 PCIe build. Throughput (2026-09-21, `turbo-bench`,
`testdata/receipts/turbo/bench/hailo-pi5ai1-embed-2026-09-21.json`): 76
rows/s at every batch and sequence length (the HEF is batch 1 with a
128-token frame; 13.2 ms per row).

### P6 Generation
`ggml` provider for GGUF generation across CUDA, SYCL, Metal, and CPU using
the pull iterator, chat templates from the bundle, cancellation, logprobs;
then the push `turbo_generate` wrapper over the same iterator (declared
since P0, stubbed until here);
MLX generation on Apple through the same provider vtable; Hailo-10H GenAI
LLM. Tokenize/detokenize for generative bundles.
Gate: streaming conformance (token order, stop strings, cancel mid-stream,
seed reproducibility on CPU, pull and push producing identical token
sequences for one seed), throughput receipts on `krick`, `krick-1`, M2,
`nano1`, and the Hailo-10H Pi.

Status (2026-09-21): the `ggml` provider (`providers/ggml`, llama.cpp through
`llama-cpp-2`) generates from GGUF bundles on the CUDA and CPU devices of
`krick` and, through llama.cpp's own Metal backend (the provider's `metal`
Cargo feature, not the separate MLX-based `metal` provider this section
scopes), on `krickert-mac` (Apple M2); all with the pull iterator, chat
templates from the bundle or the GGUF, stop strings and tokens,
cancellation, logprobs, seeded sampling, and GBNF grammars.
`turbo_generate` (push) is implemented over the pull iterator and its C
conformance test checks the two forms yield one token sequence. Receipt:
`testdata/receipts/turbo/ggml-2026-09-21.json`. GGUF embedding bundles
run through the same provider (pooling in the graph, L2 and `output_dim`
on the host, results placed in host memory) with the 13 live embedding
checks green on the CUDA and CPU devices of `krick` and the Metal and CPU
devices of `krickert-mac`, all at cosine 0.99999 or better against the
FP32 references; the runs are in the same receipt. Not yet: MLX generation
through the dedicated `metal` provider, Hailo-10H generation,
tokenize/detokenize for GGUF vocabularies, JSON-schema constrained output,
and the throughput receipts.

### P7 Java FFM and Swift packages
Port the conformance suite to Java and Swift (the same cases through the
bindings). Publish native, prepared-token Java, and text Java timings
separately against the C numbers.
Gate: Java suite green under `--illegal-native-access=deny` on `krick-1`
and `krick`; Swift suite green on M2; documented binding overhead.

Status (2026-09-21): the Java binding (`bindings/java`, `ai.pipestream:turbo`)
is in, with the raw layer generated by jextract from `include/turbo/turbo.h`
and the conformance cases passing through it on the mock provider on
`krick` and `krick-1` under `--illegal-native-access=deny`: twelve cases
including generation (pull iterator, cancel, seeded repeatability, refusal
by field) and the tokenizer, then fifteen once the second review added a
held-result BUSY case, a cross-thread cancel case, and a regression case
for the jextract indexed-accessor fix (H-6). The Swift
package (`bindings/swift`, module `PipestreamTurbo`) is in, with its
generation and tokenizer wrappers mirroring the Java ones and the same
fourteen cases as an executable runner, all passing on `krickert-mac`
(Apple M2, 2026-09-22). Not yet: the push form `turbo_generate` and the
chunk planner in the bindings, and the timings.

### P8 Packaging and SDK
Per-platform archives built in containers (manylinux_2_28 floor for Linux
x86_64 and aarch64; macOS arm64 notarized xcframework; Debian packages for
the Pi), version script, SONAME, `abidiff` gate, Maven classifier artifacts
plus an `-auto` aggregator through the Central Publisher Portal, a
clean-consumer install test per platform.
Gate: a fresh machine per platform installs, verifies a bundle, and runs
the conformance smoke without a source checkout.

Status (2026-09-21): `scripts/package.sh` builds the per-machine archive
(libturbo, headers, `turbo-bundle`, `turbo-bench`, and every provider the
machine can build), gates `libturbo.so` and the mock provider on `ldd`,
compiles and runs the C smoke test against the extracted archive, and
dlopens every packaged provider through `turbo-bench discover --strict`.
`scripts/package-container.sh` builds the Linux x86_64 archive on the
manylinux_2_28 floor in a container (`packaging/Dockerfile`; aarch64 through
qemu binfmt, untested). Not yet: the macOS xcframework, Debian packages for
the Pi, the SONAME and `abidiff` gate, Maven classifier artifacts and the
Central Publisher Portal, and the clean-consumer install test per platform.

### Dependency budget
Fewer crates is faster to build, smaller to ship and easier to audit, so
every crate declares only what it uses and test-only crates live in
`[dev-dependencies]`. Counted with `cargo tree -e normal` on 2026-09-22:
`libturbo` (turbo-shared) resolved 86 crates in the morning, 72 of them
through the Hugging Face `tokenizers` crate; by the evening it resolves
26. The core tokenizer is now native (`crates/turbo-core/src/wordpiece.rs`
over generated Unicode tables, `scripts/gen-unicode-nfd.py`): it reads
BERT-family `tokenizer.json` files (BertNormalizer, BertPreTokenizer,
WordPiece, TemplateProcessing or BertProcessing, the WordPiece decoder)
and a parity test holds it to the Hugging Face crate's ids, type ids,
byte offsets and decoded text over the STS corpus and adversarial strings
(`cargo test -p turbo-core --features hf-tokenizers`). The Hugging Face
crate stays available behind the `hf-tokenizers` feature for BPE and
Unigram files; without it such a file is `TURBO_E_UNSUPPORTED` naming the
feature. It is also faster: one thread on `krick`, release build, the
native path encodes 5.1M tokens/s on the STS sentences (2.5 us per text)
and 4.7M tokens/s on a 1000-token paragraph, 3.2x and 2.3x the Hugging
Face crate on the same texts (`cargo test -p turbo-core --features
hf-tokenizers --release speed -- --ignored --nocapture`). A native BPE
for the Qwen-style vocabularies is the next cut.
`turbo-inferstream` resolves tokio, axum, tonic and their runtime on top
of that; nothing else.

### P9 Inferstream on the new ABI
Rebuild the server as a consumer: OIP v2 for `ModelInfer`/metadata over the
`RUN` and `EMBED` tasks, plus OpenAI-shaped `/v1/embeddings`, `/v1/rerank`,
`/v1/chat/completions` with streaming, since that is the ecosystem's
embedding lingua franca (KServe 0.15+ exposes the same routes). `info`
mirrors TEI's fields from `turbo_model_info`. Sessions are fixed-shape, so
the server keeps a session pool per model keyed by (batch, seq) bucket and
pads within a bucket; a request larger than the largest bucket is rejected
with the limit, never silently truncated. Classification and token
classification are served through OIP `ModelInfer` and a `/v1/classify`
extension route.
Gate: existing e2e parity suites pass against the new server on all three
GPU machines; catalog aliases resolve to bundles.

Status (2026-09-22): landed as `server/` (`turbo-inferstream`, binary
`inferstream`, Rust): one engine, two listeners. gRPC serves
`inference.GRPCInferenceService` generated from the protocol's own
`open_inference_grpc.proto` (tonic 0.14, the current line); HTTP serves
the OIP v2 REST binding under `/v2`, the OpenAI-shaped `/v1/embeddings`,
`/v1/rerank`, `/v1/chat/completions` (SSE streaming with cancellation on
disconnect) and `/v1/classify`, and `/info` with the
text-embeddings-inference fields. Every model kind the ABI has is mapped
(embedding, reranker, classifier, token classifier, generative, generic
RUN with typed tensors). Sessions are pooled per `(batch, seq)` bucket
as planned: a request is served by the smallest fitting bucket, split by
the widest batch, and rejected with `TURBO_E_CAPACITY` naming the limit
when its longest text exceeds the longest bucket unless it sets
`truncate` to `right` or `left`; the default never cuts. Errors carry the
Turbo status name and field index on both bindings. Direct dependencies
are tokio, axum, tonic, tonic-prost, prost, serde, serde_json, clap and
futures-core (all already in the tree). Verified on `krick`: every route
on the six mock bundles, and MiniLM plus the ms-marco reranker through
`cuda` on the RTX 4080 SUPER with Qwen2.5-0.5B through `ggml`, over both
bindings (`server/README.md`). Not yet: the parity suites on all three
GPU machines, a catalog of aliases, gRPC reflection, `ModelStreamInfer`,
and the repository extension (index, load, unload).

### P10 Android JNI and GraalVM/OpenNLP
JNI adapter implementing `turbo-api`, AAR with `arm64-v8a` and `x86_64`, one
physical device named before work starts. GraalVM 25.2 native-image sample
that loads a bundle and runs `EMBED` and a token-classification `RUN` model,
with foreign-call metadata generated by the tracing agent. This is the
OpenNLP integration shape; the OpenNLP provider itself is written in the
OpenNLP repository against `turbo-api`.
Gate: Android conformance subset on the device; native-image sample runs on
linux-x64, linux-aarch64, macos-aarch64.

Dependencies: P0 then P1 are strictly first. P2, P3, P4, P5 are independent
after P1 and can run in parallel on their machines; P2 sets the correctness
baseline so it starts first. P6 needs P1 and at least one GPU provider. P7
needs P2 or P3. P8 needs P7. P9 needs P8. P10 needs P8.

## 11. Conformance and benchmark protocol

Conformance is one suite, parameterized by provider and device, run through
the C ABI (a C test binary) and through each binding. A provider is listed
as supported only when every group passes or is explicitly excluded by a
capability bit that the suite verified is reported.

Benchmarks are matched: for each provider, a direct-native reference program
using the runtime alone, and the same workload through `libturbo`. Workloads
are batch {1, 8, 32} by sequence {32, 128, 256} for embeddings, 32 documents
for rerank, and 128 new tokens for generation. Reported: p50/p99 latency,
tokens/s (when the bundle's tokenizer counted the tokens), H2D/D2H bytes
and host allocations per run, device memory; p99 from 100 iterations up.
Receipts carry machine ID, runtime versions, driver versions, bundle hashes,
and the commit the tool was built from. Budgets are set from the first run
per provider and then held: the budget must name the same provider, device
and bundle, and a cell the budget has that a later run did not measure is a
violation.

## 12. Risks and defaults chosen

| risk | default in this plan |
|---|---|
| Hailo-10H embedding HEF needs a DFC 5 encoder compile | Ship Hailo-10H generation first (HailoRT GenAI); track the embedding HEF as an open item with the DFC steps documented |
| No JetPack 7 ORT wheel channel; the `sbsa/cu130` aarch64 wheel may lack sm_87 kernels | `nano1` stays on JetPack 7.2.1 (owner keeps it current); probe the wheel first, fall back to a pinned ORT source build (CUDA 13.2, TensorRT 10.16, sm_87); the recipe is committed under `providers/cuda/jetson/` |
| Intel NPU hardware not yet available | NPU code path built and unit-tested with static shapes; supported status withheld until a receipt exists |
| A Metal result that is not the GPU's own memory | Every buffer is `MTLResourceStorageModeShared`; the result's `host_ptr` is the `MTLBuffer` contents and `d2h_bytes` is 0, which the live suite asserts on unified-memory devices |
| OpenVINO GenAI RAG pipelines allocate outputs | Not used; the fused-graph path is the provider |
| Swift `@c` needs Swift 6.3 | Toolchain floor Swift 6.3; `@_cdecl` only as a temporary shim |
| GraalVM has no macOS x64 FFM | Documented; macOS Intel is not a target |
| `--illegal-native-access=deny` may become default in a later JDK | CI already runs under `deny` |
| KServe OIP is frozen and unused by the embedding ecosystem | Keep OIP for predictive compatibility; OpenAI-shaped routes are the primary embedding surface |

## 13. Decisions for the owner

1. Prefix and library name `turbo_` / `libturbo` with TurboEmbed as the
   project name. Decided 2026-09-21.
2. Refactor in place: tag the PoC and remove it from the tree. Decided
   2026-09-21.
3. Jetson runs JetPack 7.2.1, which is what `nano1` has. Decided 2026-09-21.
4. Generation streaming: both forms in the header; pull iterator first,
   push wrapper stubbed until pull passes conformance. Decided 2026-09-21.
5. Java package `ai.pipestream.turbo`, Maven group `ai.pipestream`.
   Decided 2026-09-21.

All five were decided on 2026-09-21; execution proceeds from P0.

## 14. Sources consulted (2026-09-21)

ONNX Runtime 1.30.0 release notes and `onnxruntime_c_api.h`; ONNX Runtime
GenAI 0.16.0 `ort_genai_c.h`; OpenVINO 2026.4.0 release notes, NPU device
docs, `intel_gpu/ocl/ocl.hpp`, `intel_npu/level_zero/level_zero.hpp`;
OpenVINO GenAI 2026.4 `rag/text_embedding_pipeline.hpp`; HailoRT 5.4.0 and
4.24.0 (`hailo8` branch), `hailo/genai/llm/llm.hpp`; NVIDIA JetPack 7.2.1
and 6.2.3 release pages, `pypi.jetson-ai-lab.io`; MLX 0.32.2, mlx-c 0.6.0,
mlx-swift 0.31.6; llama.cpp v0.4.1 `llama.h` and `ggml-backend.h`; TEI
1.9.4 router and `tei.proto`; DJL 0.38.0 `Engine.java` and `Criteria`;
sentence-transformers 6.1.0 pooling and config schemas; KServe 0.20 and the
open-inference-protocol repository; ExecuTorch 1.5 `Error`; LiteRT 2.2 C
API; IREE runtime C API; JEP 454 and JDK 25 restricted-methods docs;
jextract `25-jextract+2-4`; GraalVM 25.2 FFM reference; Android NDK r30 and
16 KB page-size guide; SE-0495; UniFFI 0.32.1; cbindgen 0.29.4; manylinux
policy (PEP 600); libabigail `abidiff`; Maven Central Publisher Portal.
