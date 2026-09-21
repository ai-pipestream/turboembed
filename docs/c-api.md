# C API

This walks `include/turbo/turbo.h` function family by function family. The
header itself is generated (`scripts/gen-header.sh`) from
`crates/turbo-abi` and `crates/turbo-capi`; the doc comments on each function
in the header are the normative per-function contract. This page adds the
ownership/threading rule, the status-code grades, and the option/capability
mapping in one place instead of repeating them per function.

General rules that apply to every function (from the header's own preamble
and `crates/turbo-capi/src/lib.rs`):

- Every descriptor starts with `uint32_t struct_size`. Pass
  `sizeof(the_struct)`. A size the library does not recognize (see
  "Descriptor versioning rule" below) is `TURBO_E_INVALID_STRUCT_SIZE`.
- `turbo_error *err` may be `NULL` in every call that takes one; on success
  `code == 0`. Passing a real `turbo_error` and checking `err.code` after
  every call is the only supported error-handling pattern (see
  `crates/turbo-conformance/c/smoke.c`).
- Release functions accept `NULL` as a no-op.

## Version and status (no handle; any thread)

`turbo_abi_version`, `turbo_version`, `turbo_status_name`. Pure functions,
no handles, callable from any thread at any time.

## Runtime (`turbo_runtime_*`, `turbo_can_run`)

`turbo_runtime_create` / `turbo_runtime_release`; `turbo_runtime_load_provider`;
`turbo_runtime_device_count`, `turbo_runtime_device_info`,
`turbo_runtime_select_device`, `turbo_runtime_capability`; `turbo_can_run`.

Ownership: `turbo_runtime_create` returns an owned, reference-counted handle.
`turbo_context_create` retains it, so releasing the runtime while contexts
are still open is safe (see the smoke test: `turbo_runtime_release(rt)`
immediately after creating a context on it). Threading: the runtime and its
query functions (`device_count`, `device_info`, `select_device`,
`capability`, `can_run`) are usable from any thread concurrently; there is
no per-runtime lock in `crates/turbo-core/src/runtime.rs` that a caller needs
to serialize around.

`turbo_runtime_load_provider(rt, path, err)` loads a provider library
(`include/turbo/turbo_provider.h`) and registers its devices; the library
stays loaded for the runtime's lifetime, and a provider whose id is already
registered is rejected with `TURBO_E_PROVIDER_LOAD`. `turbo_runtime_create`
does the same for every path in `turbo_runtime_desc.provider_paths`, in
order, before returning; a failure there fails the whole call. By default a
runtime starts with the statically linked built-in providers (`mock` and
`static`); set the `TURBO_RUNTIME_NO_DEFAULT_PROVIDERS` bit in
`turbo_runtime_desc.flags` to start with none and load only explicit paths.
There is no default filesystem search path scanned automatically; see
`docs/architecture.md` for how this differs from `PLAN.md` section 4.1 and
`docs/providers.md` for the ownership rules a provider library must follow.

## Context (`turbo_context_*`)

`turbo_context_create` / `turbo_context_release`; `turbo_context_device`.

Ownership: a context retains the runtime it was created from
(`Context` holds `Arc<Runtime>`). Buffers and models created from a context
retain it in turn. Threading: any thread; a context has no session-like
single-owner state.

## Buffer (`turbo_buffer_*`)

`turbo_buffer_alloc`, `turbo_buffer_import`, `turbo_buffer_release`,
`turbo_buffer_get_desc`, `turbo_buffer_host_ptr`, `turbo_buffer_export`.

Ownership: a buffer retains its context. A buffer obtained from
`turbo_result_buffer` additionally retains the result's lease (see Results
below); releasing that buffer view is required before the lease clears.
Threading: any thread may call these; the underlying memory's own read/write
safety is the caller's responsibility exactly as for any C API — a buffer
bound into a running session must not be concurrently written from another
thread. `mock` and `static` only allocate `TURBO_PLACE_HOST`; the OpenVINO
provider also allocates `TURBO_PLACE_DEVICE` on GPU, exportable as
`TURBO_HANDLE_CL_MEM` (`docs/providers.md`). Any other placement or handle
kind not offered by the loaded provider fails with
`TURBO_E_UNSUPPORTED_PLACEMENT` (see `docs/architecture.md`).

## Model (`turbo_model_*`)

`turbo_model_load`, `turbo_model_release`, `turbo_model_get_info`,
`turbo_model_io_info`, `turbo_model_label`.

Ownership: a model retains its context. Sessions and generations created
from a model retain it. Threading: any thread; a loaded model is immutable
(`ModelInfo` is fixed at load time). `turbo_model_load` opens and verifies
the bundle (hash-checks every listed file; see `docs/bundles.md`) before
returning, so it is not a cheap call to make repeatedly on a hot path.

## Session (`turbo_session_*`)

`turbo_session_create`, `turbo_session_release`, `turbo_session_write_text`,
`turbo_session_write_tokens`, `turbo_session_write_pairs`,
`turbo_session_write_text_classify`, `turbo_session_bind`,
`turbo_session_run`, `turbo_session_get_stats`.

Ownership: a session retains its model. A result leased from `run` retains
the session. Threading: **single owner**. A session accepts one operation
(a write, a bind, or a run) at a time; a concurrent call from another thread
returns `TURBO_E_BUSY` immediately rather than blocking or corrupting state
(`Session::lock` in `crates/turbo-core/src/handles.rs` uses
`Mutex::try_lock`). Distinct sessions on the same model may run
concurrently. `run` also fails with `TURBO_E_BUSY` while a previous result
(or any buffer view derived from it) has not been released, and fails with
`TURBO_E_INVALID_STATE` if no write/bind call has succeeded since the last
run. Which write function applies depends on the model's kind
(`turbo_model_get_info().kind`): `write_text`/`write_tokens` for embedders,
`write_pairs` for rerankers, `write_text_classify` for classifiers and
token classifiers, `bind` for generic `RUN` models.

Two checks apply to every write call on every provider, independent of any
capability bit, so a provider can never clamp or substitute instead of
failing: `max_tokens` above the session's `max_seq` is `TURBO_E_CAPACITY`
naming the option's `max_tokens` field (field 3 on `turbo_embed_options`,
`turbo_rerank_options`, and `turbo_classify_options`, `check_budget` in
`crates/turbo-core/src/handles.rs`); and an `embed_options.prompt_role`
naming a role the bundle declares no prefix for is
`TURBO_E_INVALID_ARGUMENT` on field 4 (`prompt_role`), since honoring it
would silently embed the bare text instead of the requested prompt.

## Result (`turbo_result_*`)

`turbo_result_get_info`, `turbo_result_output_info`, `turbo_result_buffer`,
`turbo_result_read`, `turbo_result_spans`, `turbo_result_release`.

Ownership: a result retains its session; a buffer obtained from
`turbo_result_buffer` retains the result. Threading: a result is a
point-in-time snapshot of the session's last run; `turbo_result_read` is an
explicit blocking copy to host memory and is safe to call from any thread
that holds the result handle, but the result itself is not meant to be
shared for concurrent mutation (there is none to do — results are
read-only). Releasing every outstanding result and its buffer views is what
allows the owning session to accept its next write or run
(`TURBO_E_BUSY` otherwise).

## Generation (`turbo_generation_*`, `turbo_generate`)

`turbo_generation_create`, `turbo_generation_prompt`,
`turbo_generation_prompt_tokens`, `turbo_generation_step`,
`turbo_generation_cancel`, `turbo_generation_release`; `turbo_generate`.

Ownership: a generation retains its model. Threading: single owner, same
rule as sessions (`Generation::lock` also uses `try_lock`). Exactly one of
`turbo_generation_prompt` or `turbo_generation_prompt_tokens` may be called,
exactly once, before stepping; `turbo_generation_step` fails with
`TURBO_E_INVALID_STATE` before a prompt is set or after the generation has
finished. Pointers inside the `turbo_generation_chunk` written by `step` are
valid only until the next call on the same generation. `turbo_generate` (the
push-style wrapper that drives `step` internally via a callback) is declared
from P0 but returns `TURBO_E_NOT_IMPLEMENTED` in this build.

## Tokenizer and chunker (`turbo_tokenizer_*`, `turbo_chunk_plan_*`)

`turbo_tokenizer_create(rt, bundle_path, out, err)` loads the tokenizer a
bundle declares (`tokenizer.files["tokenizer.json"]`); the general path is
the Hugging Face `tokenizers` crate (`crates/turbo-core/src/tokenizer.rs`),
which reads WordPiece, BPE, and Unigram `tokenizer.json` files. There is no
C entry point that takes a bare `tokenizer.json` path directly; a tokenizer
is always loaded from a bundle directory, including a tokenizer-only bundle
with no artifacts (`testdata/bundles/minilm-tokenizer/`, `task: "tokenize"`).
A tokenizer does not depend on a runtime's device state and is thread-safe;
`rt` is only used to validate the handle.

- `turbo_tokenizer_get_info` returns `turbo_tokenizer_info` (vocab size,
  bundle `max_seq`, `specials_per_sequence`, and `pad_id`/`bos_id`/`eos_id`/
  `unk_id`, each `-1` if the tokenizer has none).
- `turbo_tokenizer_encode(t, texts, count, opts, ids, mask, types,
  row_stride, lengths, err)` writes directly into caller-owned row-major
  `[count, row_stride]` arrays: each row is truncated per `opts->truncate`
  (`TURBO_TRUNCATE_MODEL|NONE|RIGHT|LEFT`) to `opts->max_tokens` (0 = the
  bundle's `max_seq`), gets the prompt prefix for `opts->prompt_role`, and is
  padded with the pad id and mask 0 up to `opts->pad_to` (or to `row_stride`
  when `pad_to` is 0); `lengths[row]` receives the row's live (unpadded)
  token count. `types` and `lengths` may be `NULL`. Truncation `NONE` on
  over-budget input is `TURBO_E_CAPACITY`, not a silent drop; `row_stride`
  smaller than the effective token budget is also `TURBO_E_CAPACITY`.
- `turbo_tokenizer_decode(t, ids, count, skip_special_tokens, dst, capacity,
  written, err)` writes UTF-8 bytes (not NUL-terminated) into `dst`;
  `written` always receives the decoded length, and a `capacity` too small
  for it is `TURBO_E_CAPACITY` with `written` already set to the required
  size. An id outside `0..vocab_size` is `TURBO_E_INVALID_ARGUMENT`.
- `turbo_tokenizer_count(t, text, add_special_tokens, out, err)` counts
  tokens without truncation or a prompt prefix.

`turbo_chunk_plan_create(desc, text, tokenizer, out, err)` plans byte-offset
chunks over `text` (`crates/turbo-core/src/chunker.rs`), using `tokenizer` to
count content tokens against `desc->max_tokens` and
`desc->reserved_tokens`, with up to `desc->overlap_tokens` shared between
consecutive chunks of one paragraph. The plan stores only byte offsets
(`turbo_chunk { byte_start, byte_end, paragraph, n_tokens }`); the caller
keeps the source text. `turbo_chunk_plan_count`/`_get` read the resulting
chunks by index; a single character that alone exceeds the token budget
fails plan creation with `TURBO_E_CAPACITY` rather than looping or silently
truncating.

Ownership: both tokenizers and chunk plans are independent, reference
counted handles with no parent; `turbo_tokenizer_release`/
`turbo_chunk_plan_release` accept `NULL`. `PLAN.md` section 5 also sketches
convenience wrappers like `turbo_embed`; those are not present in the
generated header at this commit.

## Status codes

`int32_t`, graded by the hundreds digit so a caller can branch without
parsing `message` (`crates/turbo-abi/src/lib.rs`):

| grade | meaning | codes |
|---|---|---|
| `0x000` | ok | `TURBO_OK` (0) |
| `0x1xx` | argument/contract violation by the caller | `INVALID_ARGUMENT` (0x100), `INVALID_STRUCT_SIZE` (0x101), `INVALID_UTF8` (0x102), `INVALID_HANDLE` (0x103), `INVALID_SHAPE` (0x104), `INVALID_STATE` (0x105), `INVALID_ENUM` (0x106) |
| `0x2xx` | unsupported on this device/model/option (capability) | `UNSUPPORTED` (0x200), `UNSUPPORTED_OPTION` (0x201), `UNSUPPORTED_TASK` (0x202), `UNSUPPORTED_DTYPE` (0x203), `UNSUPPORTED_PLACEMENT` (0x204), `NOT_IMPLEMENTED` (0x205), `UNSUPPORTED_MODALITY` (0x206) |
| `0x3xx` | resource | `OUT_OF_MEMORY` (0x300), `BUSY` (0x301), `OVERLOADED` (0x302), `CAPACITY` (0x303) |
| `0x4xx` | device/runtime/provider failure | `DEVICE_NOT_FOUND` (0x400), `DEVICE_UNAVAILABLE` (0x401), `RUNTIME` (0x402), `PROVIDER_LOAD` (0x403), `ABI_MISMATCH` (0x404), `CANCELLED` (0x405) |
| `0x5xx` | bundle integrity | `BUNDLE_NOT_FOUND` (0x500), `BUNDLE_INVALID` (0x501), `BUNDLE_INTEGRITY` (0x502), `BUNDLE_NO_ARTIFACT` (0x503) |
| `0x6xx` | internal | `INTERNAL` (0x600), `PANIC` (0x601) |

`turbo_error.field` is the 1-based index of the offending field in the
descriptor for `TURBO_E_UNSUPPORTED_OPTION` and most `TURBO_E_INVALID_ARGUMENT`
cases, otherwise 0. `turbo_status_name(code)` returns the symbolic name (or
`"TURBO_E_UNKNOWN"`) as a static string.

## Option to capability-bit mapping

Each per-call option field maps to a `TURBO_CAP_OPT_*` bit in
`turbo_device_info.caps`; a provider that does not advertise the bit rejects
a non-default value for that field with `TURBO_E_UNSUPPORTED_OPTION` and the
field index below (field 1 is always `struct_size`). Field indices come from
the `FIELD_*` constants in `crates/turbo-core/src/provider.rs`.

Every option field in every options struct is now covered: it either has a
capability bit here, or is one of the two unconditional checks above
(`max_tokens`, `prompt_role`'s empty-prefix case). There is no third,
ungated case left (`docs/reviews/2026-09-21-p0-p2.md`'s Medium item on
generation options and `raw_scores`, closed in commit `b85ccf1`).

`turbo_embed_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `prompt_role` | 4 | `TURBO_CAP_OPT_PROMPT_ROLE` | `!= TURBO_PROMPT_NONE` |
| `normalize` | 5 | `TURBO_CAP_OPT_NORMALIZE` | differs from the bundle contract |
| `pooling` | 6 | `TURBO_CAP_OPT_POOLING_OVERRIDE` | differs from the bundle contract |
| `output_dim` | 7 | `TURBO_CAP_OPT_OUTPUT_DIM` | `!= 0` and `!=` the model's `dim`; must also be one of the bundle's `truncate_dims` |
| `output_dtype` | 8 | `TURBO_CAP_OPT_OUTPUT_DTYPE` | requests a dtype other than the model's own `dtype_used` (asking for the dtype already computed is always free, `Model::validate_embed` in `crates/turbo-core/src/handles.rs`) |

`turbo_rerank_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `top_n` | 4 | `TURBO_CAP_OPT_TOP_N` | `!= 0` |
| `return_sorted` | 5 | `TURBO_CAP_OPT_TOP_N` | set (its own field index; previously misreported as field 4) |
| `raw_scores` | 6 | `TURBO_CAP_OPT_RAW_SCORES` | set |

`turbo_classify_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `aggregation` | 4 | `TURBO_CAP_OPT_AGGREGATION` | differs from the bundle contract; only meaningful for token classifiers (`TURBO_E_INVALID_ARGUMENT` on any other model kind) |
| `raw_scores` | 5 | `TURBO_CAP_OPT_RAW_SCORES` | set |

`turbo_generate_desc` (every gated field):

| field | index | capability bit |
|---|---|---|
| `min_new_tokens` | 3 | `TURBO_CAP_OPT_GEN_MIN_TOKENS` (`!= 0`) |
| `n_sequences` | 4 | `TURBO_CAP_OPT_GEN_N` (`> 1`) |
| `temperature` | 5 | `TURBO_CAP_OPT_GEN_SAMPLING` (`!= 0`) |
| `top_k` | 6 | `TURBO_CAP_OPT_GEN_SAMPLING` (`!= 0`) |
| `top_p` | 7 | `TURBO_CAP_OPT_GEN_SAMPLING` (`!= 0` and `!= 1`) |
| `min_p` | 8 | `TURBO_CAP_OPT_GEN_SAMPLING` (`!= 0`) |
| `repeat_penalty` | 9 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0` and `!= 1`) |
| `presence_penalty` | 10 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0`) |
| `frequency_penalty` | 11 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0`) |
| `has_seed` | 12 | `TURBO_CAP_OPT_GEN_SEED` |
| `n_stop` | 14 | `TURBO_CAP_OPT_GEN_STOP_STRINGS` (`> 0`) |
| `n_stop_tokens` | 15 | `TURBO_CAP_OPT_GEN_STOP_TOKENS` (`> 0`) |
| `n_logit_bias` | 18 | `TURBO_CAP_OPT_GEN_LOGIT_BIAS` (`> 0`) |
| `logprobs` | 19 | `TURBO_CAP_OPT_GEN_LOGPROBS` (`> 0`) |
| `structured_kind` | 21 | `TURBO_CAP_OPT_GEN_STRUCTURED` (`!= TURBO_STRUCTURED_NONE`) |
| `echo` | 22 | `TURBO_CAP_OPT_GEN_ECHO` (set) |
| `n_tools` | 24 | `TURBO_CAP_OPT_GEN_TOOLS` (`> 0`) |

The remaining `TURBO_CAP_*` bits (`ASYNC`, `HOST_PTR_IMPORT`,
`DEVICE_RESULT`, `EXTERNAL_QUEUE`, `DMABUF`, `UNIFIED_MEMORY`,
`DYNAMIC_SHAPE`, `WEIGHT_SHARING`, `DEVICE_TOKENIZE`, `DEVICE_POSTPROCESS`,
`DETERMINISTIC`) describe device- or provider-level behavior rather than
gating a specific option field; see `docs/architecture.md`'s memory and
threading sections for what each currently means in this tree.

## Which providers set which bits

The four providers in this tree (`crates/turbo-core/src/mock.rs`
`MOCK_CAPS`, `providers/static/src/lib.rs` `STATIC_CAPS`,
`providers/openvino/src/provider.cpp` `caps_of`/`kCapsCommon`,
`providers/cuda/src/lib.rs` `CUDA_CAPS`):

| bit | mock | static | openvino (GPU/iGPU) | openvino (CPU) | cuda |
|---|---|---|---|---|---|
| `HOST_PTR_IMPORT` | yes | yes | yes | yes | yes |
| `DEVICE_RESULT` | no | no | yes | no | yes |
| `DEVICE_POSTPROCESS` | no | no | no | no | yes |
| `DYNAMIC_SHAPE` | yes | yes | no | no | yes |
| `WEIGHT_SHARING` | yes | yes | no | no | yes |
| `DETERMINISTIC` | yes | yes | yes | yes | no |
| `OPT_TRUNCATE` | yes | yes | yes | yes | yes |
| `OPT_MAX_TOKENS` | yes | yes | yes | yes | yes |
| `OPT_PROMPT_ROLE` | yes | yes | yes | yes | yes |
| `OPT_NORMALIZE` | yes | no | no | no | yes |
| `OPT_POOLING_OVERRIDE` | no | no | no | no | yes |
| `OPT_OUTPUT_DIM` | yes | yes | no | no | yes |
| `OPT_OUTPUT_DTYPE` | no | no | no | no | no |
| `OPT_TOP_N` | yes | no | yes | yes | yes |
| `OPT_AGGREGATION` | yes | no | yes | yes | yes |
| `OPT_RAW_SCORES` | yes | no | no | no | yes |
| `OPT_GEN_STOP_STRINGS` | yes | no | no | no | no |
| `OPT_GEN_STOP_TOKENS` | yes | no | no | no | no |
| `OPT_GEN_SEED` | yes | no | no | no | no |
| `OPT_GEN_LOGPROBS` | yes | no | no | no | no |
| `OPT_GEN_SAMPLING` | yes | no | no | no | no |
| `OPT_GEN_MIN_TOKENS` | yes | no | no | no | no |
| `OPT_GEN_ECHO` | yes | no | no | no | no |

`static` and `openvino`'s CPU device offer no task where `EMBED`-only bits
like `OPT_NORMALIZE`/`OPT_POOLING_OVERRIDE`/`OPT_RAW_SCORES` would matter
differently than shown; a `no` above means the bit is clear in
`device_info.caps`, so a non-default value for the corresponding option
field on that provider always fails with `TURBO_E_UNSUPPORTED_OPTION`, never
a silent default. `mock` is the only provider offering `GENERATE`; `static`,
`openvino`, and `cuda` fail a generation call with `TURBO_E_UNSUPPORTED_TASK`
before any option is checked. `openvino`'s NPU device (enumerated, not
qualified) reports `caps = 0`.

## Descriptor versioning rule

Every descriptor starts with `uint32_t struct_size`, set by the caller to
`sizeof(the_struct_as_compiled)`. The library accepts a size only when it is
the end of a field the struct has ever had: the current size, or the offset
of a field from an earlier version of the struct (an older caller compiled
against a smaller version). A size that ends inside a field (half a pointer,
half a count) or that the struct has never had, including any size larger
than the current struct, is `TURBO_E_INVALID_STRUCT_SIZE`.

The accepted sizes are not a range check; they are a per-struct table,
`crates/turbo-abi/src/versioned.rs` (the `Versioned` trait's `SIZES`,
generated by `scripts/gen-versioned.py` from the field lists in
`crates/turbo-abi/src/lib.rs` and `provider.rs`). `check_size` in
`crates/turbo-capi/src/lib.rs` looks a caller's declared size up in that
table. Regenerate `versioned.rs` whenever a struct's fields change;
`scripts/gen-versioned.py --check` fails CI if the committed file is stale.
Fields appended in a future version must default to a zero value that means
"old behavior" so that an old caller's smaller struct, zero-extended,
behaves the same as before.

## Functions declared but not implemented in this build

| function | returns | milestone |
|---|---|---|
| `turbo_generate` (push-style generation) | `TURBO_E_NOT_IMPLEMENTED` | P6 (wraps the pull iterator once it passes streaming conformance) |

This is the only remaining gap between the header and this build.
`turbo_runtime_load_provider`, `turbo_tokenizer_*`, and `turbo_chunk_plan_*`
are implemented (see above); `turbo_generation_create`/`_prompt`/
`_prompt_tokens`/`_step`/`_cancel`/`_release` (the pull iterator) are also
implemented and exercised by the conformance suite's generation tests
against the `mock` provider (`MockGeneration` in `crates/turbo-core/src/mock.rs`).
`static` and `openvino` do not offer `GENERATE`; a generation call on either
fails with `TURBO_E_UNSUPPORTED_TASK`.
