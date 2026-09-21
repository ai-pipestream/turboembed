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
  `sizeof(the_struct)`. A size the library does not recognize is
  `TURBO_E_INVALID_STRUCT_SIZE`.
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
to serialize around. `turbo_runtime_load_provider` is declared but returns
`TURBO_E_NOT_IMPLEMENTED` (planned P1; see "Not implemented" below).

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
thread. `turbo_buffer_alloc` currently only succeeds for
`TURBO_PLACE_HOST` (the only provider is `mock`); other placements fail with
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

`turbo_tokenizer_create`, `turbo_tokenizer_release`,
`turbo_chunk_plan_create`, `turbo_chunk_plan_release` are declared in the
header. Both `_create` functions return `TURBO_E_NOT_IMPLEMENTED` in this
build. Note that `PLAN.md` section 5 also sketches
`turbo_tokenizer_encode`/`decode`/`count` and convenience wrappers like
`turbo_embed`; those are not present in the generated header at this
commit — the header currently declares only bundle-backed tokenizer/chunk
creation and release. Treat `PLAN.md` section 5 as the target shape for P1,
not as a description of the current header.

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

`turbo_embed_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `prompt_role` | 4 | `TURBO_CAP_OPT_PROMPT_ROLE` | `!= TURBO_PROMPT_NONE` |
| `normalize` | 5 | `TURBO_CAP_OPT_NORMALIZE` | differs from the bundle contract |
| `pooling` | 6 | `TURBO_CAP_OPT_POOLING_OVERRIDE` | differs from the bundle contract |
| `output_dim` | 7 | `TURBO_CAP_OPT_OUTPUT_DIM` | `!= 0` and `!=` the model's `dim`; must also be one of the bundle's `truncate_dims` |
| `output_dtype` | 8 | `TURBO_CAP_OPT_OUTPUT_DTYPE` | `!= TURBO_OUTPUT_MODEL` and `!= TURBO_OUTPUT_F32` |

`turbo_rerank_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `top_n` | 4 | `TURBO_CAP_OPT_TOP_N` | `!= 0` or `return_sorted` set |
| `return_sorted` | 5 | `TURBO_CAP_OPT_TOP_N` | set |
| `raw_scores` | 6 | (ungated) | always allowed |

`turbo_classify_options`:

| field | index | capability bit | gated when |
|---|---|---|---|
| `truncate` | 2 | `TURBO_CAP_OPT_TRUNCATE` | `!= TURBO_TRUNCATE_MODEL` |
| `max_tokens` | 3 | `TURBO_CAP_OPT_MAX_TOKENS` | `!= 0` |
| `aggregation` | 4 | `TURBO_CAP_OPT_AGGREGATION` | differs from the bundle contract; only meaningful for token classifiers (`TURBO_E_INVALID_ARGUMENT` on any other model kind) |
| `raw_scores` | 5 | (ungated) | always allowed |

`turbo_generate_desc` (selected gated fields):

| field | index | capability bit |
|---|---|---|
| `n_sequences` | 4 | `TURBO_CAP_OPT_GEN_N` (`> 1`) |
| `repeat_penalty` | 9 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0` and `!= 1`) |
| `presence_penalty` | 10 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0`) |
| `frequency_penalty` | 11 | `TURBO_CAP_OPT_GEN_PENALTIES` (`!= 0`) |
| `has_seed` | 12 | `TURBO_CAP_OPT_GEN_SEED` |
| `n_stop` | 14 | `TURBO_CAP_OPT_GEN_STOP_STRINGS` (`> 0`) |
| `n_logit_bias` | 18 | `TURBO_CAP_OPT_GEN_LOGIT_BIAS` (`> 0`) |
| `logprobs` | 19 | `TURBO_CAP_OPT_GEN_LOGPROBS` (`> 0`) |
| `structured_kind` | 21 | `TURBO_CAP_OPT_GEN_STRUCTURED` (`!= TURBO_STRUCTURED_NONE`) |
| `n_tools` | 24 | `TURBO_CAP_OPT_GEN_TOOLS` (`> 0`) |

The remaining `TURBO_CAP_*` bits (`ASYNC`, `HOST_PTR_IMPORT`,
`DEVICE_RESULT`, `EXTERNAL_QUEUE`, `DMABUF`, `UNIFIED_MEMORY`,
`DYNAMIC_SHAPE`, `WEIGHT_SHARING`, `DEVICE_TOKENIZE`, `DEVICE_POSTPROCESS`,
`DETERMINISTIC`) describe device- or provider-level behavior rather than
gating a specific option field; see `docs/architecture.md`'s memory and
threading sections for what each currently means in this tree.

## Descriptor versioning rule

Every descriptor starts with `uint32_t struct_size`, set by the caller to
`sizeof(the_struct_as_compiled)`. The library accepts that exact size or any
smaller size down to the minimum needed to identify the struct (an older
caller compiled against a smaller version of the struct); it rejects a
larger size, because it cannot know what an unrecognized trailing field was
meant to do (`check_size` in `crates/turbo-capi/src/lib.rs`). Fields appended
in a future version must default to a zero value that means "old behavior"
so that an old caller's smaller struct, zero-extended, behaves the same as
before.

## Functions declared but not implemented in this build

| function | returns | milestone |
|---|---|---|
| `turbo_runtime_load_provider` | `TURBO_E_NOT_IMPLEMENTED` | P1 (provider plugin loading) |
| `turbo_tokenizer_create` | `TURBO_E_NOT_IMPLEMENTED` | P1 |
| `turbo_chunk_plan_create` | `TURBO_E_NOT_IMPLEMENTED` | P1 |
| `turbo_generate` (push-style generation) | `TURBO_E_NOT_IMPLEMENTED` | P6 (wraps the pull iterator once it passes streaming conformance) |

Also note: `RuntimeDesc.provider_paths` passed to `turbo_runtime_create`
fails the same way (`TURBO_E_NOT_IMPLEMENTED`) rather than being silently
ignored (`crates/turbo-core/src/runtime.rs`, test
`provider_paths_are_not_silently_ignored`).
