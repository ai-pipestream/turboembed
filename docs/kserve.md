# The Open Inference Protocol over gRPC

This is the mapping of the Open Inference Protocol's gRPC service
(KServe v2, `inference.GRPCInferenceService`) onto the C interface,
`include/turbo/turbo.h`. A server that follows it is a projection of the
library: every answer it gives comes from a `turbo_*` call named here,
and it adds nothing the header does not have. No option has a second
name, no value is filled in that the header does not fill in, and no
vector is changed after `turbo_result` gives it.

The messages are those of `open_inference_grpc.proto` in the Open
Inference Protocol specification. KServe's `grpc_predict_v2.proto` has
the same messages and field numbers, less
`ModelMetadataResponse.properties` and the `double_param` and
`uint64_param` choices of `InferParameter`; a client built from it reads
every response here except the properties. This document covers the
gRPC binding only.

Status names below are the header's without the `TURBO_E_` prefix
where a table has no room for it; enum values written as text are the
header constant without the `TURBO_` prefix, as in the bundle manifest
(docs/bundle.md): `TRUNCATE_RIGHT`, `DTYPE_F32`, `CAP_SUPPORTED`.

## Serving a bundle

One served model is one bundle loaded on one device. The server makes
one `turbo_runtime` (`turbo_runtime_create`), one `turbo_context` per
device it uses (`turbo_context_create`), and for each model one
`turbo_model` (`turbo_model_load`) and a fixed number of sessions
(`turbo_session_create`), all made with the same `turbo_session_desc`.

The model name is the bundle: the name of the bundle directory, the
last component of the path handed to `turbo_model_load`, exactly as it
is spelled. Two models with the same name cannot be served together;
the server refuses to start. The model has one version, its
`turbo_model_info.revision`.

These are set in the server's configuration, per model, and never by a
request:

| Setting | Header | Absent |
|---|---|---|
| `bundle` | `bundle_path` of `turbo_model_load` | Required. |
| `device` | the runtime device index for `turbo_context_create`, or `select` for the index `turbo_runtime_select` gives for `TURBO_TASK_EMBED` | Required. The header has no default device. |
| `precision` | `turbo_session_desc.precision`: `PRECISION_MODEL`, `PRECISION_FASTEST` or `PRECISION_EXACT` | 0, `TURBO_PRECISION_MODEL`. |
| `max_batch` | `turbo_session_desc.max_batch` | 0, the model's. |
| `max_seq` | `turbo_session_desc.max_seq` | 0, the model's. |
| `sessions` | how many sessions `turbo_session_create` makes | Required, at least 1. |

The device and precision are therefore per model, not per request. The
header fixes a session's compute dtype at `turbo_session_create` and has
no per-run choice, so a request cannot name either; a request parameter
`precision` or `device` is an unknown parameter (Errors).

A failure in any of these calls stops the server before it is ready. It
logs the call, `turbo_status_name` of the code, the field if the error
names one, and `turbo_error.message`, and exits with a non-zero status.
A bundle whose `turbo_model_info.task` is not `TURBO_TASK_EMBED` is
refused the same way with `TURBO_E_UNSUPPORTED_TASK` (Tasks).

## RPCs

| RPC | Answer | From |
|---|---|---|
| `ServerLive` | `live` true | The process is up and answering gRPC. No library call. |
| `ServerReady` | `ready` true when every configured model is ready | `ModelReady` of each. |
| `ModelReady` | `ready` true when the model is loaded and all its sessions are made | `turbo_model_load` and every `turbo_session_create` returned `TURBO_OK`. |
| `ServerMetadata` | `name` `turboembed`, `version` the string `turbo_version()` returns, `extensions` empty | `turbo_version`. |
| `ModelMetadata` | name, versions, platform, inputs, outputs, properties | `turbo_model_get_info`, `turbo_session_get_info`, `turbo_runtime_device_info`, `turbo_runtime_capability`. |
| `ModelInfer` | the vectors, and the run's summary as response parameters | `turbo_embed_write_text` or `turbo_embed_write_tokens`, then `turbo_session_run`, `turbo_result_read` or `turbo_result_buffer`, `turbo_result_get_info`. |

Every other method of the service is answered as in Not served.

A request that names a model the server does not serve, or a
`version` / `model_version` that is neither empty nor that model's
revision, is `TURBO_E_BUNDLE_NOT_FOUND`: the model name is the bundle,
and there is no such bundle here. `ModelReady` answers it the same way
rather than with `ready` false.

## Readiness

The server answers gRPC before it loads anything, so `ServerLive` is
true while bundles are verified and loaded, which for a large artifact
takes as long as hashing its files (docs/bundle.md, Loader rules).

A model is ready only after `turbo_model_load` and every one of its
`turbo_session_create` calls have returned `TURBO_OK`. Until then
`ModelReady` is false, `ServerReady` is false, and `ModelMetadata` and
`ModelInfer` for it are `TURBO_E_INVALID_STATE`: a call made before the
state it needs. Once ready, a model stays ready until the server stops;
the server has no unload.

## Metadata

`ModelMetadataResponse`:

| Field | Value |
|---|---|
| `name` | The model name. |
| `versions` | One entry, `turbo_model_info.revision`. |
| `platform` | Empty. The protocol's platform names are a fixed list of other frameworks' formats, and the header reports no artifact format. |
| `inputs` | `texts` BYTES `[-1]`; `ids` INT32 `[-1, -1]`; `mask` INT32 `[-1, -1]`; `types` INT32 `[-1, -1]`. A request uses `texts` alone, or `ids` and `mask` with `types` optional (Inference). |
| `outputs` | `vectors` FP32 `[-1, -1]`. The second extent is `dim` unless `output_dim` asks for fewer. |
| `properties` | Below. |

The properties are the header's facts about the model on its device,
keyed by struct and field, the struct named without its `turbo_`
prefix. Integers are decimal, enum values are constant names without
`TURBO_`, floats are the shortest decimal that reads back as the same
`float`, and strings are the struct's, up to its NUL.

| Key | Source |
|---|---|
| `model_info.<field>` | `turbo_model_get_info`: `task`, `dim`, `pooling`, `normalize`, `max_seq`, `max_batch`, `dtype`, `model_id`, `revision`, `manifest_sha256`, `artifact_sha256`, `tokenizer_sha256`, `prefix_query`, `prefix_document`. |
| `session_info.<field>` | `turbo_session_get_info` of the model's first session (all are made alike): `max_batch`, `max_seq`, `precision`, `compute_dtype`. |
| `device_info.<field>` | `turbo_runtime_device_info` of the model's device: `kind`, `ordinal`, `unified_memory`, `memory_total`, `arch`, `name`, `vendor`, `backend`, `runtime_version`, `driver_version`. |
| `capability.<field>` | `turbo_runtime_capability` for the model's device, `turbo_model_info.task` and the sessions' `precision`: `status`, `dtype`, `options_honored`, `cosine_floor`, `speed_ratio`, `benchmark`, `reason`. |

`options_honored` is the bit mask as a decimal integer: bit (i-1) is
field i of `turbo_embed_options` (Parameters). `memory_free` is left
out; it is a reading of the moment, not a fact about the model. The
server reads these on each call.

## Inference

One `ModelInfer` is one write, one run and one read on one session. The
server never splits a request across runs or sessions, never joins
requests into one run, and never routes a request to a session by its
size.

### Inputs

The input tensors a request carries choose the write function. Tensor
names are the header's parameter and field names.

| Inputs | Datatype, shape | Call |
|---|---|---|
| `texts` | BYTES `[batch]` | `turbo_embed_write_text(s, texts, count = batch, opts)`, each element one `turbo_text` |
| `ids`, `mask`, and optionally `types` | INT32 `[batch, seq]` each, the same shape | `turbo_embed_write_tokens(s, batch, opts)`, with `turbo_token_batch` `batch`, `seq`, `row_stride` 0, and `ids`, `mask`, `types` pointing at the tensors; `types` absent is `types` NULL, all zero |

Any other set of inputs, a name given twice, or another datatype (INT64
ids, say) is `TURBO_E_INVALID_ARGUMENT`; nothing is converted. A shape
of the wrong rank, a negative extent, `mask` or `types` shaped unlike
`ids`, or contents whose element count is not the shape's product is
`TURBO_E_INVALID_SHAPE`. An extent larger than a `uint32_t` holds is
`TURBO_E_INVALID_SHAPE`.

The server checks only what it must to build the call. What the header
checks, the library checks: UTF-8 (`TURBO_E_INVALID_UTF8`), the batch
and sequence limits (`TURBO_E_CAPACITY`), a row with no mask entry of 1,
ids outside the vocabulary, and every option.

### Contents

A request carries all of its inputs as `raw_input_contents`, one entry
per input in the order of `inputs`, or all of them as typed `contents`.
Both in one request, or a count of raw entries other than the number of
inputs, is `TURBO_E_INVALID_ARGUMENT`.

| Input | Typed | Raw |
|---|---|---|
| `texts` | `bytes_contents`, one element per text | Each element a 4-byte little-endian unsigned length followed by that many bytes, back to back, with nothing left over. The protocol leaves the raw BYTES layout open; this is the one its common clients write. A malformed entry is `TURBO_E_INVALID_ARGUMENT`. |
| `ids`, `mask`, `types` | `int_contents` | Little-endian int32, row-major, 4 × batch × seq bytes. |

Contents in any other typed field are `TURBO_E_INVALID_ARGUMENT`.

Copies. Exactly one copy is the server's: raw INT32 contents are copied
into a `TURBO_PLACE_HOST` buffer (`turbo_buffer_alloc` on the model's
context, `turbo_buffer_host_ptr`), and `turbo_token_batch` points there.
It cannot be avoided: a protobuf `bytes` field has no alignment for
`int32_t`. Nothing else is copied by the server. Each `turbo_text`
points into the request's own bytes, typed or raw; typed `int_contents`
are already `int32_t` arrays once protobuf has parsed them, and
`turbo_token_batch` points at them. What the library does with the rows
after that is the header's: a CPU session copies them into its own
memory, and a session on a device with its own memory stages pageable
rows through its page-locked memory (`turbo_token_batch`).

### Output

One output tensor, `vectors`, datatype FP32, shape `[batch, dim]` from
`turbo_result_info.batch` and `.dim`, in `raw_output_contents[0]`: the
`turbo_result_info.bytes` bytes of the result, little-endian, row-major,
exactly as the library gives them. The server does not normalize, cut,
reorder, round or convert them. The response always uses
`raw_output_contents`; `contents` is never set.

`turbo_result_info.dtype` names the datatype: F32 in this cut. Were it
another, the datatype would be its name in the protocol (`DTYPE_F16`
FP16, `DTYPE_BF16` BF16, `DTYPE_I32` INT32), never a conversion.

Where the vectors are decides how they reach the response. When
`turbo_result_info.placement` is `TURBO_PLACE_HOST`, `TURBO_PLACE_PINNED`
or `TURBO_PLACE_SHARED`, the server takes `turbo_result_buffer` and
`turbo_buffer_host_ptr` and the response's bytes are that memory, not a
copy of it. When it is `TURBO_PLACE_DEVICE`, `turbo_result_read` writes
them into the response's bytes; that read is the download, counted in
`d2h_bytes`. Either way the result and its buffer are released once the
response is written, and only then is the session free again.

`outputs` in the request may be empty, meaning every output, or name
`vectors` once. Another name is `TURBO_E_INVALID_ARGUMENT`.

The response's `model_name` is the model name, `model_version` the
revision, and `id` the request's `id`.

### Response parameters

The response's `parameters` are the scalar fields of
`turbo_result_info`, read with `turbo_result_get_info` after the vectors
are read, so `d2h_bytes` includes that read. Integers are `int64_param`,
enum values and strings `string_param` as in Metadata.

| Parameter | Field |
|---|---|
| `task`, `batch`, `dim`, `dtype`, `compute_dtype`, `placement`, `device` | the same |
| `bytes`, `h2d_bytes`, `d2h_bytes`, `host_allocs`, `device_allocs` | the same |
| `backend`, `arch`, `runtime_version`, `manifest_sha256`, `artifact_sha256`, `tokenizer_sha256` | the same |

`stage_count` and `stage` are not carried: a parameter holds one
scalar, and the stages are an array.

## Parameters

A request's `parameters` are the fields of `turbo_embed_options`, by
their header names, and nothing else. The server fills one
`turbo_embed_options` with `struct_size` set and every field 0, sets the
fields the request names, and passes it to the write. A field the
request leaves out stays 0, which the header defines as what the bundle
says; the server has no default of its own.

| Parameter | Field | Type | Meaning | Absent |
|---|---|---|---|---|
| `truncate` | 1 | `string_param`: `TRUNCATE_MODEL`, `TRUNCATE_NONE`, `TRUNCATE_RIGHT`, `TRUNCATE_LEFT` | How a text longer than the token budget is cut; `TRUNCATE_NONE`: too long is an error. Texts only. | `TRUNCATE_MODEL`: the bundle's `tokenizer.truncation`. |
| `max_tokens` | 2 | `int64_param`, 0 to 4294967295 | Token budget per row, specials included. Above the session's `max_seq` is `TURBO_E_CAPACITY`. | 0: what the bundle says. |
| `prompt_role` | 3 | `string_param`: `PROMPT_NONE`, `PROMPT_QUERY`, `PROMPT_DOCUMENT` | Prepend the bundle's query or document prefix. Texts only. | `PROMPT_NONE`: no prefix. |
| `normalize` | 4 | `string_param`: `NORMALIZE_MODEL`, `NORMALIZE_NONE`, `NORMALIZE_L2` | Normalization of each vector. | `NORMALIZE_MODEL`: the bundle's `embed.normalize`. |
| `pooling` | 5 | `string_param`: `POOLING_MODEL`, `POOLING_MEAN`, `POOLING_CLS`, `POOLING_LAST` | How token states become one vector. | `POOLING_MODEL`: the bundle's `embed.pooling`. |
| `output_dim` | 6 | `int64_param`, 0 to 4294967295 | Keep the first `output_dim` values of each vector, cut before normalize. | 0: what the bundle says, the full `dim`. |

For rows given as tokens, `truncate` and `prompt_role` other than their
0 value are `TURBO_E_INVALID_ARGUMENT` naming the field, from
`turbo_embed_write_tokens`: the rows are already cut.

The server refuses, before any library call:

- a parameter name that is not a field above: `TURBO_E_INVALID_ARGUMENT`,
  field 0, the message naming it;
- a value of another `InferParameter` type (`bool_param`,
  `double_param`, `uint64_param`, or a number as a string):
  `TURBO_E_INVALID_ARGUMENT` naming the field;
- a number outside 0 to 4294967295: `TURBO_E_INVALID_ARGUMENT` naming
  the field;
- a string that is not one of that field's constants, including another
  field's constant, a lower-case spelling, or the name with its `TURBO_`
  prefix: `TURBO_E_INVALID_ENUM`, the message naming the field and the
  value.

Whether the model on this device honors an option is the library's to
say. An option it cannot honor fails the write with
`TURBO_E_UNSUPPORTED_OPTION` and `turbo_error.field` naming it, and the
server answers that, mapped as in Errors. The server does not consult
`capability.options_honored` to refuse or drop anything itself; it
publishes it so a client can look before asking.

Input tensor `parameters` and requested output `parameters` are not
defined here; any is `TURBO_E_INVALID_ARGUMENT`.

## Batch and sequence limits

The limits are the session's, `turbo_session_info.max_batch` and
`max_seq`, as published under `session_info` in the model's properties.
They come from the configuration's `max_batch` and `max_seq`, 0 being
the model's `turbo_model_info` values.

| Request | Answer |
|---|---|
| More texts, or more token rows, than `max_batch` | `TURBO_E_CAPACITY` from the write. The server does not split it. |
| Token rows with `seq` above `max_seq` | `TURBO_E_CAPACITY` from `turbo_embed_write_tokens`. Nothing is cut. |
| A text whose tokens exceed the budget | Cut only as `truncate` says. `TRUNCATE_MODEL` cuts as the bundle says at its `embed.max_seq`; a row that fits the model but not the session (a fixed-shape artifact's smaller `max_seq`) is `TURBO_E_CAPACITY`, never cut differently on one device (docs/bundle.md, docs/conformance.md). `TRUNCATE_NONE`: too long is an error. |
| `max_tokens` above `max_seq` | `TURBO_E_CAPACITY`. |

## Concurrency

Each model has the configured number of sessions, made at load. The
model, context and runtime are shared by all of them; the header allows
that from any thread, and a precision that needs a converted copy of the
weights makes one copy, shared by every session of the model.

A `ModelInfer` takes an idle session of its model and holds it from the
write until the response is written and the result, and any buffer from
`turbo_result_buffer`, are released; the header keeps a session busy
until then. When every session of the model is held, the request is
answered at once with `TURBO_E_BUSY`, the status the header gives a call
on a busy session. The server keeps no queue and does not wait.

The header has no way to stop a run. A request whose client cancels,
or whose deadline passes, during a run keeps its session until the run
ends and the result is released; the answer is then dropped.

`ModelReady`, `ServerReady`, `ServerLive`, `ServerMetadata` and
`ModelMetadata` take no session and are never `TURBO_E_BUSY`.

## Errors

Every error the server answers is a turbo status code. The gRPC status
code is from the table; the status message is

```
<turbo_status_name(code)>: <message>
<turbo_status_name(code)> field <n> (<name>): <message>
```

the second form when `turbo_error.field` is not 0, `<name>` being the
field's header name in the struct the call took: `turbo_embed_options`
for a write, `turbo_session_desc` for `turbo_session_create`.
`<message>` is `turbo_error.message`, or for a refusal the server makes
itself, one line saying what in the request was refused. The trailing
metadata carries `turbo-code`, the status code in decimal, and
`turbo-field`, the field number, 0 when none.

| Turbo status | Code | gRPC status | When it reaches a client |
|---|---|---|---|
| `TURBO_E_INVALID_ARGUMENT` | 256 | `INVALID_ARGUMENT` | A bad request: inputs, contents, parameters. |
| `TURBO_E_INVALID_STRUCT_SIZE` | 257 | `INTERNAL` | Never, unless the server is wrong. |
| `TURBO_E_INVALID_UTF8` | 258 | `INVALID_ARGUMENT` | A text that is not UTF-8. |
| `TURBO_E_INVALID_HANDLE` | 259 | `INTERNAL` | Never, unless the server is wrong. |
| `TURBO_E_INVALID_SHAPE` | 260 | `INVALID_ARGUMENT` | A tensor's shape or element count. |
| `TURBO_E_INVALID_STATE` | 261 | `FAILED_PRECONDITION` | The model is not loaded yet (Readiness). |
| `TURBO_E_INVALID_ENUM` | 262 | `INVALID_ARGUMENT` | A parameter value that is no constant of its field. |
| `TURBO_E_UNSUPPORTED` | 512 | `UNIMPLEMENTED` | This device cannot do what was asked. |
| `TURBO_E_UNSUPPORTED_OPTION` | 513 | `UNIMPLEMENTED` | An option this model on this device does not honor, the field named. |
| `TURBO_E_UNSUPPORTED_TASK` | 514 | `UNIMPLEMENTED` | A task the model or device does not offer. |
| `TURBO_E_OUT_OF_MEMORY` | 768 | `RESOURCE_EXHAUSTED` | Host or device memory ran out. |
| `TURBO_E_BUSY` | 769 | `UNAVAILABLE` | Every session of the model is held. |
| `TURBO_E_CAPACITY` | 771 | `OUT_OF_RANGE` | Over the session's `max_batch` or `max_seq`, or `max_tokens` over `max_seq`. |
| `TURBO_E_DEVICE_NOT_FOUND` | 1024 | `FAILED_PRECONDITION` | At startup only. |
| `TURBO_E_DEVICE_UNAVAILABLE` | 1025 | `UNAVAILABLE` | At startup only. |
| `TURBO_E_RUNTIME` | 1026 | `INTERNAL` | The vendor runtime failed; the message has its text. |
| `TURBO_E_BUNDLE_NOT_FOUND` | 1280 | `NOT_FOUND` | A model name or version the server does not serve. |
| `TURBO_E_BUNDLE_INVALID` | 1281 | `FAILED_PRECONDITION` | At startup only. |
| `TURBO_E_BUNDLE_INTEGRITY` | 1282 | `DATA_LOSS` | At startup only. |
| `TURBO_E_BUNDLE_NO_ARTIFACT` | 1283 | `FAILED_PRECONDITION` | At startup only. |
| `TURBO_E_INTERNAL` | 1536 | `INTERNAL` | A bug in the library. |
| `TURBO_E_PANIC` | 1537 | `INTERNAL` | Caught at the language boundary. |
| any other | | `UNKNOWN` | A code this table does not have; the message carries `turbo_status_name` of it. |

`TURBO_OK` is `OK`. "At startup only" codes come from loading, which
stops the server (Serving a bundle); the table gives them so the mapping
is total.

`UNSUPPORTED_OPTION` is `UNIMPLEMENTED` rather than `INVALID_ARGUMENT`
because the same request is valid for another model or device: this
service does not do it, and retrying it here will not help. `CAPACITY`
is `OUT_OF_RANGE` for the same reason: the request is well formed and
over this session's limits. `BUSY` is `UNAVAILABLE`, the one a client
may retry as it is.

## Tasks

The server serves `TURBO_TASK_EMBED`, the one task the header has. A
bundle of any other task is refused at startup with
`TURBO_E_UNSUPPORTED_TASK`.

The header grows a task by one rule: a task number `TURBO_TASK_<TASK>`,
an options struct `turbo_<task>_options`, one `turbo_<task>_write_<input>`
per input shape, `turbo_<task>_read_<what>` only when the output is not
one `[batch, dim]` block, and model facts at the end of
`turbo_model_info`. The mapping grows with it, by the rules embed
follows here:

- `turbo_model_info.task` of the loaded bundle chooses the task; the
  model's metadata lists that task's inputs and outputs.
- Parameters are the fields of `turbo_<task>_options` by their header
  names, 0 when absent, with the same refusals.
- Each `turbo_<task>_write_<input>` is one input set, its tensors named
  after its parameters or fields.
- A `[batch, dim]` output is one tensor as in Output; each
  `turbo_<task>_read_<what>` is one more.
- New model facts join the `model_info` properties.

Rerank, or any other task, is mapped in this document when the header
defines it, and not before. Until then its bundles are refused as above.

## Not served

The server implements the six RPCs above. What the protocol, or
extensions servers commonly add to it, has beyond the library is
answered as follows.

| Protocol | Answer |
|---|---|
| Any method not in `GRPCInferenceService` (repository load and unload, streaming inference, shared-memory registration, statistics, trace, logging, `grpc.health.v1.Health`) | `UNIMPLEMENTED`, from gRPC: the server does not register them. |
| `ServerMetadataResponse.extensions` | Empty: no extension is implemented. |
| More than one version of a model | One: the revision. |
| Typed output `contents` | Not used; outputs are always raw. |
| Datatypes other than BYTES for `texts` and INT32 for `ids`, `mask`, `types` | `TURBO_E_INVALID_ARGUMENT`. |
| Request parameters other than `turbo_embed_options` fields (priority, timeout, sequence or classification parameters) | `TURBO_E_INVALID_ARGUMENT`. |
| Input and requested-output tensor parameters | `TURBO_E_INVALID_ARGUMENT`. |
| Per-request device, precision or session shape | Not in a request; set per model (Serving a bundle). |
| Tokenizing without embedding (`turbo_tokenizer_encode`, `turbo_tokenizer_count`) | No RPC; the protocol has none. |
| `platform` | Empty (Metadata). |

## Open questions for the header

- `turbo_status_name` does not say whether its name has the `TURBO_E_` prefix (or what it returns for `TURBO_OK` and an unknown code), and the error message format quotes it.
- `turbo_model_info` does not report the bundle's `embed.output_dims`, so metadata cannot say which `output_dim` values the model takes or give the output's width exactly.
- `turbo_embed_options.max_tokens` says 0 is "what the bundle says" without naming the bundle value (`turbo_encode_options.max_tokens` names `max_seq`).
- `turbo_embed_options.max_tokens` does not say what it does for rows written with `turbo_embed_write_tokens`: applied, ignored, or refused as `truncate` and `prompt_role` are.
- `TURBO_TRUNCATE_NONE` says "too long is an error" without naming the status code (`TURBO_E_CAPACITY` or `TURBO_E_INVALID_ARGUMENT`).
- There is no way to read a bundle's task before `turbo_model_load`, and `turbo_runtime_select` takes the task, so `device = select` can only name `TURBO_TASK_EMBED` until the header offers one.
- The header gives the output block no name; this document calls it `vectors`, the word `turbo_result_read` and `turbo_result_buffer` use.
