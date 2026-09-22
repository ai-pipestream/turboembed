# Architecture

This describes what the code in this tree does on branch `turbo-v2` as of 2026-09-21
(milestones P0, P1, and P2 done; P3 landed on x86_64 and, for embedding, on
Jetson `nano1`; P5 landed the `hailo` embedding provider on two Hailo-8
Pis; P6 landed the `ggml` generation provider and the push-style
`turbo_generate`; P7 landed the Java FFM binding and the Swift package).
It is derived from `PLAN.md` section 4; where this tree does not yet
implement something `PLAN.md` describes, that is marked "planned" with the
milestone that adds it. See `PLAN.md` itself for the design rationale.

## Layers

```
 bindings:   Rust crate (crates/turbo) | Java (bindings/java, JDK 25 FFM) |
             Swift (bindings/swift, PipestreamTurbo over the C ABI): landed
             Android/JNI: planned (P10)
 ------------------------------------------------------------------------
 libturbo:   crates/turbo-capi (C ABI) over crates/turbo-core, packaged as
             libturbo by crates/turbo-shared
             runtime + provider registry | device discovery | buffers
             bundles | tokenizers | chunker | sessions | results | errors
             provider plugin loading (turbo_runtime_load_provider)
 ------------------------------------------------------------------------
 providers:  mock, static (built into libturbo; crates/turbo-core::mock and
             providers/static)
             openvino (providers/openvino, C++, loaded as a plugin library)
             cuda (providers/cuda, Rust, loaded as a plugin library; x86_64
             EXPERIMENTAL, embedding landed on Jetson aarch64)
             ggml (providers/ggml, Rust, loaded as a plugin library; GGUF
             generation EXPERIMENTAL on CUDA/CPU on krick and on Metal on
             Apple M2, all via llama.cpp)
             hailo (providers/hailo, C++ over HailoRT 4.x; embeddings
             EXPERIMENTAL on the Hailo-8 Pis with an INT8 HEF)
             metal: planned (P4)
 ------------------------------------------------------------------------
 runtimes:   none for mock/static; OpenVINO 2026.3.1 for openvino; ONNX
             Runtime 1.28 (1.24.0 on Jetson) CUDA execution provider for
             cuda; llama.cpp (via llama-cpp-2) for ggml; HailoRT 4.23 for
             hailo
```

`crates/turbo-capi` is a thin, panic-safe adapter: it validates handles and
`struct_size`, converts `turbo_text` views, calls into `turbo-core`, and
copies errors into the caller-owned `turbo_error`. All state and enforcement
live in `crates/turbo-core`. `crates/turbo-abi` is the single source of the
`#[repr(C)]` types and constants that `scripts/gen-header.sh` turns into
`include/turbo/turbo_types.h`, `turbo_provider.h`, and `turbo.h` with
cbindgen; see `crates/turbo-abi/src/lib.rs` and `crates/turbo-capi/src/lib.rs`.

Provider plugin loading (a provider as a separate `dlopen`-able library
exporting `turbo_provider_get`, per `PLAN.md` section 4.1) is implemented in
`crates/turbo-core/src/plugin.rs`. `Runtime::load_provider` (exposed as
`turbo_runtime_load_provider`) calls `libloading::Library::new`, resolves the
`turbo_provider_get` symbol, calls it with the core's `TURBO_ABI_VERSION`,
and validates the returned vtable (size, ABI version, every required
function pointer non-NULL) before registering the provider; a NULL return
from the entry point is `TURBO_E_ABI_MISMATCH`, a missing symbol or vtable
defect is `TURBO_E_PROVIDER_LOAD`. The loaded `libloading::Library` is
leaked (`Box::leak`) into a `&'static` reference and kept mapped for the
life of the process, on purpose: the runtimes providers wrap (CUDA, ONNX
Runtime, OpenVINO, llama.cpp) hold worker threads, driver contexts, and
global destructors that are not safe to tear down with `dlclose`, and
unloading a provider library and loading it again in the same process is
where intermittent crashes showed up on the Jetson with a dynamically
linked ONNX Runtime (`crates/turbo-core/src/plugin.rs` `LoadedProvider`).
Releasing a `turbo_runtime` (`turbo_runtime_release`) therefore never
unmaps a provider library it loaded; only the process exiting does.
`RuntimeDesc.provider_paths` (from
`turbo_runtime_desc.provider_paths`) is loaded the same way at creation, in
order; a failure fails the whole `turbo_runtime_create` call. There is no
default filesystem search path: a runtime always starts with the statically
linked built-in providers (`mock` and `static`, from `turbo::builtin_providers()`)
unless `TURBO_RUNTIME_NO_DEFAULT_PROVIDERS` (`RuntimeDesc.no_default_providers`)
is set, plus whatever `provider_paths` names explicitly. This differs from
`PLAN.md` section 4.1's "the core loads providers from a search path and
from explicit calls" — there is no search-path scan in this tree, only
built-in-or-not plus an explicit path list.

A Rust provider crate (`mock`, `static`, `cuda`, `ggml`) defines its
`turbo_provider_get` symbol with `turbo_core::export_provider!`
(`crates/turbo-core/src/plugin_export.rs`), which builds the vtable from a
`turbo_core::provider::Provider` implementation, catches panics at every
shim, and converts errors into the caller-owned `turbo_error`. A provider
written directly against `turbo_provider.h` (the OpenVINO provider, in C++)
implements the vtable and `turbo_provider_get` by hand instead; see
`docs/providers.md` for the ownership rules both routes must follow.

## Object model

```
turbo_runtime      library instance; owns the provider registry
  turbo_device     enumerated hardware; immutable info + capabilities
    turbo_context  device + memory domain; owns allocators
      turbo_buffer typed memory in a placement
      turbo_model  loaded bundle on a context; immutable contract + placement
        turbo_session  execution workspace; one in-flight op
          turbo_result   leased output of the last op
        turbo_generation streaming generation state
  turbo_tokenizer  from a bundle
    turbo_chunk_plan  byte-offset chunk plan over a text, counted by a tokenizer
```

This matches `crates/turbo-core/src/handles.rs` exactly: `Context` holds an
`Arc<Runtime>`, `Model` holds an `Arc<Context>`, `Session` holds an
`Arc<Model>`, `ResultHandle` holds an `Arc<Session>`, `Generation` holds an
`Arc<Model>`, and a `Buffer` returned from a result additionally holds an
`Arc<ResultHandle>` as its lease. Every handle is reference counted; releasing
a parent does not invalidate a live child (`handles.rs` test
`children_outlive_parents`). `turbo_tokenizer` (`crates/turbo-core/src/tokenizer.rs`)
loads independently of any device or context, straight from a bundle
directory; `turbo_chunk_plan` (`crates/turbo-core/src/chunker.rs`) is created
from a text and a tokenizer and holds only byte offsets, never a copy of the
text.

A session accepts one operation at a time: `Session::lock` uses
`Mutex::try_lock` and returns `TURBO_E_BUSY` on contention rather than
blocking (`handles.rs`). A result leases the session's output storage —
`Session::run` sets an atomic lease flag, and every subsequent write or run
on that session fails with `TURBO_E_BUSY` until every `ResultHandle`/`Buffer`
view derived from the result is dropped (`ResultHandle`'s `Drop` clears the
flag). `Generation` is single-owner the same way.

## Provider contract

A provider implements the traits in `crates/turbo-core/src/provider.rs` at
task granularity, not graph granularity:

- `Provider`: device enumeration, the capability matrix, `can_run`, and
  context creation.
- `ProviderContext`: buffer `alloc`/`import`, and `load_model`.
- `ProviderModel`: `info()` and session/generation creation.
- `ProviderSession`: `write_text`, `write_tokens`, `write_pairs`,
  `write_text_classify`, `bind` (for `RUN` models), `run`, `stats`. Every
  method except `run` and `stats` has a default that fails with
  `TURBO_E_UNSUPPORTED_TASK`, so a provider only implements the tasks its
  capability matrix actually offers.
- `ProviderGeneration`: `prompt`, `prompt_tokens`, `step`, `cancel`.

A provider reaches the core through one of two routes to the same vtable
(`include/turbo/turbo_provider.h`):

- A Rust provider (`mock`, `static`, `cuda`, `ggml`) implements the traits above and
  calls `turbo_core::export_provider!` once to generate its
  `turbo_provider_get` symbol; the macro
  (`crates/turbo-core/src/plugin_export.rs`) builds the vtable, boxes
  handles as `Arc<dyn Provider*>`/`Box<dyn Provider*>` behind `*mut c_void`,
  and wraps every entry point in `catch_unwind` so a panic becomes
  `TURBO_E_PANIC` instead of unwinding across the C boundary.
- A provider in another language (`openvino`, C++) implements the vtable and
  `turbo_provider_get` directly; it never goes through `turbo-core`'s Rust
  traits.

`crates/turbo-core/src/plugin.rs` is the other side: `PluginProvider` adapts
a loaded vtable back into the `Provider`/`ProviderContext`/... traits so a
plugin-loaded provider is indistinguishable from a built-in one everywhere
else in `turbo-core`.

`turbo-core` enforces everything that does not depend on hardware once, for
every provider: handle lifetimes, single-owner sessions and result leases,
option validation against the capability matrix (`Model::validate_embed`,
`validate_rerank`, `validate_classify`, `validate_generate` in `handles.rs`),
and error containment (panics at the C boundary are caught, see
`crates/turbo-capi/src/lib.rs`'s `boundary` function). Every option field in
every options struct now either gates on a `TURBO_CAP_OPT_*` bit this way or
is one of two unconditional checks every provider gets for free:
`max_tokens` above the session's `max_seq` is `TURBO_E_CAPACITY`
(`Session::check_budget`), and an `embed_options.prompt_role` naming a role
the bundle declares no prefix for is `TURBO_E_INVALID_ARGUMENT`
(`Model::validate_embed`); see `docs/c-api.md`'s capability-bit tables for
the full field list and which provider sets which bit.

The provider does the work and reports what it actually did through
`ModelInfo::stages` (`stage_placement[TOKENIZE|ENCODE|POOL|NORMALIZE|
POSTPROCESS]` and `fully_accelerated`). The mock provider reports every
applicable stage as `HOST` (`crates/turbo-core/src/mock.rs`), which is
honest for a provider with no device backing it. The OpenVINO provider
reports `ENCODE`, `POOL`, `NORMALIZE`, and `POSTPROCESS` as `FUSED` on GPU
(one compiled graph) and `TOKENIZE` as `HOST` always, since WordPiece runs
on the CPU (`providers/openvino/src/provider.cpp`); `fully_accelerated` is
therefore 0 for every OpenVINO model (tokenization never moves off the host
in this tree). The CUDA provider reports `TOKENIZE` as `HOST` (native
WordPiece on the CPU), `ENCODE` as `DEVICE`, `POOL`/`NORMALIZE` as `DEVICE`
for embedding models, and `POSTPROCESS` as `DEVICE` for rerank/classify
(the activation kernel) but `HOST` for token-classify (span aggregation
reads the device output back once); `fully_accelerated` is therefore also 0
for every CUDA model (`providers/cuda/src/lib.rs`, see `docs/providers.md`).
`ggml` reports `ENCODE` and, for its GGUF embedding cells, `POOL` as
`DEVICE` on a GPU device and `HOST` on its CPU device (the same
`llama_batch` decode either way, just on a different `ggml_backend_dev`),
`NORMALIZE` always `HOST` (L2 and `output_dim` run after the pooled vector
comes back), and `TOKENIZE` and, for generation, `POSTPROCESS` always
`HOST` (`providers/ggml/src/lib.rs`). `hailo` reports `ENCODE` as `DEVICE`
(the HEF through vstreams) and every other stage — `TOKENIZE`, the
word-embedding gather, `POOL`, `NORMALIZE` — as `HOST`, since the public
HEF holds only the transformer body. `fully_accelerated` is 0 for both.

## Capability matrix

Capabilities are per (device, task, modality). `Provider::capability`
returns a `Capability { status, dtype, reference_dtype, cosine_floor,
max_abs_error, deterministic, notes }`; `status` is one of `UNSUPPORTED`,
`PLANNED`, `EXPERIMENTAL`, `SUPPORTED` (`TURBO_CAP_*` constants, now defined
in `turbo_provider.h`). The mock provider reports `SUPPORTED` with
`cosine_floor = 1.0` and `max_abs_error = 0.0` for every task on `TEXT`
modality on its two devices (ordinal 0 = CPU, ordinal 1 = Accel), and
`UNSUPPORTED` for every other modality or ordinal
(`MockProvider::capability`). This is honest for the mock because its
outputs are a pure, deterministic function of the input, not because
`SUPPORTED` is a default: the mock also applies the bundle's declared
`contract.activation` to rerank and classify scores (softmax, sigmoid, or
none, matching what a real model's head would do) rather than returning raw
numbers, and models sampled generation deterministically: a positive
`temperature` shrinks the token pool by `top_k`/`top_p`/`min_p` and draws
from a seeded distribution, while `temperature == 0` (greedy) always picks
the same token and ignores `seed` (`crates/turbo-core/src/mock.rs`).
`static`, `openvino`, `cuda`, `ggml`, and `hailo` report `EXPERIMENTAL` for
the cells they offer (`EMBED x TEXT x CPU` for `static`;
`{EMBED,RERANK,CLASSIFY,TOKEN_CLASSIFY} x TEXT x {GPU,CPU}` for `openvino`;
the same four tasks x `TEXT` x GPU for `cuda` on `krick`, x86_64;
`GENERATE` x GPU and CPU for `ggml` on `krick` and `krickert-mac`;
`EMBED x TEXT` on the Hailo-8 NPU for `hailo` on `pi5ai1` and `cm5ai1`)
because a conformance and precision receipt exists for each
(`testdata/receipts/turbo/openvino-*-2026-09-21.json`,
`testdata/receipts/turbo/cuda-2026-09-21.json`,
`testdata/receipts/turbo/ggml-2026-09-21.json`,
`testdata/receipts/turbo/hailo-2026-09-21.json`) but the matched-native
benchmark receipt `PLAN.md` section 2 item 7 requires before `SUPPORTED`
does not yet exist for any of the five. OpenVINO NPU devices are enumerated
but offer no capability cells (listed, not qualified). CUDA on Jetson
(`nano1`, aarch64) is no longer untried: the device-enumeration fix in
`providers/cuda/src/cuda.rs` (reading compute capability through
`cudaDeviceGetAttribute` and the device name through the driver library's
`cuDeviceGetName`, both stable across CUDA toolkit majors, instead of the
runtime's re-versioned `cudaGetDeviceProperties`) unblocked loading the
provider there, and all 12 live embedding tests now pass at cosine 1.000
against a dynamically linked ONNX Runtime 1.24.0; there is no committed
receipt for `nano1` yet and the task suite (rerank, classify,
token-classify) is still being verified on that board (`PLAN.md`
section 10, P3).

Two further checks exist:

- `turbo_runtime_capability(rt, device_index, task, modality, ...)` reads
  the static cell before any allocation
  (`turbo_capi::turbo_runtime_capability`).
- `turbo_can_run(rt, device_index, bundle_path, task, modality, ...)` also
  opens and verifies the bundle and calls `Provider::can_run`, so it catches
  bundle-specific infeasibility the static cell cannot (for example: the
  mock rejects a bundle with no `mock` artifact even though the cell is
  `SUPPORTED`).

`turbo_model_get_info` reports what actually happened after load: `dim`,
`pooling`, `normalize`, `max_seq`, `max_batch`, `dtype_used`,
`fully_accelerated`, and `stage_placement`, all frozen from the bundle
contract and the provider's own report (`turbo_model_info` in
`turbo_types.h`; `ModelInfo` in `provider.rs`).

## Memory model

`crates/turbo-core/src/buffer.rs` defines `BufferDesc` (placement, dtype,
shape, strides, bytes) with checked arithmetic and a `HostBuffer` allocator.
The mock and `static` providers' `ProviderContext::alloc` still only accept
`TURBO_PLACE_HOST`, rejecting anything else with
`TURBO_E_UNSUPPORTED_PLACEMENT`. The OpenVINO and CUDA providers produce
`TURBO_PLACE_DEVICE` buffers: OpenVINO keeps GPU results in an OpenCL
`cl_mem` remote tensor (exportable as `TURBO_HANDLE_CL_MEM`) and CPU results
in host memory directly (`providers/openvino/src/provider.cpp`); CUDA keeps
every result on the device (exportable as `TURBO_HANDLE_CUDA_PTR`,
`providers/cuda/src/lib.rs`), both until `turbo_result_read` copies to host
(see `docs/providers.md`). `PINNED` and `SHARED` placements, and the
remaining native handle kinds (Level Zero USM, `MTLBuffer`, DMA-BUF fd), are
declared in the ABI (`TURBO_PLACE_*`, `TURBO_HANDLE_*`) for the providers
that need them, starting P4; there is no code path that produces them yet.

Sessions preallocate their input/output storage at creation for the declared
maximum shape (`MockModel::create_session` sizes `out`, `sorted`, `ids`,
`mask` once); `run` does not grow them. `turbo_session_stats` /
`SessionStats` reports `host_allocs` (or "not counted" when the adapter keeps no tally; a hardcoded zero is not allowed), `h2d_bytes`, `d2h_bytes`,
`input_bytes`, `output_bytes`, and a provider-reported `provider_allocs`
counter (`u64::MAX` sentinel for "unknown" at the ABI boundary). The mock's
own `allocs` counter only increments after the first run (see
`MockSession::note_alloc`), matching the "`allocs_per_run == 0` after
warmup" contract in `PLAN.md` section 10 for capability cells that claim it.
A result's `turbo_result_read` is an explicit blocking host copy
(`ResultHandle::read`); `turbo_result_buffer` returns a buffer view that
keeps the result's lease alive instead. Asynchronous execution
(`turbo_session_submit` + fences, `TURBO_CAP_ASYNC`) is declared in the
capability bitset but has no implementation; it is planned alongside the
providers that need it (Hailo async in P5).

## Threading model

Runtime, context, and model handles have no interior mutability that needs
synchronization beyond what `Arc` already gives them (`Context`, `Model` in
`handles.rs` hold no `Mutex`) and are documented as usable from any thread.
Session and generation are single-owner: `Session::lock` /
`Generation::lock` use `Mutex::try_lock` and turn `WouldBlock` into
`TURBO_E_BUSY` rather than blocking the caller, which is what "a concurrent
call returns `BUSY` without corrupting state" means concretely in this
tree. Callback reentry rejection for generation callbacks and Rust
`Send`-not-`Sync` typing for session/generation wrapper types
(`PLAN.md` section 4.6) apply to bindings other than the raw C ABI and are
exercised by the threading group of the Rust conformance suite
(`crates/turbo-conformance/tests/threading_rust.rs`,
`threading_c.rs`); see `docs/testing.md`.

## Error model

`crates/turbo-abi` defines one `i32` status space graded by the hundreds
digit: `0x000` ok, `0x1xx` argument/contract, `0x2xx` unsupported
(capability), `0x3xx` resource, `0x4xx` device/runtime, `0x5xx` bundle
integrity, `0x6xx` internal. See `docs/c-api.md` for the full table. Errors
are caller-owned (`turbo_error`, may be `NULL`); no thread-local or shared
error state exists anywhere in `turbo-capi` (`fail`/`clear`/`boundary` in
`crates/turbo-capi/src/lib.rs` write only into the caller's pointer, and
only into the fields the caller's `struct_size` covers). A Rust panic
crossing the C boundary is caught by `catch_unwind` in `boundary` and turned
into `TURBO_E_PANIC` rather than unwinding into C.

## Versioning

`TURBO_ABI_VERSION` is `2` (`crates/turbo-abi`). Every public struct starts
with `uint32_t struct_size`; the library reads only fields below the
caller's declared size, and accepts a size only when it is the end of a
field the struct has ever had, per struct in a table
(`crates/turbo-abi/src/versioned.rs`, generated by
`scripts/gen-versioned.py`); any other size, including one that ends inside
a field, is `TURBO_E_INVALID_STRUCT_SIZE` (`check_size` in `turbo-capi`; see
`docs/c-api.md`'s "Descriptor versioning rule").
Enumerations in ABI position are `u32` constants, never a C `enum`; an
unrecognized value is `TURBO_E_INVALID_ENUM` (the `abi_enum!` macro in
`crates/turbo-core/src/types.rs`). `TURBO_PROVIDER_ABI_VERSION` equals
`TURBO_ABI_VERSION`; a test asserts they stay equal. The header set (three
files: `turbo_types.h`, `turbo_provider.h`, `turbo.h`) is generated by
cbindgen from `turbo-abi` and `turbo-capi` and is never hand-edited; see
`scripts/gen-header.sh` and `AGENTS.md`.
