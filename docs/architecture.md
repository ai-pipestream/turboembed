# Architecture

This describes what the code in this tree does at commit `fc1f828`
(milestone P0). It is derived from `PLAN.md` section 4; where this tree does
not yet implement something `PLAN.md` describes, that is marked "planned"
with the milestone that adds it. See `PLAN.md` itself for the design
rationale.

## Layers

```
 bindings:   Rust crate (crates/turbo)                         | Java, Swift, C/C++: planned (P7, P10)
 ------------------------------------------------------------------------
 libturbo:   crates/turbo-capi (C ABI) over crates/turbo-core
             runtime + provider registry | device discovery | buffers
             bundles | sessions | results | errors
             tokenizers, chunker, dynamic provider loading: declared, P1
 ------------------------------------------------------------------------
 providers:  mock (crates/turbo-core::mock, packaged as providers/mock)
             cpu, cuda, openvino, metal, hailo, ggml: planned (P1, P3, P2, P4, P5, P6)
 ------------------------------------------------------------------------
 runtimes:   none yet; the mock provider has no external runtime dependency
```

`crates/turbo-capi` is a thin, panic-safe adapter: it validates handles and
`struct_size`, converts `turbo_text` views, calls into `turbo-core`, and
copies errors into the caller-owned `turbo_error`. All state and enforcement
live in `crates/turbo-core`. `crates/turbo-abi` is the single source of the
`#[repr(C)]` types and constants that `scripts/gen-header.sh` turns into
`include/turbo/turbo.h` and `turbo_types.h` with cbindgen; see
`crates/turbo-abi/src/lib.rs` and `crates/turbo-capi/src/lib.rs`.

Provider plugin loading (a provider as a separate `dlopen`-able library
exporting `turbo_provider_get`, per `PLAN.md` section 4.1) is declared in the
ABI (`turbo_runtime_load_provider`, `TURBO_PROVIDER_ENTRY_SYMBOL`) but not
implemented: `Runtime::new` returns `TURBO_E_NOT_IMPLEMENTED` if
`RuntimeDesc.provider_paths` is non-empty (`crates/turbo-core/src/runtime.rs`).
This is planned for P1. Today the only provider is the one statically linked
into `turbo-core` (`turbo_core::builtin_providers()`, which returns the mock
provider).

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
  turbo_tokenizer  from a bundle: planned, P1
```

This matches `crates/turbo-core/src/handles.rs` exactly: `Context` holds an
`Arc<Runtime>`, `Model` holds an `Arc<Context>`, `Session` holds an
`Arc<Model>`, `ResultHandle` holds an `Arc<Session>`, `Generation` holds an
`Arc<Model>`, and a `Buffer` returned from a result additionally holds an
`Arc<ResultHandle>` as its lease. Every handle is reference counted; releasing
a parent does not invalidate a live child (`handles.rs` test
`children_outlive_parents`). `turbo_tokenizer` and `turbo_chunk_plan` are
declared in the header (`turbo_tokenizer_create`, `turbo_chunk_plan_create`)
and return `TURBO_E_NOT_IMPLEMENTED`; they are planned for P1.

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

`turbo-core` enforces everything that does not depend on hardware once, for
every provider: handle lifetimes, single-owner sessions and result leases,
option validation against the capability matrix (`Model::validate_embed`,
`validate_rerank`, `validate_classify`, `validate_generate` in `handles.rs`),
and error containment (panics at the C boundary are caught, see
`crates/turbo-capi/src/lib.rs`'s `boundary` function). The provider does the
work and reports what it actually did through `ModelInfo::stages`
(`stage_placement[TOKENIZE|ENCODE|POOL|NORMALIZE|POSTPROCESS]` and
`fully_accelerated`). The mock provider reports every applicable stage as
`HOST` (`crates/turbo-core/src/mock.rs`), which is honest for a provider with
no device backing it; a hardware provider is expected to report `DEVICE` or
`FUSED` for stages it actually runs off the host (`PLAN.md` section 4.7).

## Capability matrix

Capabilities are per (device, task, modality). `Provider::capability`
returns a `Capability { status, dtype, reference_dtype, cosine_floor,
max_abs_error, deterministic, notes }`; `status` is one of `UNSUPPORTED`,
`PLANNED`, `EXPERIMENTAL`, `SUPPORTED` (`TURBO_CAP_*` constants in
`turbo_types.h`). The mock provider reports `SUPPORTED` with
`cosine_floor = 1.0` and `max_abs_error = 0.0` for every task on `TEXT`
modality on its two devices (ordinal 0 = CPU, ordinal 1 = Accel), and
`UNSUPPORTED` for every other modality or ordinal
(`MockProvider::capability`). This is honest for the mock because its
outputs are a pure, deterministic function of the input, not because
`SUPPORTED` is a default — a real provider only reports `SUPPORTED` once it
carries a conformance and precision receipt (`PLAN.md` section 2, item 7).

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
Today only `TURBO_PLACE_HOST` is actually allocatable: the mock provider's
`ProviderContext::alloc` rejects any other placement with
`TURBO_E_UNSUPPORTED_PLACEMENT` (`MockContext::alloc` in `mock.rs`), and
`import` accepts only `TURBO_HANDLE_HOST_PTR`. `PINNED`, `DEVICE`, and
`SHARED` placements, and every other native handle kind (CUDA pointer,
`cl_mem`, Level Zero USM, `MTLBuffer`, DMA-BUF fd), are declared in the ABI
(`TURBO_PLACE_*`, `TURBO_HANDLE_*`) for hardware providers to implement
starting P2 (OpenVINO) and continuing through P3-P6; there is no code path
that produces them yet.

Sessions preallocate their input/output storage at creation for the declared
maximum shape (`MockModel::create_session` sizes `out`, `sorted`, `ids`,
`mask` once); `run` does not grow them. `turbo_session_stats` /
`SessionStats` reports `host_allocs`, `h2d_bytes`, `d2h_bytes`,
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
exercised only by the (not yet written) conformance suite; see
`docs/testing.md`.

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
caller's declared size, and a size it does not recognize (smaller than the
minimum needed to report a code, or larger than the struct the library
knows) is `TURBO_E_INVALID_STRUCT_SIZE` (`check_size` in `turbo-capi`).
Enumerations in ABI position are `u32` constants, never a C `enum`; an
unrecognized value is `TURBO_E_INVALID_ENUM` (the `abi_enum!` macro in
`crates/turbo-core/src/types.rs`). The header itself is generated by
cbindgen from `turbo-abi` and `turbo-capi` and is never hand-edited; see
`scripts/gen-header.sh` and `AGENTS.md`.
