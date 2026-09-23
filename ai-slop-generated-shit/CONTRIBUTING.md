# Contributing to Turbo

Read [`PLAN.md`](PLAN.md) first; it is the governing design document. Read
[`AGENTS.md`](AGENTS.md) for the working rules this file assumes. This
document covers toolchain setup, the exact commands, and the review
checklist.

## Toolchain

- Rust: the version pinned in [`rust-toolchain.toml`](rust-toolchain.toml)
  (`stable`, with `rustfmt` and `clippy` components). `rustup` picks this up
  automatically inside the repository.
- `cbindgen`, for regenerating `include/turbo/*.h`:
  `cargo install cbindgen --locked`.
- A C compiler on `PATH` (`cc`; CI also compiles the header under
  `c++`/C++17) to build and run the C smoke test.
- `protoc` is not needed yet; it becomes relevant once `server/` (P9) is
  built against the vendored proto files.
- To build the OpenVINO provider (optional; not part of the default
  workspace build): an OpenVINO archive distribution (2026.3 or later),
  OpenCL headers and loader, CMake 3.20+, and a C++17 compiler. See
  [`providers/openvino/README.md`](providers/openvino/README.md).
- To build the CUDA provider (`providers/cuda`, a workspace member, so the
  commands below exclude it by default): a CUDA toolkit with `nvcc` and
  `cuda_runtime.h`, `libcudart.so`, and network access on the first build
  (the `ort` crate downloads a prebuilt ONNX Runtime CUDA bundle). See
  [`providers/cuda/README.md`](providers/cuda/README.md).
- The `ggml` provider (`providers/ggml`) is also a workspace member, but
  unlike CUDA and OpenVINO it is *not* excluded below: its default (CPU
  backend) build needs only `cmake` and a C/C++ compiler, which
  `llama-cpp-sys-2` uses to build llama.cpp from source at compile time (the
  first build is noticeably slower for this reason). `cargo build
  --workspace --exclude turbo-provider-cuda` therefore builds `ggml`'s CPU
  backend too. The `cuda`/`metal`/`vulkan` ggml backends are opt-in Cargo
  features, not built by the commands below. See
  [`providers/ggml/README.md`](providers/ggml/README.md).

## Building and testing

```bash
cargo build --workspace --exclude turbo-provider-cuda
cargo test --locked --workspace --exclude turbo-provider-cuda
cargo clippy --locked --workspace --exclude turbo-provider-cuda --all-targets -- -D warnings
cargo fmt --all -- --check
scripts/gen-header.sh --check          # headers match crates/turbo-abi + crates/turbo-capi
scripts/gen-versioned.py --check       # struct_size table matches crates/turbo-abi's struct field lists
scripts/c-smoke.sh                     # builds libturbo, compiles and runs the C smoke test
```

Run all of these before opening a PR; `.github/workflows/ci.yml`'s
`contract` job runs the same sequence plus a standalone C11/C++17 compile
of the header. Two further, independent CI jobs are not part of this
sequence: `java` (the Java binding's conformance cases under JDK 25,
`bindings/java`, `docs/bindings.md`) and `packaging`
(`scripts/package.sh --no-cuda --no-openvino`, `docs/packaging.md`).

If your change touches `crates/turbo-abi` or `crates/turbo-capi` (new
constant, struct, or function), regenerate the header and commit it:

```bash
scripts/gen-header.sh
```

If your change adds, removes, or reorders fields in an ABI struct
(anything starting with `struct_size` in `crates/turbo-abi/src/lib.rs` or
`provider.rs`), regenerate the accepted `struct_size` table and commit it:

```bash
scripts/gen-versioned.py
```

This writes `crates/turbo-abi/src/versioned.rs`, the per-struct list of
every `struct_size` the library accepts (`AGENTS.md`'s "ABI rules").

If your change touches `crates/turbo-core/src/mock.rs` in a way that
changes the mock bundle shape, regenerate the fixtures and commit them:

```bash
cargo run -p turbo-core --example write_mock_bundles
```

## Running the C smoke test manually

```bash
cargo build -p turbo-shared
cc -std=c11 -Wall -Wextra -Wpedantic -Werror -Iinclude \
    crates/turbo-conformance/c/smoke.c -Ltarget/debug -lturbo -lm \
    -Wl,-rpath,target/debug -o /tmp/turbo-c-smoke
/tmp/turbo-c-smoke testdata/bundles/mock
```

`turbo-shared` is the crate that produces `libturbo` (`cdylib` name
`turbo`); it links `turbo-capi`'s `extern "C"` exports. `scripts/c-smoke.sh`
does exactly this (release build with `--release`).

## Running the Rust conformance suite

```bash
cargo test -p turbo-conformance
```

By default this runs against the built-in providers and the committed mock
bundles (`testdata/bundles/mock`). To point it at a loadable provider
instead, set `TURBO_CONFORMANCE_PROVIDER_PATHS` (colon-separated library
paths to load in addition to the built-ins), `TURBO_CONFORMANCE_PROVIDER`
(the provider id to select), and `TURBO_CONFORMANCE_ORDINAL` (the device
ordinal within it); see `docs/testing.md`. The live provider tests
(`tests/live_embed.rs`, `tests/live_tasks.rs`, `tests/live_generate.rs`) are
a separate, narrower path that loads one real provider library and reads
its own `TURBO_LIVE_*` environment variables
(`crates/turbo-conformance/src/live.rs`), skipping themselves when
`TURBO_LIVE_LIB` or `TURBO_LIVE_PROVIDER` is unset; see
[`providers/openvino/README.md`](providers/openvino/README.md) for OpenVINO,
[`providers/cuda/README.md`](providers/cuda/README.md) for CUDA, and
[`providers/ggml/README.md`](providers/ggml/README.md) for `live_generate.rs`.

## Branch and PR expectations

- One logical change per PR. Split contract changes (header/ABI) from
  provider or documentation changes where practical.
- A PR that changes `crates/turbo-abi` or `crates/turbo-capi` includes the
  regenerated `include/turbo/*.h` in the same commit as the source change,
  not a follow-up.
- A PR that adds or changes behavior includes a test that would fail
  without it. See "Writing a test" below.
- Follow the commit rules in `AGENTS.md`: focused commits, no AI
  attribution lines, no product renames.
- Do not introduce a Cargo feature flag as a way to select hardware or a
  provider; providers are separate crates/libraries (`PLAN.md` section 1).

## Coding standards

- `#![deny(missing_docs)]` is set on every Rust crate in the workspace
  (`turbo-abi`, `turbo-core`, `turbo-capi`, `turbo-shared`, `turbo`,
  `turbo-conformance`, `providers/mock`, `providers/static`,
  `providers/cuda`, `providers/ggml`, `tools/turbo-bundle`). Every public
  item needs a doc comment.
- `clippy -D warnings` must be clean; do not add `#[allow(...)]` to silence
  a real finding. `turbo-abi` and `turbo-capi` also deny
  `unsafe_op_in_unsafe_fn`: every `unsafe` block, even inside an `unsafe fn`,
  is explicit.
- Every `unsafe` block needs a `// SAFETY:` comment stating the invariant
  that makes it sound (see `crates/turbo-core/src/buffer.rs` and
  `crates/turbo-capi/src/lib.rs` for the existing style: what the caller
  promised, what was checked already, why the operation is valid).
- No `.unwrap()` or `.expect()` on a fallible path outside tests. Use
  `unwrap_or_else(|p| p.into_inner())` for recovering a poisoned mutex (the
  pattern used throughout `turbo-core`), and return a typed `Error`
  otherwise. `.unwrap()` is fine in `#[cfg(test)]` modules and doctests.
- New ABI structs start with `struct_size: u32`; new ABI enumerations are
  `u32` constants with a `from_abi`/`as_abi` pair (see the `abi_enum!` macro
  in `crates/turbo-core/src/types.rs`), not a bare Rust `enum` exposed
  through FFI.
- Errors carry a graded status code (`crates/turbo-abi`'s `TURBO_E_*`
  constants) and, where applicable, the 1-based field index of the
  offending struct field (`Error::with_field`).

## Writing a test

Most behavior lives in `turbo-core` and is tested there with `#[cfg(test)]`
modules next to the code (see `handles.rs`, `runtime.rs`, `bundle.rs`,
`buffer.rs`, `mock.rs` for the existing patterns: build a mock runtime and
bundle with `crate::mock::write_mock_bundle`, exercise the handle API, and
assert on `Error::code()`). Prefer:

- A capability-honesty test: an option the mock does not advertise is
  rejected with `TURBO_E_UNSUPPORTED_OPTION` and the correct field index
  (`ungated_option_is_rejected_with_field` in `handles.rs` is the pattern).
- A lifetime test: children outlive released parents
  (`children_outlive_parents` in `handles.rs`).
- A device-policy test: `AUTO` never returns a CPU device
  (`auto_never_selects_cpu` in `runtime.rs`).

For a change that reaches the C ABI, add or extend a case in
`crates/turbo-conformance/c/smoke.c` using the `CHECK`/`EXPECT` macros
already there, and run `scripts/c-smoke.sh`.

For a change that is provider-agnostic contract behavior (something every
provider must do, not a `turbo-core`-internal detail), add a case to
`crates/turbo-conformance/tests/` instead: one file per group (contract,
lifetime, capability, device, threading, allocation, bundle, tasks,
generation), each with a `_rust` variant (through the safe `turbo` crate)
and a `_c` variant (through `turbo_capi`'s `extern "C"` functions directly,
using the helpers in `crates/turbo-conformance/src/lib.rs`'s `c` module).
Use `turbo_conformance::Target` to resolve the runtime, device, and bundle
under test so the same case runs against the mock and against real hardware
by only changing the environment (`docs/testing.md`).

## Review checklist

Derived from `PLAN.md` section 2; a reviewer should be able to answer yes to
each that applies to the change:

- [ ] Does this add a second way to express something the common header
      already expresses (a vendor-specific struct, a parallel API)? If so,
      extend the header instead.
- [ ] Does every new per-call option map to a `TURBO_CAP_*` bit, honored or
      rejected with `TURBO_E_UNSUPPORTED_OPTION` and a field index?
- [ ] Does any code path silently ignore, clamp, or substitute a value
      instead of failing or reporting an honest capability status?
- [ ] Does pooling/normalization/post-processing default to the host when it
      could run where the data already is, without being reported in
      `stage_placement`?
- [ ] Can `AUTO` device selection ever return a CPU? (It must not.)
- [ ] Does every new handle type that has a parent hold an `Arc`/reference to
      it, verified by a "child outlives released parent" test?
- [ ] Does anything infer per-model behavior (pooling, prefixes, labels)
      from a name or alias instead of the bundle contract?
- [ ] Is any capability marked in a way that overstates what has been
      measured (a `SUPPORTED` cell without a receipt)?
- [ ] Is `include/turbo/*.h` regenerated and committed if the ABI changed?
- [ ] Are `testdata/bundles/mock/**` regenerated and committed if the mock
      provider's bundle shape changed?
