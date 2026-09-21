# Agent and contributor rules for Turbo

Read [`PLAN.md`](PLAN.md) before touching this tree. It is the governing
plan: architecture, naming, principles, and milestone scope. This file gives
the working rules that follow from it. If this file and `PLAN.md` conflict,
`PLAN.md` wins and the conflict should be fixed here, not worked around.

## Non-negotiable principles (PLAN.md section 2), as review checks

A reviewer applies these to every change, not just provider code:

1. **One contract.** No second header, no vendor-specific API surface for a
   capability the common header can express. If a provider needs something
   the header cannot say, the header is extended for every provider, not
   forked.
2. **Honest capabilities.** Every option and placement maps to a
   `TURBO_CAP_*` bit. A provider either honors an option exactly or reports
   the bit clear and the call fails with `TURBO_E_UNSUPPORTED_OPTION` naming
   the field. Reject: any code path that silently ignores, clamps, or
   substitutes a value for an option the caller set.
3. **Lowest layer per device.** Pooling and normalization run where the
   hidden state was produced, not pulled to the host by default. Where a
   device cannot do a stage, the provider reports `fully_accelerated = 0`
   and names the stage in `stage_placement`. Reject: unconditional host
   pooling behind a fused-looking API.
4. **Device policy.** `AUTO` never selects a CPU. CPU runs only when
   explicitly selected. An absent device is an error, never a silent
   fallback to something else. The mock provider serves only mock bundles.
5. **Ownership is executable.** Child handles hold an `Arc`/reference to
   their parent; releasing a parent must not invalidate a live child.
   Reject: any handle relationship enforced only by a comment or a doc
   string instead of the type.
6. **Per-model truth lives in the bundle.** Pooling, normalization, sequence
   limit, prefixes, dimension, dtype, and tokenizer identity come from
   `bundle.json`, not from a model-name heuristic or an alias table.
7. **Measured, not claimed.** A capability cell marked `SUPPORTED` needs a
   conformance receipt, a precision receipt, and a matched-native benchmark
   from a named machine. `EXPERIMENTAL` and `PLANNED` are honest,
   non-blocking states — use them instead of overclaiming.
8. **No one-offs.** A specific consumer's need is met by extending the
   common surface, not by adding a special-cased function or flag for that
   consumer alone.

## Where things live

- `crates/turbo-abi` — `#[repr(C)]` types and constants, including the
  provider vtable types (`provider.rs`); the single source cbindgen reads to
  produce the headers. `no_std`.
- `crates/turbo-capi` — `extern "C"` exports; thin, panic-safe adapters over
  `turbo-core`.
- `crates/turbo-shared` — links `turbo-capi` into `libturbo` (`cdylib` +
  `staticlib`, crate name `turbo`). Nothing is defined here.
- `crates/turbo-core` — the runtime, device registry, buffers, bundle
  loader/verifier, tokenizers (`tokenizer.rs`), the chunk planner
  (`chunker.rs`), handles (ownership/lifetime/lease enforcement), the
  provider trait set (`provider.rs`), provider plugin loading
  (`plugin.rs`, the C vtable to Rust trait adapter) and exporting
  (`plugin_export.rs`, the `export_provider!` macro), and the `mock`
  provider.
- `crates/turbo` — the safe Rust API; re-exports `turbo-core` and adds
  `builtin_providers()` (`mock`, `static`).
- `crates/turbo-conformance` — the provider-agnostic contract suite:
  `c/smoke.c` and a Rust suite (`tests/`, one file per group: contract,
  lifetime, capability, device, threading, allocation, bundle, tasks,
  generation, each in a `_c` and a `_rust` variant, plus `header_parity.rs`
  and the OpenVINO live tests described below).
- `providers/mock/`, `providers/static/` — Rust providers built with
  `export_provider!`. `providers/openvino/` — a C++ provider that
  implements `turbo_provider.h`'s vtable directly and builds separately
  with CMake; see its own `README.md`.
- `tools/turbo-bundle/` — `import`, `verify`, `inspect` (`docs/bundles.md`).
- `include/turbo/` — generated headers (`turbo.h`, `turbo_types.h`,
  `turbo_provider.h`). Never hand-edit.
- `testdata/bundles/mock/` — generated mock bundle fixtures.
  `testdata/bundles/minilm-tokenizer/` — an imported tokenizer-only fixture;
  edit it only by re-running the importer or the fixture writer, never by
  hand.
- `testdata/receipts/turbo/` — conformance and precision receipts for this
  tree's providers (OpenVINO today).
- `docs/` — current documentation; `docs/history/poc/` holds retired PoC
  documentation, unedited.

## Validation commands and what CI runs

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --exclude turbo-provider-cuda --all-targets -- -D warnings
cargo test --locked --workspace --exclude turbo-provider-cuda
scripts/gen-header.sh --check
scripts/gen-versioned.py --check
cargo run -p turbo-core --example write_mock_bundles && git diff --exit-code -- testdata/bundles
scripts/c-smoke.sh
```

`.github/workflows/ci.yml` runs exactly this sequence, plus a check that
`include/turbo/turbo.h` compiles standalone as both C11 and C++17. Run all of
it locally before opening a PR; a change to `turbo-abi` or `turbo-capi`
almost always requires regenerating headers and, if it touches the mock
provider's manifests, regenerating fixtures. CI does not build the OpenVINO
provider (needs an OpenVINO install) and excludes `turbo-provider-cuda`
(hosted runners have no CUDA toolkit), so it does not run either provider's
live tests (`crates/turbo-conformance/tests/live_embed.rs`, `live_tasks.rs`);
those load one real provider library and skip themselves when
`TURBO_LIVE_LIB` or `TURBO_LIVE_PROVIDER` is unset. See `docs/testing.md`
for running them by hand.

## ABI rules

- Every public struct starts with `uint32_t struct_size`. The library reads
  only fields below the caller's declared size, and accepts a size only when
  it is the end of a field the struct has ever had (every accepted prefix is
  a layout that could have shipped); a size that ends inside a field, or any
  other size the struct has never had, is `TURBO_E_INVALID_STRUCT_SIZE`. The
  accepted sizes are a per-struct table (`crates/turbo-abi/src/versioned.rs`,
  the `Versioned` trait), generated from the struct field lists by
  `scripts/gen-versioned.py`; regenerate it when a struct's fields change,
  and `scripts/gen-versioned.py --check` fails CI if it drifts.
- ABI-position enumerations are `uint32_t` named constants, never a C
  `enum`. An unrecognized value is `TURBO_E_INVALID_ENUM`, never mapped to a
  default.
- Errors are caller-owned (`turbo_error`, may be `NULL`). No thread-local or
  shared error state anywhere in the library.
- Handles are opaque and reference counted. A child handle retains its
  parent; a release function accepts `NULL` and is a no-op.
- Sessions and generations are single-owner: a concurrent call on the same
  handle returns `TURBO_E_BUSY` rather than racing.
- Every capability bit either gates an option (the provider honors it) or
  the corresponding call rejects the option with `TURBO_E_UNSUPPORTED_OPTION`
  and the 1-based field index. There is no third behavior.
- No silent fallbacks, no swallowed errors, fail loudly. If you are tempted
  to catch an error and substitute a default, that substitution needs its
  own capability bit and an honest `UNSUPPORTED`/`PLANNED` status instead.

## The header is generated

Never hand-edit `include/turbo/turbo.h`, `include/turbo/turbo_types.h`, or
`include/turbo/turbo_provider.h`. Change `crates/turbo-abi` (constants and
`#[repr(C)]` types, including `crates/turbo-abi/src/provider.rs` for the
plugin vtable) or `crates/turbo-capi` (function signatures and doc
comments), then run:

```bash
scripts/gen-header.sh
```

Commit the regenerated headers alongside the Rust change in the same PR.
`scripts/gen-header.sh --check` (what CI runs) fails the build if they drift.

## Mock fixtures are generated

`testdata/bundles/mock/*/bundle.json` and `mock.json` are written by:

```bash
cargo run -p turbo-core --example write_mock_bundles
```

Do not hand-edit them. If you change `crates/turbo-core/src/mock.rs`
(bundle shape, contract fields, salt/vocab constants), regenerate and commit
the fixtures in the same PR; CI diffs the working tree against a fresh run.

## Adding a provider

1. A Rust provider implements the traits in `crates/turbo-core/src/
   provider.rs` (`Provider`, `ProviderContext`, `ProviderModel`,
   `ProviderSession`, `ProviderGeneration` as applicable) and exports them
   with `turbo_core::export_provider!` (see `providers/mock`,
   `providers/static`). A provider in another language implements
   `include/turbo/turbo_provider.h`'s vtable directly and exports
   `turbo_provider_get` (see `providers/openvino`, C++). Either way the
   library is loaded with `turbo_runtime_load_provider` or
   `turbo_runtime_desc.provider_paths`; see `docs/providers.md`.
2. Report an honest capability matrix from `Provider::capability`
   (`(*capability)` in the C vtable): mark a cell `SUPPORTED` only once it
   has a conformance receipt, a precision receipt, and a benchmark; use
   `EXPERIMENTAL` or `PLANNED` otherwise. Set `fully_accelerated` and
   `stage_placement` in `ModelInfo` to what the provider actually did, not
   what it intends to do.
3. Run the conformance suite (`crates/turbo-conformance`) against the new
   provider: set `TURBO_CONFORMANCE_PROVIDER_PATHS` to the provider's
   library path (and `TURBO_CONFORMANCE_PROVIDER`/`TURBO_CONFORMANCE_ORDINAL`
   to select its device) and run `cargo test -p turbo-conformance`; see
   `docs/testing.md`. A provider with no real bundle to test against yet
   should at least carry the handle-lifetime, capability-honesty, and
   device-policy unit tests the `mock` provider and `turbo-core` carry (see
   `handles.rs`, `runtime.rs`, `mock.rs` test modules for the pattern).
4. Ship receipts under `testdata/receipts/turbo/` per `PLAN.md` sections
   10-11: machine ID, runtime/driver versions, bundle hashes, and commit
   (see `testdata/receipts/turbo/openvino-minilm-2026-09-21.json` for the
   shape).

## Documentation rules

- Separate current behavior from plans. State what the code does today in
  plain sentences; mark anything not yet implemented as planned, naming the
  milestone (`PLAN.md` section 10).
- Name the machine for any measurement (latency, throughput, memory). A
  number without a named machine and commit is not a claim, it is noise.
- No repeated status prose. State a fact once, in the file that owns it, and
  link to it elsewhere instead of restating it.
- No slogans, marketing adjectives, or emoji. Plain sentences, no
  em-dashes, repository-relative links, commands that work from the
  repository root.

## Commit rules

- Focused commits: one logical change per commit, not a batch of unrelated
  fixes.
- No AI attribution lines in commit messages or code comments.
- No product renames. The library is `turbo` / `libturbo`; the project and
  repository stay `TurboEmbed`; the server stays `Inferstream`
  (`PLAN.md` section 3). Do not introduce another name for any of these.
