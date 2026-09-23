# Working rules for TurboEmbed

Read this file, then section 0 of [`PLAN.md`](PLAN.md), before touching the
tree. Section 0 is the statement every other document is held against; if
anything below or elsewhere disagrees with it, section 0 wins and the
disagreement is fixed, not worked around.

## The mission

One native library, one interface, every path as fast as its hardware
allows. A program written once against `libturbo`, from C, Rust, Java,
Swift, Android or over gRPC, chunks, tokenizes, embeds, reranks, classifies
and generates the same way on an NVIDIA GPU, an Intel GPU or NPU, a Hailo
NPU, a Jetson or an Apple machine. The interface speaks in tasks, and a
provider runs the whole task on the vendor's own pipeline, at the lowest
layer that hardware has, with nothing in between: CUDA itself on NVIDIA,
not a framework runtime over it. A framework engine such as ONNX Runtime
is the last resort, taken only for a model the direct path does not
implement, and reported when taken. A stage runs where its
data already is; the hidden state never comes back to the host for a stage
the device can run, and a copy that is taken is reported. The model's
contract travels in a hash-verified bundle, so the answer is the same
everywhere. Selection is part of the interface: the caller names a task
and its constraints, and the library resolves the bundle and picks the
device and provider that will run it fastest here, and says why.
Speed is proven, never claimed: a matched benchmark against the fastest
known implementation on that hardware, committed as a receipt, is the only
thing that turns a capability cell green. Slow hardware is fine. A path
slower than that hardware's best is a bug. A feature one configuration has
and another lacks is fine, and the matrix says so.

Performance over elegance. ONNX Runtime is built for elegance: one graph
format, one session API, every backend behind it, and the copies and the
host round trips that buys. This library is a better ONNX: the same one
interface, as close to the metal as each hardware allows, and no layer
between the task and the kernels. Existing unifying libraries fail on one
of two sides: they lower every task to generic operations and pay for the
copies between them, or they keep the vendor pipeline and lose the common
interface. This library keeps the vendor pipeline and puts the common
interface above it. That is the whole point, and everything in the tree
either serves it or does not belong. When a choice is between a cleaner
abstraction and a faster path, the faster path wins and the abstraction
is made to fit it.

## What every change is checked against

A reviewer applies these to every change, not just provider code. Each is
a reject rule: a change that breaks one is sent back with the number.

1. **One contract.** No second header, no vendor-specific surface for a
   capability the common header can express. If a provider needs something
   the header cannot say, the header is extended for every provider.
2. **Honest capabilities.** Every option and placement maps to a
   `TURBO_CAP_*` bit. A provider either honors an option exactly or reports
   the bit clear and the call fails with `TURBO_E_UNSUPPORTED_OPTION` naming
   the field. A missing device capability that is not an option fails with
   the matching `TURBO_E_UNSUPPORTED*` code. Reject: any path that silently
   ignores, clamps, or substitutes a value for an option the caller set.
3. **Lowest layer per device.** The encoder runs on the vendor's compute
   layer (CUDA kernels, Metal, OpenVINO, HailoRT, ggml), and pooling,
   normalization, segment pooling and any other stage run where the
   hidden state was produced. A framework engine is a fallback that the
   placement and the cell name. Where a device cannot do a stage, the
   provider reports `fully_accelerated = 0` and names the stage in
   `stage_placement`. Reject: host pooling behind a fused-looking API; a
   device-to-host copy that is not reported; a framework engine presented
   as the direct path.
4. **Device policy.** `AUTO` never selects a CPU. CPU runs only when
   explicitly selected. An absent device is an error, never a silent
   fallback. The mock provider serves only mock bundles.
5. **Ownership is executable.** Child handles hold a reference to their
   parent; releasing a parent must not invalidate a live child. Reject: a
   handle relationship enforced only by a comment.
6. **Per-model truth lives in the bundle.** Pooling, normalization,
   sequence limit, prefixes, dimension, dtype and tokenizer identity come
   from `bundle.json`, never from a model-name heuristic or an alias table.
7. **Measured, not claimed.** A cell marked `SUPPORTED` names a
   conformance receipt, a precision receipt and a matched benchmark at or
   above 0.95 of the reference on a named architecture, from a clean
   commit. `EXPERIMENTAL` and `PLANNED` are honest states; use them.
   The reference is the pinned checkout under `/work/reference-code`
   listed in [`docs/reference-code.md`](docs/reference-code.md), and the
   receipt names its commit.
8. **No one-offs.** A consumer's need is met by extending the common
   surface, not by a special-cased function or flag for that consumer.
9. **No layers.** A change adds a task, a provider, a stage, a receipt, a
   binding projection, or a fix. A change that adds an abstraction none of
   those need is rejected, however tidy it looks.
10. **A capability is a cell.** A feature only one library or one device
    offers is welcome: it becomes a task or an option on the interface and
    a cell in the matrix, checked where it fits and `UNSUPPORTED` elsewhere.
    It does not become a side door.
11. **Tokens are the same everywhere.** The core tokenizer is the reference.
    A provider that fuses its own tokenizer replaces it only after an
    equivalence check against the core on the conformance corpus, recorded
    in a receipt.
12. **Provenance travels with the result.** Every result can say which
    provider, device architecture, runtime, artifact hash and tokenizer
    hash produced it, and where each stage ran.
13. **No Python in the tree.** Vendor Python toolchains run inside
    digest-pinned containers driven from Rust. The exceptions are the
    generators under `scripts/` that produce committed files, and demos.
14. **Public tree.** No private hostnames, addresses, user names or checkout
    paths in code, docs or receipts. Machines are named by architecture
    label (`rtx4080`, `b70`, `intel-npu`, `orin-nano`, `pi5-hailo8`,
    `pi5-hailo10h`, `m2`); `turbo-bench` takes `TURBO_BENCH_MACHINE` and
    writes `$HOME`.
15. **The README states; it does not sell.** Mission, one example, the
    matrix, the benchmark table. Screenshots live with the demo they show.

## Working with agents

Rules are by tier, not by vendor. Any model can fill any tier; the tier
says what the task is allowed to touch and what it must hand back.

- **Design and review.** The plan, the interface, the ranking rule, the
  receipt protocol, and reviews of the other tiers' work. Reads
  `PLAN.md` section 0 in full and cites the rule number for every finding.
- **Implementation.** One item from the roadmap, on one branch, with the
  receipt that proves it. Ends with a committed receipt from a clean tree
  or a stated failure with the output. Never both a claim and no receipt.
- **Remedial.** Renames, fixtures, generated files, doc sync, formatting,
  relabeling. Never touches the interface, a provider's hot path, or a
  receipt's numbers.

For every tier: one task per agent; read this file and section 0 first;
build and test before reporting; a benchmark receipt names the commit that
ran and the reference commit it was compared with; nothing generated is
edited by hand; commits carry no attribution lines; if the task cannot be
finished, say what was done, what was not, and why, and stop.

## Where things live

- `crates/turbo-abi` — `#[repr(C)]` types and constants, including the
  provider vtable types (`provider.rs`); the single source cbindgen reads to
  produce the headers. `no_std`.
- `crates/turbo-capi` — `extern "C"` exports; thin, panic-safe adapters over
  `turbo-core`.
- `crates/turbo-shared` — links `turbo-capi` into `libturbo` (`cdylib` +
  `staticlib`, crate name `turbo`). Nothing is defined here.
- `crates/turbo-core` — the runtime, device registry, buffers, bundle
  loader and verifier, tokenizers (`tokenizer.rs`), the chunk planner
  (`chunker.rs`), handles (ownership, lifetime, lease enforcement), the
  provider trait set (`provider.rs`), provider plugin loading (`plugin.rs`)
  and exporting (`plugin_export.rs`, the `export_provider!` macro), and
  the `mock` provider.
- `crates/turbo` — the safe Rust API; re-exports `turbo-core` and adds
  `builtin_providers()` (`mock`, `static`).
- `crates/turbo-conformance` — the provider-agnostic contract suite:
  `c/smoke.c` and a Rust suite (`tests/`, one file per group, each in a
  `_c` and a `_rust` variant, plus `header_parity.rs` and the live provider
  tests).
- `crates/turbo-bench` — the workload runner that writes receipts, and
  `compare`, which reads a libturbo receipt and a native receipt and writes
  the verdict.
- `reference/` — the direct-native half of each matched benchmark: the
  vendor API alone, with no libturbo in the timed path.
- `providers/mock/`, `providers/static/`, `providers/cuda/`,
  `providers/ggml/` — Rust providers built with `export_provider!`.
  `providers/openvino/` (C++), `providers/metal/` (Objective-C++) and
  `providers/hailo/` (C++) implement `turbo_provider.h`'s vtable directly
  and build with CMake; see each one's `README.md`.
- `bindings/java` — the JDK 25 FFM binding; `ai/pipestream/turbo/ffi` is
  generated by `scripts/gen-java-ffi.sh` from the header, not edited by
  hand. `bindings/swift` — the SwiftPM package over the C ABI.
- `tools/turbo-bundle/` — `import`, `verify`, `inspect` (`docs/bundles.md`).
- `include/turbo/` — generated headers (`turbo.h`, `turbo_types.h`,
  `turbo_provider.h`). Never hand-edit.
- `crates/turbo-core/src/unicode_data.rs` — generated by
  `scripts/gen-unicode-nfd.py`. Do not hand-edit.
- `testdata/bundles/mock/` — generated mock bundle fixtures.
  `testdata/bundles/minilm-tokenizer/` — an imported tokenizer-only fixture;
  edit it only by re-running the importer, never by hand.
- `testdata/receipts/turbo/` — conformance and precision receipts;
  `testdata/receipts/turbo/bench/` — benchmark receipts, native receipts
  and the compare verdicts, named by architecture label.
- `docs/` — current documentation; `docs/history/` holds retired
  documentation, unedited; `docs/reviews/` holds reviews as delivered.

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

`.github/workflows/ci.yml`'s `contract` job runs exactly this sequence,
plus a check that `include/turbo/turbo.h` compiles standalone as both C11
and C++17. Run all of it locally before opening a PR; a change to
`turbo-abi` or `turbo-capi` almost always requires regenerating headers
and, if it touches the mock provider's manifests, regenerating fixtures.
CI does not build the OpenVINO, CUDA, Metal or Hailo providers (hosted
runners have no toolkit), so it runs no provider's live tests; those load
one real provider library and skip themselves when `TURBO_LIVE_LIB` or
`TURBO_LIVE_PROVIDER` is unset. Two further CI jobs run independently:
`java` (the Java binding's conformance cases under JDK 25 against the mock
provider) and `packaging` (`scripts/package.sh --no-cuda --no-openvino`).
See `docs/testing.md` for the live tests, and `docs/bindings.md` and
`docs/packaging.md` for the other two jobs.

## ABI rules

- Every public struct starts with `uint32_t struct_size`. The library reads
  only fields below the caller's declared size, and accepts a size only when
  it is the end of a field the struct has ever had; any other size is
  `TURBO_E_INVALID_STRUCT_SIZE`. The accepted sizes are a per-struct table
  (`crates/turbo-abi/src/versioned.rs`, the `Versioned` trait), generated
  by `scripts/gen-versioned.py`; regenerate it when a struct's fields
  change, and `scripts/gen-versioned.py --check` fails CI if it drifts.
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
  own capability bit and an honest `UNSUPPORTED` or `PLANNED` status.

## Generated files

Never hand-edit `include/turbo/turbo.h`, `include/turbo/turbo_types.h`,
`include/turbo/turbo_provider.h`, `crates/turbo-abi/src/versioned.rs`,
`crates/turbo-core/src/unicode_data.rs`, the Java `ffi` package, or
`testdata/bundles/mock/*`. Change the source (`crates/turbo-abi`,
`crates/turbo-capi`, `crates/turbo-core/src/mock.rs`) and run the
generator (`scripts/gen-header.sh`, `scripts/gen-versioned.py`,
`scripts/gen-java-ffi.sh`, `cargo run -p turbo-core --example
write_mock_bundles`). Commit the regenerated output alongside the source
change in the same PR; CI diffs each against a fresh run.

## Adding a provider

1. A Rust provider implements the traits in `crates/turbo-core/src/
   provider.rs` and exports them with `turbo_core::export_provider!` (see
   `providers/mock`, `providers/static`). A provider in another language
   implements `include/turbo/turbo_provider.h`'s vtable directly and
   exports `turbo_provider_get` (see `providers/openvino`). Either way the
   library is loaded with `turbo_runtime_load_provider` or
   `turbo_runtime_desc.provider_paths`; see `docs/providers.md`.
2. Report an honest matrix from `Provider::capability`: `SUPPORTED` only
   with the three receipts named in rule 7; `EXPERIMENTAL` or `PLANNED`
   otherwise, with the reason. Set `fully_accelerated` and
   `stage_placement` to what the provider did, not what it intends.
3. Run the conformance suite against the new provider: set
   `TURBO_CONFORMANCE_PROVIDER_PATHS` to the library path (and
   `TURBO_CONFORMANCE_PROVIDER` and `TURBO_CONFORMANCE_ORDINAL` to select
   its device) and run `cargo test -p turbo-conformance`; see
   `docs/testing.md`.
4. Write the direct-native reference program under `reference/` and the
   matched benchmark pair with `turbo-bench`, from a clean commit, and
   commit the three receipts under `testdata/receipts/turbo/bench/` with
   `TURBO_BENCH_MACHINE` set to the architecture label.

## Documentation rules

- Separate current behavior from plans. State what the code does today in
  plain sentences; mark anything not yet implemented as planned, naming the
  roadmap item.
- Name the architecture label and the commit for any measurement. A number
  without both is noise, not a claim.
- State a fact once, in the file that owns it, and link to it elsewhere.
- No slogans, marketing adjectives, or emoji. Plain sentences, no
  em-dashes, repository-relative links, commands that work from the
  repository root.

## Commit rules

- Focused commits: one logical change per commit.
- No AI attribution lines in commit messages or code comments.
- No product renames. The library is `turbo` / `libturbo`; the project and
  repository stay `TurboEmbed`; the server stays `Inferstream`. Do not
  introduce another name for any of these.
