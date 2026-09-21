# Testing

## Today

Testing in this tree today is:

- `#[cfg(test)]` unit test modules inside `crates/turbo-core/src/{handles,
  runtime,bundle,buffer,provider,mock}.rs`, run by `cargo test --workspace`.
  These already cover pieces of what `PLAN.md` section 10 calls out for the
  P0 conformance suite: capability-honesty (`ungated_option_is_rejected_with_field`),
  lifetime (`children_outlive_parents`, `result_lease_blocks_writes_and_runs`),
  device policy (`auto_never_selects_cpu`, `explicit_needs_provider_and_matches_ordinal`),
  and bundle integrity (`hash_mismatch_is_integrity_error`, `rejects_path_escape`,
  `wrong_version_rejected`). They are unit tests against `turbo-core`
  directly, not a provider-agnostic suite driven through the C ABI.
- `crates/turbo-conformance/c/smoke.c`, a C program compiled against the
  installed header and run against the mock provider through
  `scripts/c-smoke.sh`. It exercises runtime/device/context/model/session
  lifecycle, capability queries, option rejection with field index, result
  leasing, and generation through the pull iterator (see the file itself
  for the exact sequence).
- Header parity (`scripts/gen-header.sh --check`) and mock fixture parity
  (`cargo run -p turbo-core --example write_mock_bundles` diffed against
  `testdata/bundles/mock/`), both run in CI.

`crates/turbo-conformance` exists as a workspace crate (`turbo-conformance`,
depending on `turbo`, `turbo-capi`, `turbo-abi`) with an empty `tests/`
directory. The provider-agnostic Rust conformance suite that `PLAN.md`
describes below has not been written yet in this tree.

## The conformance suite (PLAN.md sections 10 and 11)

Once written, the suite is one set of test groups, parameterized by provider
and device, run both through the C ABI (a C test binary) and through each
binding. A provider counts as supported for a capability only when every
applicable group passes, or is explicitly excluded by a capability bit the
suite verified is reported correctly. The P0 groups (`PLAN.md` section 10):

- **Contract**: empty text, embedded NUL, invalid UTF-8, unknown
  `struct_size`, unknown option constants, oversized shapes are all
  rejected with the right status code.
- **Lifetime**: release order in every permutation, results outliving
  sessions and models, two contexts open at once.
- **Capability honesty**: every `TURBO_CAP_OPT_*` bit either passes its
  honor test (the option takes effect) or its rejection test (the option is
  refused with `TURBO_E_UNSUPPORTED_OPTION` and the right field index).
- **Device policy**: `AUTO` never returns a CPU, explicit CPU selection
  works, an absent device selector fails rather than substituting one.
- **Threading**: two sessions run concurrently without interference; an
  overlapping call on one session returns `TURBO_E_BUSY`; reentrant
  callbacks are rejected.
- **Allocation**: `allocs_per_run == 0` after warmup on a provider's own
  counter, for every capability cell that claims it, with provider-internal
  allocations reported separately.

`PLAN.md` section 11 additionally defines the benchmark protocol that runs
once a provider exists: a matched pair per provider (a direct-native
reference program using the runtime alone, and the same workload through
`libturbo`), workloads at batch {1, 8, 32} by sequence {32, 128, 256} for
embeddings, 32 documents for rerank, 128 new tokens for generation.
Reported: p50/p99 latency, tokens/s, H2D/D2H bytes, host allocations per
run, device memory. Receipts carry machine ID, runtime/driver versions,
bundle hashes, and commit, and budgets are set from the first run per
provider and then held.

## Running what exists today

```bash
cargo test --workspace          # unit tests, including the mock provider's
scripts/gen-header.sh --check   # include/turbo/*.h matches turbo-abi/turbo-capi
scripts/c-smoke.sh              # builds libturbo, runs smoke.c against testdata/bundles/mock
cargo run -p turbo-core --example write_mock_bundles \
  && git diff --exit-code -- testdata/bundles   # fixtures match mock.rs
```

`.github/workflows/ci.yml` runs all four, plus `cargo fmt --check`,
`cargo clippy -D warnings`, and a standalone C11/C++17 compile of the
header.

## Fixtures

`testdata/bundles/mock/{embedding,reranker,classifier,token-classifier,
generative,generic}/` are generated, not hand-written; see
`docs/bundles.md`'s "Mock bundle layout" section and
`crates/turbo-core/examples/write_mock_bundles.rs`. Regenerate and commit
them whenever `crates/turbo-core/src/mock.rs` changes their shape.

## Pointing a run at a real provider (planned)

Once a hardware provider exists (P2 onward) and the conformance suite is
written, a run against real hardware instead of the mock is expected to be
selected through environment variables the suite reads:

- `TURBO_CONFORMANCE_PROVIDER` — the provider id to load (`openvino`,
  `cuda`, `metal`, `hailo`, `ggml`), instead of the built-in `mock`.
- `TURBO_CONFORMANCE_ORDINAL` — the device ordinal within that provider to
  run against.
- `TURBO_CONFORMANCE_BUNDLES` — a directory of real (non-mock) bundles to
  load for the suite's goldens and capability tests, in place of
  `testdata/bundles/mock`.

These variables are not read by any code in this tree yet — they describe
the interface the conformance suite is expected to expose once it is
written, so that a hardware run is "point the suite at a provider and a
bundle directory" rather than a separate test binary per provider. Treat
this section as the target shape, not as documentation of existing behavior.
