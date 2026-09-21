# Testing

## Today

Testing in this tree today is:

- `#[cfg(test)]` unit test modules inside `crates/turbo-core/src/{handles,
  runtime,bundle,buffer,provider,mock,tokenizer,chunker,plugin}.rs` and
  inside the provider crates (`providers/mock`, `providers/static`,
  `providers/cuda`), run by `cargo test --locked --workspace --exclude
  turbo-provider-cuda` (`providers/cuda`'s own unit tests need the CUDA
  toolkit to build; see "Running what exists today" below).
- `crates/turbo-conformance/c/smoke.c`, a C program compiled against the
  installed header and run against the mock provider through
  `scripts/c-smoke.sh`. It exercises runtime/device/context/model/session
  lifecycle, capability queries, option rejection with field index, result
  leasing, and generation through the pull iterator.
- The provider-agnostic Rust conformance suite under
  `crates/turbo-conformance/tests/`, run automatically as part of `cargo
  test --locked --workspace --exclude turbo-provider-cuda` (each file is a
  normal Rust integration test binary). This is the suite `PLAN.md`
  section 10 describes; it now exists.
- Header parity (`scripts/gen-header.sh --check`), the `struct_size` table
  (`scripts/gen-versioned.py --check`), and mock fixture parity (`cargo run
  -p turbo-core --example write_mock_bundles` diffed against
  `testdata/bundles/mock/`), all run in CI.

`docs/reviews/2026-09-21-p0-p2.md` is an independent review of the P0-P2
tree that tracks each finding to closure; items closed by a commit carry
the test that would have failed before the fix (for example
`contract_struct_size_inside_a_field_is_rejected`, closing finding C3).

## The conformance suite

`crates/turbo-conformance/src/lib.rs` is the harness: it resolves a target
(runtime, device, bundle root) from the environment and provides the mock
bundle fixtures as defaults. The cases live under `tests/`, over 200 tests
across 24 files; most groups appear twice, once through the safe Rust API
(`*_rust.rs`) and once through the C ABI called directly (`*_c.rs`), plus
`header_parity.rs`, the two live-hardware files, and two Rust-only files
closing specific review findings:

| file | group |
|---|---|
| `contract_rust.rs`, `contract_c.rs`, `contract_c_errors.rs` | empty text, embedded NUL, invalid UTF-8, unknown `struct_size`, unknown option/enum constants, oversized shapes |
| `lifetime_rust.rs`, `lifetime_c.rs` | release order in every permutation, results outliving sessions and models, two contexts open at once |
| `capability_rust.rs`, `capability_c.rs` | every `TURBO_CAP_OPT_*` bit passes its honor test or its rejection test (`TURBO_E_UNSUPPORTED_OPTION` with the right field index) |
| `honesty_rust.rs` | every option field maps to a bit or an unconditional check, with no ungated third case: generation sampling/min-tokens/echo/stop-tokens/raw-scores bits, `max_tokens` over the session width, and a `prompt_role` with no bundle prefix (`docs/reviews/2026-09-21-p0-p2.md` Medium items and H7) |
| `device_rust.rs`, `device_c.rs` | `AUTO` never returns a CPU, explicit selection works, an absent device selector fails |
| `threading_rust.rs`, `threading_c.rs` | two sessions run concurrently without interference; an overlapping call on one session returns `TURBO_E_BUSY` |
| `allocation_rust.rs`, `allocation_c.rs` | `allocs_per_run == 0` after warmup on a provider's own counter, for capability cells that claim it |
| `bundle_rust.rs`, `bundle_c.rs` | bundle hash/shape/path-escape integrity, per `docs/bundles.md` |
| `bundle_integrity_rust.rs` | manifests reject unknown fields and unparseable enumerated values, scored kinds require `contract.activation`/`aggregation`, a symlink resolving outside the bundle directory is refused, `manifest_sha256` identifies the manifest (`docs/reviews/2026-09-21-p0-p2.md` H5 and a Medium item) |
| `tasks_rust.rs`, `tasks_c.rs` | embed/rerank/classify/token-classify/tokenize/chunk task behavior against the mock provider's bundle kinds |
| `generation_rust.rs`, `generation_c.rs` | pull-iterator generation: prompt, step, cancel, finish reasons |
| `header_parity.rs` | the committed headers match a fresh `cbindgen` run |
| `live_embed.rs`, `live_tasks.rs` | live checks against a real provider library (`openvino` or `cuda`) and bundles; see "Live provider tests" below |

A provider counts as supported for a capability only when every applicable
group passes, or is explicitly excluded by a capability bit the suite
verified is reported correctly (`PLAN.md` sections 10-11).

Environment variables the harness reads (`crates/turbo-conformance/src/lib.rs`):

- `TURBO_CONFORMANCE_PROVIDER` — provider id to select explicitly, instead
  of `AUTO` over the built-in providers (default: unset).
- `TURBO_CONFORMANCE_ORDINAL` — device ordinal within that provider
  (default 0, only used with an explicit provider).
- `TURBO_CONFORMANCE_BUNDLES` — directory holding one subdirectory per
  bundle kind, in place of the committed `testdata/bundles/mock`.
- `TURBO_CONFORMANCE_PROVIDER_PATHS` — colon-separated provider library
  paths loaded in addition to the built-in providers before the suite runs
  (for example, point it at `libturbo_provider_openvino.so` or
  `libturbo_provider_cuda.so` to run the suite's mock-bundle-independent
  groups against that provider).

`PLAN.md` section 11 additionally defines the benchmark protocol: a matched
pair per provider (a direct-native reference program using the runtime
alone, and the same workload through `libturbo`), workloads at batch
{1, 8, 32} by sequence {32, 128, 256} for embeddings, 32 documents for
rerank, 128 new tokens for generation. Reported: p50/p99 latency, tokens/s,
H2D/D2H bytes, host allocations per run, device memory. Receipts carry
machine ID, runtime/driver versions, bundle hashes, and commit, and budgets
are set from the first run per provider and then held. No benchmark receipt
exists in this tree yet (`crates/turbo-bench/` is not written); this is why
`static`, `openvino`, and `cuda` are `EXPERIMENTAL` rather than `SUPPORTED`.

## Live provider tests

`crates/turbo-conformance/tests/live_embed.rs` and `live_tasks.rs` load one
real provider library against real bundles; they are provider-agnostic
(any provider that offers the task), unlike the rest of the suite which
runs against the mock bundles by default. They replace the earlier
`openvino_live.rs`/`openvino_live_tasks.rs` (provider-specific, OpenVINO
only). Each test prints a reason and returns (not a failure) when its
required environment variables are unset, so they run safely as part of
`cargo test --workspace` in an environment with no provider configured.
They are not run in CI: `.github/workflows/ci.yml` has no OpenVINO install
and excludes `turbo-provider-cuda` (hosted runners have no CUDA toolkit).

Provider selection (`crates/turbo-conformance/src/live.rs`), read by both
files:

- `TURBO_LIVE_LIB`: path to the provider library
  (`libturbo_provider_openvino.so`, `libturbo_provider_cuda.so`).
- `TURBO_LIVE_PROVIDER`: the provider id it registers (`openvino`, `cuda`).
- `TURBO_LIVE_ORDINAL` (optional): device ordinal; default is the
  provider's CPU device if it has one, else ordinal 0.
- `TURBO_REFERENCE_DIR` (optional): directory holding the reference
  vectors (`testdata/reference_embeddings/ort_cuda_minilm_*.json`) so a
  copied test binary can find them on another machine; defaults to that
  path relative to the crate.

`live_embed.rs` (embed, against a MiniLM bundle), additionally gated on
`TURBO_LIVE_BUNDLE` (a bundle with an `onnx` or `openvino_ir` artifact for
`sentence-transformers/all-MiniLM-L6-v2`, built with `tools/turbo-bundle`).
It checks the embedding vectors against ONNX Runtime CUDA FP32 reference
vectors (cosine > 0.9995), that batch rows equal single-text runs, that
truncation policies (`NONE` rejected over budget, `LEFT` vs `RIGHT` produce
different results) and the pooling override (honored or rejected, following
the capability bit) are handled correctly, that GPU results stay
`Placement::Device` with `h2d_bytes > 0` and `d2h_bytes == 0` until
`result_read`, while CPU results are `Placement::Host` with no H2D traffic,
and that a GPU result exports the provider's native handle
(`TURBO_HANDLE_CUDA_PTR` for `cuda`, `TURBO_HANDLE_CL_MEM` for `openvino`).

`live_tasks.rs` (rerank, classify, token-classify), gated on one bundle
variable per task:

- `TURBO_LIVE_RERANK_BUNDLE`: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- `TURBO_LIVE_CLASSIFY_BUNDLE`: a sequence-classification bundle
  (`distilbert/distilbert-base-uncased-finetuned-sst-2-english` in both
  committed receipts).
- `TURBO_LIVE_NER_BUNDLE`: a token-classification bundle
  (`dslim/bert-base-NER` in both committed receipts).

Each test that has its bundle variable set runs a semantic check specific to
its task (ranking order and `top_n`/`raw_scores` handling for rerank, label
names for classify, aggregated entity spans for NER); a test whose bundle
variable is unset returns without failing. These assert semantic properties,
not exact numbers, so the same test file holds across FP32 devices and
providers.

Receipts from real runs of both files are committed under
`testdata/receipts/turbo/`: `openvino-minilm-2026-09-21.json` and
`openvino-tasks-2026-09-21.json` for the OpenVINO provider, and
`cuda-2026-09-21.json` for the CUDA provider (`krick`, RTX 4080 SUPER,
cosine 1.000 against the FP32 references); see `docs/providers.md` for what
they record.

## Running what exists today

```bash
cargo test --locked --workspace --exclude turbo-provider-cuda   # unit tests plus the Rust conformance suite
scripts/gen-header.sh --check     # include/turbo/*.h matches turbo-abi/turbo-capi
scripts/gen-versioned.py --check  # crates/turbo-abi/src/versioned.rs matches its struct field lists
scripts/c-smoke.sh                # builds libturbo, runs smoke.c against testdata/bundles/mock
cargo run -p turbo-core --example write_mock_bundles \
  && git diff --exit-code -- testdata/bundles   # fixtures match mock.rs
```

`.github/workflows/ci.yml` runs all five, plus `cargo fmt --check`, `cargo
clippy -D warnings` (also excluding `turbo-provider-cuda`), and a standalone
C11/C++17 compile of the header. It excludes `turbo-provider-cuda` because
hosted runners have no CUDA toolkit, and does not build the OpenVINO
provider at all; run either provider's live tests by hand per
`providers/openvino/README.md` or `providers/cuda/README.md`.

## Fixtures

`testdata/bundles/mock/{embedding,reranker,classifier,token-classifier,
generative,generic}/` are generated, not hand-written; see
`docs/bundles.md`'s "Mock bundle layout" section and
`crates/turbo-core/examples/write_mock_bundles.rs`. Regenerate and commit
them whenever `crates/turbo-core/src/mock.rs` changes their shape.

`testdata/bundles/minilm-tokenizer/` is a real (non-mock), tokenizer-only
bundle used by the tokenizer, chunker, and importer tests; see
`docs/bundles.md`'s "Importer" section.
