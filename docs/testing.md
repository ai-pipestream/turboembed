# Testing

## Today

Testing in this tree today is:

- `#[cfg(test)]` unit test modules inside `crates/turbo-core/src/{handles,
  runtime,bundle,buffer,provider,mock,tokenizer,chunker,plugin}.rs` and
  inside the provider crates (`providers/mock`, `providers/static`,
  `providers/cuda`, `providers/ggml`), run by `cargo test --locked
  --workspace --exclude turbo-provider-cuda` (`providers/cuda`'s own unit
  tests need the CUDA toolkit to build, so it is the one crate excluded;
  `providers/ggml` is a plain workspace member and builds and runs its unit
  tests as part of this same command, CPU backend only — see "Running what
  exists today" below). `providers/mock/tests/plugin_load.rs`, an
  integration test that loads the mock provider's cdylib through the plugin
  ABI, looks first for the plain name under `target/<profile>/deps`; failing
  that, it takes whichever is newest by modification time out of the plain
  name in the profile directory and any hash-suffixed match under `deps/`.
  `cargo test` builds the cdylib into `deps/` (a plain name on macOS,
  hash-suffixed on Linux) and does not always uplift it to the profile
  directory, and an already-uplifted copy can be stale from an earlier
  `cargo build`.
- `crates/turbo-conformance/c/smoke.c`, a C program compiled against the
  installed header and run against the mock provider through
  `scripts/c-smoke.sh`. It exercises runtime/device/context/model/session
  lifecycle, capability queries, option rejection with field index, result
  leasing, and generation through both the pull iterator and the push-style
  `turbo_generate`.
- The provider-agnostic Rust conformance suite under
  `crates/turbo-conformance/tests/`, run automatically as part of `cargo
  test --locked --workspace --exclude turbo-provider-cuda` (each file is a
  normal Rust integration test binary). This is the suite `PLAN.md`
  section 10 describes; it now exists.
- `providers/openvino/tests/provider_test.cpp`, a vtable-level test binary
  built alongside the OpenVINO provider library
  (`providers/openvino/CMakeLists.txt`) that calls `turbo_provider_vtbl`
  directly instead of going through `turbo-core`, to reach cases the core
  filters out before a provider ever sees them (a caller's smaller
  `struct_size`, an unknown option enumeration, a `top_n` above the row
  count, the truncation cases where a word is cut in half, and the
  `raw_scores`-with-no-fused-activation acceptance that `turbo-core`'s
  central `OPT_RAW_SCORES` gate would otherwise always reject for this
  provider). See `providers/openvino/README.md` and `docs/providers.md`'s
  "The OpenVINO provider" section.
- Header parity (`scripts/gen-header.sh --check`), the `struct_size` table
  (`scripts/gen-versioned.py --check`), and mock fixture parity (`cargo run
  -p turbo-core --example write_mock_bundles` diffed against
  `testdata/bundles/mock/`), all run in CI.
- The Java binding's conformance cases, run under JDK 25 through
  `bindings/java` (`mvn test`) against the mock provider, and the Swift
  binding's conformance cases, run as an executable
  (`swift run turbo-conformance`) through `bindings/swift` against the mock
  provider; and `scripts/package.sh`'s own archive-verification step
  (extract, compile and run the C smoke test against the packaged headers
  and library); see `docs/bindings.md` and `docs/packaging.md`.

On `krickert-mac` (Apple M2, macOS 27, Swift 6.4 command line tools) all
sixteen Swift binding cases pass (2026-09-22), and so does the macOS core
suite (`cargo test --workspace --exclude turbo-provider-cuda`) run on that
same machine.

`docs/reviews/2026-09-21-p0-p2.md` is an independent review of the P0-P2
tree that tracks each finding to closure; items closed by a commit carry
the test that would have failed before the fix (for example
`contract_struct_size_inside_a_field_is_rejected`, closing finding C3).

## The conformance suite

`crates/turbo-conformance/src/lib.rs` is the harness: it resolves a target
(runtime, device, bundle root) from the environment and provides the mock
bundle fixtures as defaults. The cases live under `tests/`, over 200 tests
across 25 files; most groups appear twice, once through the safe Rust API
(`*_rust.rs`) and once through the C ABI called directly (`*_c.rs`), plus
`header_parity.rs`, the three live-hardware files, and two Rust-only files
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
| `generation_rust.rs`, `generation_c.rs` | pull-iterator generation (prompt, step, cancel, finish reasons) and, in `generation_c.rs`, the push-style `turbo_generate`: push and pull yield one token sequence, and a callback returning `TURBO_STREAM_STOP` halts the stream right after the chunk that returned it |
| `header_parity.rs` | the committed headers match a fresh `cbindgen` run |
| `live_embed.rs`, `live_tasks.rs` | live checks against a real embedding/task provider library (`openvino`, `cuda`, `ggml`, or `hailo`) and bundles; see "Live provider tests" below |
| `live_generate.rs` | live checks against a real generation provider library (`ggml`) and a GGUF bundle; see "Live provider tests" below |

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
rerank, 128 new tokens for generation. Reported: p50 latency (p99 from 100
iterations up), tokens/s when the bundle's tokenizer counted the tokens,
H2D/D2H bytes and host allocations per run, device memory. Receipts carry
machine ID, runtime/driver versions, bundle hashes, and the commit the tool
was built from (`--commit` names it on a machine without git); budgets are
set from the first run per provider and then held, must name the same
provider, device and bundle, and a cell the budget has that a later run did
not measure is a violation. On the Hailo boards the live suite runs with
`--test-threads=1`: each test opens its own vdevice and HailoRT admits one
per process at a time.

`crates/turbo-bench` is the `libturbo` half of each pair: `turbo-bench
embed|rerank|generate` runs the workloads through the safe Rust API (the
text path and, where the bundle carries a tokenizer, the prepared-token
path), records p50/p99/mean latency, rows/s and tokens/s, and the per-run
H2D/D2H bytes and allocation counters from the session, and writes a
receipt with the machine, provider, runtime and driver versions, device,
bundle hashes, and commit. `--budget <earlier receipt>` holds a run to the
earlier p50 figures plus a tolerance (25% by default) and fails on a
regression; the first receipt per provider is the budget. `turbo-bench
discover` surveys a machine: every device with its features and
task-by-modality capability cells, and which of the named bundles it can
run. Benchmark receipts live under `testdata/receipts/turbo/bench/`
(2026-09-21: OpenVINO CPU and GPU, CUDA, ggml CUDA and CPU on `krick`;
Hailo-8 on `pi5ai1`). The direct-native reference program of each pair is
still to be written, which is why `openvino`, `cuda`, `ggml`, and `hailo`
stay `EXPERIMENTAL` rather than `SUPPORTED`.

## Live provider tests

`crates/turbo-conformance/tests/live_embed.rs`, `live_tasks.rs`, and
`live_generate.rs` load one real provider library against real bundles;
they are provider-agnostic (any provider that offers the task), unlike the
rest of the suite which runs against the mock bundles by default. The first
two replace the earlier `openvino_live.rs`/`openvino_live_tasks.rs`
(provider-specific, OpenVINO only). Each test prints a reason and returns
(not a failure) when its required environment variables are unset, so they
run safely as part of `cargo test --workspace` in an environment with no
provider configured. They are not run in CI: `.github/workflows/ci.yml` has
no OpenVINO install, excludes `turbo-provider-cuda` (hosted runners have no
CUDA toolkit), and builds `turbo-provider-ggml` with its CPU backend only
(no GGUF bundle is available to point `TURBO_LIVE_GGUF_BUNDLE` at in CI).

Provider selection (`crates/turbo-conformance/src/live.rs`), read by all
three files:

- `TURBO_LIVE_LIB`: path to the provider library
  (`libturbo_provider_openvino.so`, `libturbo_provider_cuda.so`,
  `libturbo_provider_ggml.so`, `libturbo_provider_hailo.so`).
- `TURBO_LIVE_PROVIDER`: the provider id it registers (`openvino`, `cuda`,
  `ggml`, `hailo`).
- `TURBO_LIVE_ORDINAL` (optional): device ordinal; default is the
  provider's CPU device if it has one, else ordinal 0.
- `TURBO_REFERENCE_DIR` (optional): directory holding the reference
  vectors (`testdata/reference_embeddings/ort_cuda_minilm_*.json`) so a
  copied test binary can find them on another machine; defaults to that
  path relative to the crate.

`live_embed.rs` (embed, against a MiniLM bundle), additionally gated on
`TURBO_LIVE_BUNDLE` (a bundle for `sentence-transformers/all-MiniLM-L6-v2`
with the artifact the provider reads: `onnx` or `openvino_ir`, `gguf`, or
`hef` plus `hailo_tables`; built with `tools/turbo-bundle`).
It checks the embedding vectors against ONNX Runtime CUDA FP32 reference
vectors at a floor `Live::embed_cosine_floor` derives from the device's
`EMBED x TEXT` capability cell (`crates/turbo-conformance/src/live.rs`):
0.9995 for an FP32 or unstated compute dtype; for a quantized dtype, the
floor the suite itself owns per provider and dtype
(`testdata/reference_embeddings/quantized_floors.json`, set from a
committed receipt — Hailo I8: 0.45), and the device must also report a
measured `cosine_floor` no higher than the suite's, so a provider cannot
set its own gate and one the suite has no floor for fails outright. Gates
ranking with Spearman > 0.85 over `testdata/corpus/sts-pairs.jsonl`
(0.944 for FP32 MiniLM, 0.937 for the INT8 Hailo HEF), caps the session
at the model's `max_seq` for fixed-shape artifacts, that batch rows equal
single-text runs, that
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

`live_generate.rs` (generation), gated on `TURBO_LIVE_GGUF_BUNDLE` (an
instruct GGUF model such as Qwen2.5-0.5B-Instruct, built with
`tools/turbo-bundle` per `docs/bundles.md`'s GGUF example). Its seven checks
run through the safe API against any provider offering `GENERATE`: the pull
iterator yields tokens in order and stops on the model's own end token or
the exact `max_new_tokens` budget, greedy decoding is deterministic and a
seeded sample reproduces, stop strings and stop tokens end the stream,
cancellation and logprobs work, `min_new_tokens` suppresses the end token,
and `structured_kind = JSON_SCHEMA` plus `n_sequences > 1` are refused
naming the field. All seven pass on `krick` (RTX 4080 SUPER CUDA device,
and the CPU device) and on `krickert-mac` (Apple M2, llama.cpp's own Metal
backend, the provider's `metal` Cargo feature, which is not the separate
`metal` provider under `providers/metal`).

Receipts from real runs are committed under `testdata/receipts/turbo/`:
`openvino-minilm-2026-09-21.json` and `openvino-tasks-2026-09-21.json` for
the OpenVINO provider, `cuda-2026-09-21.json` for the CUDA provider
(`krick`, RTX 4080 SUPER, cosine 1.000 against the FP32 references),
`ggml-2026-09-21.json` for the `ggml` provider (`krick`'s CUDA and CPU
devices, and `krickert-mac`'s Metal device), `metal-2026-09-22.json` for
the `metal` provider (`krickert-mac`, Apple M2, cosine 1.000 against the
FP32 references with results in unified memory), and `hailo-2026-09-21.json`
for the `hailo` provider (`pi5ai1` and `cm5ai1`, Hailo-8, cosine 0.32-0.71
against FP32 with Spearman 0.937 against 0.944); see
`docs/providers.md` for what they record.

### The CUDA provider on Jetson (`nano1`)

`providers/cuda`'s device-enumeration fix
(`providers/cuda/src/cuda.rs`, reading compute capability through
`cudaDeviceGetAttribute` and the device name through the driver library's
`cuDeviceGetName` instead of the ONNX Runtime's re-versioned
`cudaGetDeviceProperties`) unblocked running the provider on the Jetson
`nano1` board. On `nano1` (JetPack R39 rev 2.0, CUDA 13.2, ONNX Runtime
1.24.0 linked dynamically through `ORT_LIB_LOCATION` and
`--no-default-features`, per `providers/cuda/README.md`) the CUDA provider
passes all 12 live embedding tests (`live_embed.rs`) at cosine 1.000
against the FP32 reference vectors. There is no receipt file committed for
this machine yet, and the task suite (`live_tasks.rs`: rerank, classify,
token-classify) is still being verified there.

## Running what exists today

```bash
cargo test --locked --workspace --exclude turbo-provider-cuda   # unit tests plus the Rust conformance suite
scripts/gen-header.sh --check     # include/turbo/*.h matches turbo-abi/turbo-capi
scripts/gen-versioned.py --check  # crates/turbo-abi/src/versioned.rs matches its struct field lists
scripts/c-smoke.sh                # builds libturbo, runs smoke.c against testdata/bundles/mock
cargo run -p turbo-core --example write_mock_bundles \
  && git diff --exit-code -- testdata/bundles   # fixtures match mock.rs
```

`.github/workflows/ci.yml`'s `contract` job runs all five, plus `cargo fmt
--check`, `cargo clippy -D warnings` (also excluding `turbo-provider-cuda`),
and a standalone C11/C++17 compile of the header; `turbo-provider-ggml` is a
plain workspace member with no `--exclude`, so these same commands build
and unit-test it (CPU backend; the `cuda`/`metal`/`vulkan` features are
opt-in and not exercised here). It excludes `turbo-provider-cuda` because
hosted runners have no CUDA toolkit, and does not build the OpenVINO or
Hailo providers at all (no OpenVINO SDK or HailoRT on the runner); run any
of these providers' live tests, or `ggml`'s, by hand per
`providers/openvino/README.md`, `providers/cuda/README.md`,
`providers/ggml/README.md`, or `providers/hailo/README.md`.

Two further CI jobs, separate from `contract`: `java` builds `libturbo`
(`cargo build --locked -p turbo-shared`) and runs the Java binding's
conformance cases under JDK 25 against the mock provider
(`cd bindings/java && mvn -q -B test`, under `--illegal-native-access=deny`
per `bindings/java/pom.xml`'s surefire configuration); see
`docs/bindings.md`. `packaging`
runs `scripts/package.sh --no-cuda --no-openvino` (mock and static providers
only on a hosted runner) and uploads the resulting archive as a build
artifact; see `docs/packaging.md`.

## Fixtures

`testdata/bundles/mock/{embedding,reranker,classifier,token-classifier,
generative,generic}/` are generated, not hand-written; see
`docs/bundles.md`'s "Mock bundle layout" section and
`crates/turbo-core/examples/write_mock_bundles.rs`. Regenerate and commit
them whenever `crates/turbo-core/src/mock.rs` changes their shape.

`testdata/bundles/minilm-tokenizer/` is a real (non-mock), tokenizer-only
bundle used by the tokenizer, chunker, and importer tests; see
`docs/bundles.md`'s "Importer" section.
