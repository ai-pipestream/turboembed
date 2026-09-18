# Apple M4 qualification — Machine C checklist

Roadmap M4 requires the Apple provider to prove Metal resource ownership and
Swift concurrency, its own matched native-overhead measurement, and packaging
or a documented qualification track. Machine C (the Apple Silicon reference
host) was offline during the 2026-09-16 M4 launch on Machine A, so this page
records exactly what is landed, what is gated, and how to re-run the gated
proofs. **Nothing below claims live Metal qualification without a Machine C
receipt.**

**Status (2026-09-18):** the live Machine C receipts for steps 2, 4, and 5
landed at git_sha `0587e81` (Apple M2 host, `Kristians-MacBook-Air`):

- Step 4 (matched native overhead):
  [`testdata/receipts/bench/machine-c-metal-overhead.json`](../testdata/receipts/bench/machine-c-metal-overhead.json)
  — `pass: true`, full 18-case grid, all 54 timed repeats within the
  predeclared budgets, parity max abs ≤ 1.1e-7 and RMSE ≤ 1.2e-8 across
  cases, ABI steady-state arena allocs 0 on every case.
- Step 5 (SOLIDIFY bench):
  [`testdata/receipts/bench/machine-c-metal.json`](../testdata/receipts/bench/machine-c-metal.json)
  — `pass: true`, with the three provably stale 2026-09-12 apple-golden CJK
  entries exempted against the nvidia golden at ≥ 0.999 and listed in
  `stale_apple_golden_items` (see `testdata/e2e/goldens/apple/README.md`).
- Step 2 (live Metal MiniLM contract):
  [`testdata/receipts/turboembed/apple-minilm.json`](../testdata/receipts/turboembed/apple-minilm.json)
  — cosine vs nvidia min 0.9794 (floor 0.97), vs apple min 0.99999976
  (floor 0.99), AUTO→Metal, allocs after forward 0.

The step 3 (recorded multi-engine outputs) and step 6 (bge-small CLS
contract) **harnesses are landed** (see those steps below); their live
Machine C runs and receipts are still open. Step 7's SDK release and
acceptance passed on Machine C on 2026-09-17
(see [machine-c-rerun-2026-09-17.md](machine-c-rerun-2026-09-17.md)).

## Landed and provable off-Metal (this branch)

- Contract scaffolding and fail-loud device policy compile and run on any
  host: `make turborerank-tests-nometal` proves a Metal request without Metal
  is a loud error, never CPU; the CUDA-off `turboembed` stub rejects
  `TURBOEMBED_DEVICE_METAL` the same way.
- The Swift ownership/concurrency receipts from M0 remain valid history:
  [apple-ownership-validation-2026-09-14.md](apple-ownership-validation-2026-09-14.md)
  (18 XCTest cases on an M2: UTF-8/NUL wrappers, two-engine concurrent embed,
  Metal arena rents, result-retains-engine lifetimes).
- The catalog Apple entries (`minilm` mean+L2, `bge-small` CLS+L2,
  `max_batch_size = 32`) stay pinned to the same checkpoints the NVIDIA and
  Intel contracts qualified; no model substitution.
- The step 4 and step 7 **tooling** is landed (harness, packaging and
  acceptance scripts, Make targets — see those steps below). It was authored
  off-Metal: the Swift harness and the scripts refuse to run without macOS +
  Metal, and no receipt or release artifact is claimed until they run on
  Machine C.

## Gated on Machine C — run in this order

Each step names its receipt target. Do not mark any of them done from this
branch; they need live Metal hardware.

1. **Workspace + Swift baseline.** `git pull` this branch, then:
   - `cargo test --locked --workspace` (parallel) must stay green on macOS.
   - `cd swift && swift test` (XCTest ownership + MetalArena suites).
   Refresh `testdata/receipts/turboembed/apple-ownership-*.log` if the Swift
   surface changed.
2. **Live Metal MiniLM contract.** Provision `models/mlx/minilm`
   (`cargo xtask fetch --mlx`, SHA-pinned), then:
   `cargo test -p turboembed --features mlx-live -- --include-ignored --nocapture`
   → refreshes `testdata/receipts/turboembed/apple-minilm.json`
   (cosine vs `testdata/e2e/goldens/apple/minilm.json`, AUTO→Metal policy).
   **Done:** the live receipt is in-tree (see the status note above); the
   2026-09-17 cosine failure was resolved by exempting the provably stale
   apple-golden entries, recorded in `machine-c-metal.json`.
3. **Metal ownership under multiple live engines.** The M4 analog of the CUDA
   two-engine allocator proof: run the two-engine concurrent embed XCTest and
   the `mlx-live` multi-engine cases; record per-engine outputs matching the
   single-engine run within 1e-5. The harness is **landed**; the live run is
   the open gate. On Machine C (needs `make fetch-mlx ALIASES=minilm`):
   - `cd swift && swift test --filter MetalWrapperOwnershipTests`
     (two engines, concurrent wrapper calls, result-retains-engine lifetime).
   - `cargo test -p turboembed --features mlx-live --test apple_multi_engine
     -- --ignored --nocapture` — the Rust analog of
     [`intel_engine_isolation.rs`](../crates/turboembed/tests/intel_engine_isolation.rs):
     two `Device::Metal` engines embed concurrently, retained results survive
     peer engine destruction, and every per-engine output must match a
     single-engine baseline within 1e-5 absolute. Writes
     `testdata/receipts/turboembed/apple-multi-engine.json`
     (git_sha, chip, max_abs, pass).
4. **Matched native overhead.** The harness is **landed**; the live run is
   still the open gate. `make bench-apple-overhead` builds
   [`swift/Sources/BenchAppleOverhead`](../swift/Sources/BenchAppleOverhead/BenchAppleOverhead.swift)
   — a direct mlx-swift Metal reference (raw `MLXEmbedders` consumer, fixed
   `[batch, 256]` shapes) versus the `libTurboEmbed.dylib` C ABI resolved
   with dlopen in an isolated per-case
   [`bench-abi-worker`](../swift/Sources/BenchAbiWorker/BenchAbiWorker.swift)
   process (no MLX / swift-transformers linked into the worker, so the
   dylib's Objective-C classes are never duplicated), on identical MiniLM
   inputs (batch 1/8/32 × tokens 32/128/256
   × full/mixed, 18 cases) — and writes
   `testdata/receipts/bench/machine-c-metal-overhead.json` with the same
   predeclared budgets as the NVIDIA/Intel pilots (ABI p50 ≤ 1.05×,
   throughput ≥ 0.95×, parity max abs ≤ 5e-4, RMSE ≤ 1e-4, ABI steady-state
   arena allocs == 0). On Machine C: run the target (needs
   `make fetch-mlx ALIASES=minilm`), then validate the receipt with
   `cargo test -p turboembed --features mlx-live --test apple_overhead_receipt
   -- --ignored --nocapture`. **Done:** the live receipt landed at
   `0587e81` with `pass: true` (see the status note above). One caveat from
   that run: the harness serialized the parity gate constant through Swift
   `Float`, so the receipt records `parity_max_abs` as the f32 bit pattern
   `0.0005000000237487257`; the validator accepts exactly that rounding of
   `5e-4` (nothing looser), and receipts produced after the fix record the
   exact decimal.
5. **SOLIDIFY bench refresh.** `make bench-turbo MACHINE=C` → refreshes
   `testdata/receipts/bench/machine-c-metal.json` on the current tree.
   **Done:** the live receipt landed at `0587e81` with `pass: true` (see
   the status note above).
6. **bge-small CLS contract (model coverage).** The harness is **landed** as
   [`apple_bge_small.rs`](../crates/turboembed/tests/apple_bge_small.rs), the
   Apple equivalent of `nvidia_bge_small`; the live run is the open gate. On
   Machine C: `make fetch-mlx ALIASES=bge-small`, then
   `cargo test -p turboembed --features mlx-live --test apple_bge_small
   -- --ignored --nocapture` — CLS+L2 on `Device::Metal` (fail loud, no CPU),
   cosine ≥ 0.99 against the `parity:*` subset of
   `testdata/e2e/goldens/apple/bge-small.json`, and a mean-pooling request on
   the CLS alias must be NotImplemented (the Metal ABI now rejects a
   per-call pooling that differs from the loaded catalog contract, matching
   the NVIDIA path). Writes
   `testdata/receipts/turboembed/apple-bge-small.json`.
7. **Packaging.** The tooling is **landed**; producing and accepting the
   artifact on Machine C is still the open gate.
   [`scripts/make-apple-sdk-release.sh`](../scripts/make-apple-sdk-release.sh)
   (`make apple-sdk-release`) stages `libTurboEmbed.dylib` (install name
   `@rpath/libTurboEmbed.dylib`), the MLX Metal kernel libraries that must
   sit next to it, `include/turboembed.h`, a CMake package, the C consumer
   example, a documented swiftc-built Swift consumer, MiniLM MLX SHA-256
   model pins from `models/manifests/mlx.json`, license texts, the exported
   C-symbol record, and a hashed file manifest into
   `dist/turboembed-metal-sdk-<version>-macos-arm64.tar.gz` (+ `.sha256`).
   [`scripts/apple-sdk-consumer-acceptance.sh`](../scripts/apple-sdk-consumer-acceptance.sh)
   (`make apple-sdk-acceptance MODE=metal`, analogous to
   `scripts/nvidia-sdk-consumer-acceptance.sh`) proves a clean consumer in a
   fresh temporary directory: archive and manifest hashes, model files
   verified against the packaged pins (a tampered `model.safetensors` is
   refused), C and Swift consumers built against the extracted prefix with a
   minimal environment, and the device-policy matrix — METAL/AUTO must
   succeed on the Metal host, explicit CPU must refuse catalog aliases
   loudly, MOCK serves only the smoke alias, and `MODE=no-metal` proves a
   missing Metal device fails loudly. Both scripts refuse to run off
   macOS arm64, so the artifact can only be produced where the dylib is
   actually loaded.

## Known cross-provider follow-up found on Machine A

The NVIDIA overhead pilot found that `turbo_buffer_arena_rent` re-ran the
CUDA backend probe on every rent, re-querying device properties (~0.8 ms per
embed call); it is fixed by memoizing the probe
(`native/turbo_buffer/src/cuda.cpp`). The Metal (`metal.mm`) and Level Zero
(`ze.cpp`) backends already memoize through `init_once`, so no Apple change
was made from Machine A — but the Apple overhead pilot (step 4) should
confirm on hardware that no equivalent per-request cost hides elsewhere in
the dylib path.
