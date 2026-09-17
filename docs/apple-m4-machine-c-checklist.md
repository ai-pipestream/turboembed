# Apple M4 qualification — Machine C checklist

Roadmap M4 requires the Apple provider to prove Metal resource ownership and
Swift concurrency, its own matched native-overhead measurement, and packaging
or a documented qualification track. Machine C (the Apple Silicon reference
host) was offline during the 2026-09-16 M4 launch on Machine A, so this page
records exactly what is landed, what is gated, and how to re-run the gated
proofs. **Nothing below claims live Metal qualification without a Machine C
receipt.**

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
3. **Metal ownership under multiple live engines.** The M4 analog of the CUDA
   two-engine allocator proof: run the two-engine concurrent embed XCTest and
   the `mlx-live` multi-engine cases; record per-engine outputs matching the
   single-engine run within 1e-5.
4. **Matched native overhead.** The harness is **landed**; the live run is
   still the open gate. `make bench-apple-overhead` builds
   [`swift/Sources/BenchAppleOverhead`](../swift/Sources/BenchAppleOverhead/BenchAppleOverhead.swift)
   — a direct mlx-swift Metal reference (raw `MLXEmbedders` consumer, fixed
   `[batch, 256]` shapes) versus the `libTurboEmbed.dylib` C ABI resolved
   with dlopen, on identical MiniLM inputs (batch 1/8/32 × tokens 32/128/256
   × full/mixed, 18 cases) — and writes
   `testdata/receipts/bench/machine-c-metal-overhead.json` with the same
   predeclared budgets as the NVIDIA/Intel pilots (ABI p50 ≤ 1.05×,
   throughput ≥ 0.95×, parity max abs ≤ 5e-4, RMSE ≤ 1e-4, ABI steady-state
   arena allocs == 0). On Machine C: run the target (needs
   `make fetch-mlx ALIASES=minilm`), then validate the receipt with
   `cargo test -p turboembed --features mlx-live --test apple_overhead_receipt
   -- --ignored --nocapture`. No receipt exists yet; this branch was authored
   off-Metal and does not fake one.
5. **SOLIDIFY bench refresh.** `make bench-turbo MACHINE=C` → refreshes
   `testdata/receipts/bench/machine-c-metal.json` on the current tree.
6. **bge-small CLS contract (model coverage).** Provision
   `models/mlx/bge-small`, then run the Apple equivalent of
   `nvidia_bge_small` (CLS+L2 vs `testdata/e2e/goldens/apple/bge-small.json`).
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
