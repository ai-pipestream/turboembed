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
4. **Matched native overhead.** The Apple analog of
   `make bench-nvidia-overhead` (direct MLX/Metal reference vs the
   `libTurboEmbed.dylib` ABI on identical inputs/shapes) has **not** been
   implemented; implementing and running it on Machine C is an open M4 gate.
   Use the same predeclared budgets (ABI p50 within 5%, throughput ≥ 95%,
   parity max abs ≤ 5e-4) and write
   `testdata/receipts/bench/machine-c-metal-overhead.json`.
5. **SOLIDIFY bench refresh.** `make bench-turbo MACHINE=C` → refreshes
   `testdata/receipts/bench/machine-c-metal.json` on the current tree.
6. **bge-small CLS contract (model coverage).** Provision
   `models/mlx/bge-small`, then run the Apple equivalent of
   `nvidia_bge_small` (CLS+L2 vs `testdata/e2e/goldens/apple/bge-small.json`).
7. **Packaging.** The Apple installable artifact (dylib + headers + Swift
   package + consumer acceptance, analogous to
   `scripts/make-nvidia-sdk-release.sh` /
   `scripts/nvidia-sdk-consumer-acceptance.sh`) is an open M4 gate; land it
   from Machine C where the produced dylib can actually be loaded.

## Known cross-provider follow-up found on Machine A

The NVIDIA overhead pilot found that `turbo_buffer_arena_rent` re-ran the
CUDA backend probe on every rent, re-querying device properties (~0.8 ms per
embed call); it is fixed by memoizing the probe
(`native/turbo_buffer/src/cuda.cpp`). The Metal (`metal.mm`) and Level Zero
(`ze.cpp`) backends already memoize through `init_once`, so no Apple change
was made from Machine A — but the Apple overhead pilot (step 4) should
confirm on hardware that no equivalent per-request cost hides elsewhere in
the dylib path.
