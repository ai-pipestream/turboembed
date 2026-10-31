# Level Zero ModernBERT and nomic-bert, MODEL and FASTEST (#1461, #1575)

Head: `6f010ec8df2600c6ceee4c034019afbca0675e5b` (claude/1461-levelzero-f16
with main at `91cc625e74761b31ef7d8533f50ebc51aadc6156` merged in, and the
F16 narrowing of the rotary encoder saturating at 65504). Device: Intel Arc
Pro B70, Level Zero, release builds, `TURBO_TEST_REQUIRE_LEVELZERO=1`.
These are correctness runs, not timings: CI runners were running jobs on
the same device and host (each log's first and last lines give the load
average). The nomic-embed-text-v1.5 MODEL run was run twice: the first
lost the device at model load, before any case ran, and the log here is
the second.

## Runs

| Log | Command | Result |
|---|---|---|
| `levelzero_rotary.log` | `TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo --features levelzero --test levelzero_rotary -- --nocapture --test-threads 1` | 3 passed; device_info backend asserted `levelzero`; both families against the CPU at MODEL, EXACT and FASTEST (F32 at the generated bundles' widths), largest difference 1.89e-6 (bound 1e-4); the overflow bundles at widths 32 and 64 at FASTEST in F16, every value finite, least cosine to the CPU 0.789 for ModernBERT (floor 0.75) and 0.999996 for nomic-bert (floor 0.999) |
| `conformance-modernbert-base-model.log` | `TURBO_TEST_BUNDLE=<bundles>/modernbert-base TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model cargo test --release -p turbo --features levelzero --test conformance -- --nocapture` | 7 passed; 12 of 12 reference cases up to 8192 tokens, compute dtype F32, 1 - min cosine 1.491e-11, max abs diff 3.815e-5 (floor 1e-4) |
| `conformance-nomic-embed-text-v1.5-model.log` | the same with `nomic-embed-text-v1.5` | 7 passed; 12 of 12, compute dtype F32, 1 - min cosine 5.224e-11, max abs diff 2.956e-5 |
| `conformance-modernbert-base-fastest.log` | the first with `TURBO_TEST_PRECISION=fastest` | 7 passed; 12 of 12, compute dtype F16, 1 - min cosine 8.732e-7 (floor 1e-3), max abs diff 6.950e-3 |
| `conformance-nomic-embed-text-v1.5-fastest.log` | the same with `nomic-embed-text-v1.5` | 7 passed; 12 of 12, compute dtype F16, 1 - min cosine 4.869e-7, max abs diff 2.737e-3 |

Each conformance run also holds the tiny sealed rotary bundles (9 of 9
cases each, F32 at their widths) and the small BERT bundle's cases.

## Bundles

The CUDA leg's bundles (docs/receipts/cuda-1461 on #1580), copied to
Machine B; `manifest.json` SHA-256:

| Bundle | manifest.json |
|---|---|
| answerdotai/ModernBERT-base | `65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8` |
| nomic-ai/nomic-embed-text-v1.5 | `d5877f3f1087c39095e3363927a37df73679b6e5fcd685008f36ffe4ee591ab6` |
