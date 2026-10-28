# Level Zero ModernBERT and nomic-bert, MODEL and FASTEST (#1461, #1575)

Head: `e0b8835e3d16ec67d25d8be154982877271c172a` (claude/1461-levelzero-f16
with claude/1461-levelzero at `89f341142576d078a08f2be2fffa73f73c5718cc`
merged in, which carries main at `b649bd652fb455d1c285abe5d930fc1cfc0dab11`). Device: Intel Arc Pro B70
(Machine B), Level Zero, release builds, `TURBO_TEST_REQUIRE_LEVELZERO=1`.
These are correctness runs, not timings: the box's three b70 CI runners
were running jobs on the same device, and the host's one-minute load
average was 63.3 to 83.2 (each log's first and last lines give it).

## Runs

| Log | Command | Result |
|---|---|---|
| `levelzero_rotary.log` | `TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo --features levelzero --test levelzero_rotary -- --nocapture --test-threads 1` | 2 passed; device_info backend asserted `levelzero`; both families against the CPU at MODEL, EXACT and FASTEST (F32 at the generated bundles' widths), largest difference 1.89e-6 (bound 1e-4) |
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
