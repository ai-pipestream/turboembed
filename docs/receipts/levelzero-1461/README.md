# Level Zero ModernBERT and nomic-bert (#1461, #1561)

Head: `8e980ac6365f755c7b80d75eefbe42198d82bb55` (claude/1461-levelzero
with claude/1461-cpu at `f22199098b5dd1448212f2935c7294cb560e064f` merged
in). Device: Intel Arc Pro B70 (Machine B), Level Zero, release builds,
`TURBO_TEST_REQUIRE_LEVELZERO=1`. These are correctness runs, not
timings: the box's three b70 CI runners were running jobs on the same
device, and the host's one-minute load average was 54.6 to 75.1 (each
log's first and last lines give it).

## Runs

| Log | Command | Result |
|---|---|---|
| `levelzero_rotary.log` | `TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 cargo test --release -p turbo --features levelzero --test levelzero_rotary -- --nocapture --test-threads 1` | 2 passed; device_info backend asserted `levelzero`; both families against the CPU at MODEL and EXACT, largest difference 1.89e-6 (bound 1e-4); FASTEST refused on this head |
| `conformance-modernbert-base-model.log` | `TURBO_TEST_BUNDLE=<bundles>/modernbert-base TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model cargo test --release -p turbo --features levelzero --test conformance -- --nocapture` | 7 passed; 12 of 12 reference cases up to 8192 tokens, compute dtype F32, 1 - min cosine 1.491e-11, max abs diff 3.815e-5 (floor 1e-4) |
| `conformance-nomic-embed-text-v1.5-model.log` | the same with `nomic-embed-text-v1.5` | 7 passed; 12 of 12, compute dtype F32, 1 - min cosine 5.224e-11, max abs diff 2.956e-5 |

Each conformance run also holds the tiny sealed rotary bundles (9 of 9
cases each) and the small BERT bundle's cases.

## Bundles

The CUDA leg's bundles (docs/receipts/cuda-1461 on #1580), copied to
Machine B; `manifest.json` SHA-256:

| Bundle | manifest.json |
|---|---|
| answerdotai/ModernBERT-base | `65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8` |
| nomic-ai/nomic-embed-text-v1.5 | `d5877f3f1087c39095e3363927a37df73679b6e5fcd685008f36ffe4ee591ab6` |
