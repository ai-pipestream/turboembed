# CUDA ModernBERT and nomic-bert (#1461, #1580)

Head: `0fda7ded8` (claude/1461-cuda with claude/1461-cpu at `f22199098`
merged in). Device: RTX 4080 SUPER (sm_89), nvcc 12.4,
`TURBO_CUDA_ROOT=/usr`. These are correctness runs, not timings: the
host's one-minute load average was 62 to 72 on 32 threads when the runs
ended, and another compute process held 5.5 GB of GPU memory.

## Attention sums by row length

`nomic-embed-text-v1.5-scan.log` and `modernbert-base-scan.log` are the
output of `core/tests/cuda_variants.rs`
`attention_sums_by_row_length_of_a_real_bundle`: FASTEST on the device
against the CPU at MODEL, synthetic token rows (row r has
`seq - (r * 7) % (seq / 2)` tokens), the tensor cores' attention forced
to F16 sums (`mma128`) and to F32 sums (`mma128-acc32`).

| Rows x tokens | nomic F16 | nomic F32 | ModernBERT F16 | ModernBERT F32 |
|---|---|---|---|---|
| 4 x 512 | 5.1e-6 | 2.7e-6 | 2.1e-5 | 1.5e-5 |
| 4 x 1024 | 2.0e-5 | 2.4e-6 | 1.1e-4 | 1.6e-4 |
| 4 x 2048 | 1.8e-4 | 2.7e-6 | 2.6e-5 | 2.8e-5 |
| 4 x 4096 | 1.5e-4 | 1.2e-6 | 1.1e-4 | 3.2e-6 |
| 4 x 8192 | 9.8e-4 | 1.1e-6 | 5.5e-4 | 8.6e-6 |
| 8 x 8192 | **1.4e-3** | 1.1e-6 | 8.0e-4 | 1.4e-5 |

Values are 1 - lowest cosine with the CPU; FASTEST's bound is 1e-3. With
F16 sums nomic goes over it at 8 x 8192, which is why a session whose
max_seq is over 4096 takes F32 sums (`ACC16_ATTENTION_MAX_KEYS` in
`core/cuda/backend.cpp`). The ModernBERT rows at 1024 and 2048 are
level between the two sums, so what sets them is not the attention's
sums; ModernBERT's local layers read at most 128 keys whatever the row's
length.

```
TURBO_CUDA_ROOT=/usr TURBO_TEST_REQUIRE_CUDA=1 TURBO_TEST_DEVICE=cuda TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release --no-fail-fast -p turbo --features cuda --test 'cuda_*' -- --include-ignored --nocapture
```

## Bundles

Made with turbo-bundle from the hub checkouts below; `manifest.json`
SHA-256:

| Bundle | Source commit | max_batch x max_seq | manifest.json |
|---|---|---|---|
| nomic-ai/nomic-embed-text-v1.5 | `e9b6763023c676ca8431644204f50c2b100d9aab` | 8 x 8192 | `d5877f3f1087c39095e3363927a37df73679b6e5fcd685008f36ffe4ee591ab6` |
| answerdotai/ModernBERT-base | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | 8 x 8192 | `65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8` |

## The rest of the run

`tests.log` holds each binary's result line at the same head, all
passing: every `core/tests/cuda_*.rs` binary on the generated fixtures;
conformance on the small sealed bundle at MODEL, FASTEST and EXACT
(`TURBO_TEST_DEVICE=cuda TURBO_TEST_PRECISION=<p> cargo test --release
-p turbo --features cuda --test conformance`); and, for each real bundle,
conformance at the three precisions with `--include-ignored` and every
`cuda_*` binary with `--include-ignored` (the largest shape against the
CPU, the reference cases against upstream, the first tuned session's
cost).

No timing of the F32 sums' cost is recorded here.
