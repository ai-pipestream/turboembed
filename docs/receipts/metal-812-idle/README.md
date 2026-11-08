# #812 receipts on an idle M2

The GEMM and end-to-end receipts that #812 lists, taken with the desktop's
apps closed. They were measured at main `89e6a6ae0150730ad38e5d43a5333819e3769a24`
on the Apple M2 (Metal, macOS 27), against MLX 0.29.3. No code changed.
The earlier GEMM receipt (`docs/receipts/metal-gemm-812/`, at `618fbee963d4`)
was taken with the GPU up to 53% busy and a 1-minute load of 3.2 to 4.2.

## Files

- `schedule.jsonl`, `turbo.jsonl`, `mlx.jsonl`, `summary.json`: the GEMM
  receipt, in the format of `docs/receipts/metal-gemm-812/` (its README
  describes each file). Before each of the 400 runs, the 1-minute load was 1.8 to 3.1, the
  5-minute load 1.9 to 2.2, and the GPU 0% busy (one run 12%).
- `kernels-check.log`: `metal_f16` at the same commit (9 passed), and
  `a_real_bundle_at_its_longest_rows_stays_inside_f16` with all-MiniLM-L6-v2
  (1 - cosine 1.965e-6) and bge-base-en-v1.5 (4.635e-6), each tile the same.
- `e2e/`: one log for each `turbo-bench record` run, 5 rounds at 11 sizes,
  each with its command, commit, and the load and GPU before and after.
  The size order is reversed in even rounds. Before each run the 1-minute load was 1.6
  to 2.0, the 5-minute 2.3 to 2.5, and the GPU 0% (one run 12%).
  `e2e/summary.md` is the table below.
- `drivers/`: the two driver scripts and the two summary scripts.

## The linear layers alone (bare mode)

Ours is `linear=auto`'s tile. Each cell is least / median of medians / mean of means /
largest p99 over five rounds, host time in ms. The ratio is of the medians.

| h | M | shape | ours | MLX | ours / MLX | our fastest tile |
|---|---|---|---|---|---|---|
| 384 | 128 | qkv | 0.200 / 0.245 / 0.243 / 0.545 | 0.201 / 0.328 / 0.328 / 1.208 | 0.74 | h32n32a 0.246 |
| 384 | 128 | attn_out | 0.135 / 0.202 / 0.220 / 0.635 | 0.254 / 0.281 / 0.301 / 1.133 | 0.72 | h32n32a 0.219 |
| 384 | 128 | ffn_in | 0.233 / 0.260 / 0.261 / 0.302 | 0.366 / 0.391 / 0.407 / 1.268 | 0.66 | h32n32a 0.246 |
| 384 | 128 | ffn_out | 0.228 / 0.262 / 0.265 / 0.663 | 0.401 / 0.552 / 0.571 / 1.070 | 0.47 | h32n32 0.261 |
| 384 | 512 | qkv | 0.346 / 0.381 / 0.386 / 0.645 | 0.323 / 0.768 / 0.772 / 1.685 | 0.50 | h32n32a 0.371 |
| 384 | 512 | attn_out | 0.202 / 0.251 / 0.256 / 0.566 | 0.366 / 0.392 / 0.400 / 0.536 | 0.64 | h32n32a 0.241 |
| 384 | 512 | ffn_in | 0.401 / 0.428 / 0.441 / 0.691 | 0.375 / 0.907 / 0.896 / 2.002 | 0.47 | h32n32a 0.426 |
| 384 | 512 | ffn_out | 0.378 / 0.415 / 0.423 / 0.763 | 0.461 / 0.998 / 0.956 / 1.862 | 0.42 | h32n32a 0.415 |
| 384 | 1024 | qkv | 0.509 / 0.638 / 0.626 / 0.950 | 0.452 / 1.140 / 1.100 / 2.043 | 0.56 | h32n32a 0.554 |
| 384 | 1024 | attn_out | 0.279 / 0.303 / 0.315 / 0.780 | 0.501 / 0.539 / 0.590 / 1.500 | 0.56 | h32n32a 0.308 |
| 384 | 1024 | ffn_in | 0.683 / 0.759 / 0.765 / 1.007 | 0.538 / 1.460 / 1.245 / 1.805 | 0.52 | h32n32 0.710 |
| 384 | 1024 | ffn_out | 0.632 / 0.689 / 0.695 / 0.983 | 0.593 / 1.565 / 1.243 / 2.249 | 0.44 | h32n32a 0.654 |
| 384 | 4096 | qkv | 1.637 / 1.717 / 1.939 / 3.798 | 1.309 / 1.387 / 2.040 / 4.633 | 1.24 | h32n32a 1.533 |
| 384 | 4096 | attn_out | 0.679 / 0.752 / 0.759 / 1.218 | 0.541 / 1.185 / 1.164 / 1.669 | 0.63 | h32n32a 0.678 |
| 384 | 4096 | ffn_in | 2.131 / 2.196 / 2.207 / 2.573 | 1.657 / 1.749 / 2.274 / 5.303 | 1.26 | h32n32a 1.957 |
| 384 | 4096 | ffn_out | 2.051 / 2.127 / 2.140 / 2.649 | 1.677 / 1.785 / 2.079 / 5.101 | 1.19 | h32n32a 1.894 |
| 384 | 16384 | qkv | 6.013 / 6.104 / 6.123 / 6.596 | 4.594 / 4.659 / 4.882 / 8.638 | 1.31 | h32n32a 5.423 |
| 384 | 16384 | attn_out | 2.148 / 2.245 / 2.257 / 2.547 | 1.665 / 1.758 / 2.173 / 5.334 | 1.28 | h32n32a 1.987 |
| 384 | 16384 | ffn_in | 7.951 / 8.051 / 8.070 / 8.415 | 6.003 / 6.153 / 6.310 / 8.919 | 1.31 | h32n32a 7.174 |
| 384 | 16384 | ffn_out | 7.390 / 7.474 / 7.493 / 7.700 | 5.918 / 6.046 / 6.091 / 7.015 | 1.24 | h32n32a 6.718 |
| 768 | 128 | qkv | 0.347 / 0.381 / 0.389 / 0.785 | 0.337 / 0.781 / 0.745 / 1.700 | 0.49 | h32n32a 0.354 |
| 768 | 128 | attn_out | 0.213 / 0.247 / 0.261 / 0.753 | 0.441 / 0.470 / 0.496 / 1.378 | 0.52 | h32n32a 0.238 |
| 768 | 128 | ffn_in | 0.410 / 0.427 / 0.445 / 0.901 | 0.460 / 0.926 / 0.919 / 1.438 | 0.46 | h32n32 0.429 |
| 768 | 128 | ffn_out | 0.396 / 0.438 / 0.443 / 0.609 | 0.528 / 1.351 / 1.169 / 2.291 | 0.32 | h32n32a 0.414 |
| 768 | 512 | qkv | 0.901 / 0.963 / 0.969 / 1.206 | 0.745 / 0.955 / 1.332 / 2.433 | 1.01 | h32n32a 0.926 |
| 768 | 512 | attn_out | 0.399 / 0.422 / 0.435 / 0.934 | 0.382 / 0.926 / 0.884 / 1.825 | 0.46 | h32n32a 0.415 |
| 768 | 512 | ffn_in | 1.123 / 1.190 / 1.205 / 1.617 | 0.915 / 1.062 / 1.541 / 3.137 | 1.12 | h32n32a 1.118 |
| 768 | 512 | ffn_out | 1.061 / 1.132 / 1.143 / 1.516 | 1.003 / 1.135 / 1.598 / 3.352 | 1.00 | h32n32a 1.081 |
| 768 | 1024 | qkv | 1.485 / 1.613 / 1.630 / 2.173 | 1.277 / 1.400 / 1.945 / 4.004 | 1.15 | h32n32a 1.510 |
| 768 | 1024 | attn_out | 0.633 / 0.722 / 0.731 / 1.030 | 0.563 / 1.506 / 1.288 / 1.831 | 0.48 | h32n32a 0.690 |
| 768 | 1024 | ffn_in | 1.960 / 2.050 / 2.065 / 2.576 | 1.667 / 1.733 / 2.155 / 5.377 | 1.18 | h32n32a 1.935 |
| 768 | 1024 | ffn_out | 1.904 / 1.976 / 1.982 / 2.381 | 1.761 / 1.857 / 2.173 / 5.309 | 1.06 | h32n32a 1.872 |
| 768 | 4096 | qkv | 5.691 / 5.786 / 5.801 / 6.181 | 4.520 / 4.608 / 4.767 / 8.247 | 1.26 | h32n32a 5.177 |
| 768 | 4096 | attn_out | 2.046 / 2.127 / 2.144 / 2.637 | 1.641 / 1.736 / 2.278 / 5.431 | 1.23 | h32n32a 1.909 |
| 768 | 4096 | ffn_in | 7.496 / 7.598 / 7.617 / 8.019 | 5.978 / 6.072 / 6.188 / 8.224 | 1.25 | h32n32a 6.826 |
| 768 | 4096 | ffn_out | 7.313 / 7.403 / 7.446 / 7.758 | 6.006 / 6.084 / 6.221 / 8.606 | 1.22 | h32n32a 6.683 |
| 768 | 16384 | qkv | 22.182 / 22.246 / 22.274 / 22.526 | 17.458 / 17.576 / 17.734 / 19.911 | 1.27 | h32n32a 19.996 |
| 768 | 16384 | attn_out | 7.533 / 7.633 / 7.646 / 7.758 | 5.952 / 6.063 / 6.146 / 7.555 | 1.26 | h32n32a 6.859 |
| 768 | 16384 | ffn_in | 29.448 / 29.550 / 29.568 / 29.923 | 23.148 / 23.397 / 23.847 / 56.243 | 1.26 | h32n32a 26.583 |
| 768 | 16384 | ffn_out | 28.504 / 28.636 / 28.655 / 29.000 | 23.071 / 23.330 / 23.364 / 24.317 | 1.23 | h32n32a 26.246 |

The pattern matches the earlier receipt. MLX leads at 16384 tokens (23 to 31%), and
at 4096 tokens on every shape but 384 attn_out (19 to 26%). At h = 768 it also leads qkv, ffn_in
and ffn_out from 512 tokens up (0 to 18%). On every other shape up to 1024
tokens, ours is ahead (ratio 0.32 to 0.74).

One finding is new. Above `AUTO_H32_TOKENS`
(1024), auto runs `h64`, but `h32n32` is about 5% faster there, and the
aligned `h32n32a` is about 10% faster at every shape at 4096 and 16384.
At 768 x 16384 qkv, for example, auto takes 22.25 ms, `h32n32` 21.16 ms,
`h32n32a` 20.00 ms and MLX 17.58 ms. Even `h32n32a` trails MLX by about 14%.

## End to end: all-MiniLM-L6-v2 at FASTEST against MLX float16

`turbo-bench record ... --precision fastest --rows dense --warmup 5
--iterations 20 --no-tei --mlx-model <upstream all-MiniLM-L6-v2>`. Each cell
is the p50 in ms: the median over five rounds, with the least in brackets.

| Rows | turbo p50 median (least) | MLX p50 median (least) | ratio of medians | per-round ratios | min cosine |
|---|---|---|---|---|---|
| 32 x 256 | 99.82 (99.67) | 101.74 (101.62) | 0.981 | 0.981 0.981 0.985 0.980 0.986 | 0.9999996 |
| 32 x 128 | 45.53 (45.45) | 44.95 (44.92) | 1.013 | 1.013 1.011 1.014 1.013 1.014 | 0.9999996 |
| 16 x 128 | 23.75 (23.73) | 22.65 (22.63) | 1.049 | 1.049 1.048 1.050 1.046 1.051 | 0.9999996 |
| 8 x 128 | 11.05 (11.01) | 11.82 (11.81) | 0.935 | 0.932 0.936 0.913 0.936 0.933 | 0.9999996 |
| 4 x 128 | 6.08 (6.08) | 6.39 (6.37) | 0.952 | 0.953 0.954 0.951 0.944 0.955 | 0.9999996 |
| 2 x 128 | 3.52 (3.51) | 3.77 (3.71) | 0.935 | 0.931 0.944 0.931 0.949 0.935 | 0.9999996 |
| 1 x 128 | 2.29 (2.28) | 2.55 (2.50) | 0.898 | 0.702 0.899 0.915 0.897 0.685 | 0.9999996 |
| 1 x 32 | 1.29 (1.26) | 2.63 (2.39) | 0.490 | 0.482 0.487 0.503 0.490 0.542 | 0.9999996 |
| 1 x 512 | refused: no reference case has 512 tokens | | | | |
| 8 x 512 | refused: no reference case has 512 tokens | | | | |
| 32 x 512 | refused: no reference case has 512 tokens | | | | |

Compared with #1860's rounds at `75b4af00`, the ratios agree within 1.5%,
except at 1 x 32 (0.49 now, 0.64 then; ours fell from 1.70 to 1.29 ms). FASTEST still trails MLX at
32 x 128 (1.3%) and 16 x 128 (4.9%).

## Not run, with the current tools unchanged

- MiniLM at 1, 8 and 32 x 512: `turbo-bench` refuses, because no reference
  case in the bundle reaches 512 tokens (the model's maximum is 256).
- bge-base-en-v1.5 against MLX: `turbo-bench` refuses because
  `mlx_bench.py` mean-pools and bge-base pools by CLS. The turbo side runs
  (1 x 128: p50 10.34 ms, cosine 0.9999997). An MLX comparison needs CLS
  pooling in `mlx_bench.py`.
