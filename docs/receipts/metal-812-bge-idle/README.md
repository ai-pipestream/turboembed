# bge-base-en-v1.5 against MLX on the M2 (#812)

The end-to-end rounds of #812 for bge-base-en-v1.5 (manifest `0ecbae3e`,
CLS pooling), now that the MLX reference pools by CLS. They were measured
at `ade3608604e5965ed97095d37020511d71722d78` (main `89e6a6ae` plus the
CLS change) on the Apple M2 (Metal, macOS 27), against MLX 0.29.3 float16. FASTEST was used
at the docs/metal.md sizes plus 1, 8 and 32 x 512, over five rounds, with
the size order reversed in even rounds.

`e2e/` holds one log for each run, with its command, commit, and the load
and GPU before and after it. Before each run, the 1-minute load was 1.3 to
3.0 and the 5-minute load 1.9 to 2.6. The GPU was 0% busy before 50 of the 55 runs, and 6 to 42% before the other
five. The ChatGPT app was open throughout. `drivers/` holds the driver and summary
scripts. An earlier start stopped after one round when an app install
raised the load to 11; those runs are not used.

Each cell is the p50 in ms: the median over five rounds, with the least in brackets.

| Rows | turbo p50 median (least) | MLX p50 median (least) | ratio of medians | per-round ratios | min cosine |
|---|---|---|---|---|---|
| 32 x 256 | 830.02 (633.64) | 647.44 (520.36) | 1.282 | 1.236 1.087 1.282 1.184 1.832 | 0.9999997 |
| 32 x 128 | 334.84 (299.22) | 282.81 (248.05) | 1.184 | 1.201 1.030 1.132 1.206 1.184 | 0.9999997 |
| 16 x 128 | 154.63 (151.83) | 132.41 (125.18) | 1.168 | 1.223 1.154 1.177 1.213 1.181 | 0.9999997 |
| 8 x 128 | 69.88 (69.04) | 64.63 (63.04) | 1.081 | 1.081 1.035 1.113 1.095 1.127 | 0.9999997 |
| 4 x 128 | 35.76 (35.55) | 32.88 (32.31) | 1.088 | 1.070 1.088 1.126 1.103 1.099 | 0.9999997 |
| 2 x 128 | 18.70 (18.65) | 17.87 (17.08) | 1.046 | 1.045 1.053 1.044 1.072 1.092 | 0.9999997 |
| 1 x 128 | 10.30 (10.27) | 9.60 (9.42) | 1.074 | 1.073 1.071 1.070 1.070 1.094 | 0.9999997 |
| 1 x 32 | 4.90 (4.84) | 6.45 (6.41) | 0.759 | 0.764 0.757 0.772 0.762 0.746 | 0.9999997 |
| 1 x 512 | 45.13 (44.75) | 36.70 (35.45) | 1.230 | 1.258 1.282 1.223 1.317 1.262 | 0.9999997 |
| 8 x 512 | 373.89 (356.69) | 278.05 (272.10) | 1.345 | 1.345 1.333 1.307 1.246 1.311 | 0.9999997 |
| 32 x 512 | 1481.37 (1423.88) | 1460.03 (1130.43) | 1.015 | 1.270 0.931 1.206 1.235 0.714 | 0.9999997 |

Up to 16 x 128 and at 1 and 8 x 512, the per-round ratios agree within a
few percent. Up to 4 x 128, FASTEST trails MLX by 5 to 9%, and from 8 x 128 up by
8 to 35%. It leads only at 1 x 32. At 32 x 256 and 32 x 512, both sides'
p50 vary by 30 to 50% from round to round, at a GPU 0% busy before the run.
Those two rows are not a timing claim. A second sitting is needed for
them.
