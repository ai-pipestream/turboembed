# FASTEST end to end on the M2 with the aligned tile: correctness (#812)

At `e9b0491d` (the aligned 32 x 32 tile, #1455), `turbo-bench record
--bundle <all-minilm-l6-v2> --device metal --precision fastest --rows
dense --batch <b> --seq <s> --warmup 5 --iterations 20 --no-tei
--mlx-model <upstream all-MiniLM-L6-v2> --out <dir>` at the docs/metal.md
sizes: 32 x 256, 8 x 128, 2 x 128, 1 x 128 and 1 x 32. One record each
(`<b>x<s>.json`); `run.log` has the commit, time, host load and GPU
utilization before each.

This run is for correctness only. The host's 1-minute load was 26 to 36
and desktop applications held the GPU 25 to 60% busy, so the times in
the records are not a speed result; the timing against main and MLX on
an idle M2 is a follow-up.

| Rows | FASTEST min cosine against the bundle's reference | FASTEST max abs diff | MLX float16 min cosine against the same reference |
|---|---|---|---|
| 32 x 256 | 0.99999959 | 1.33e-4 | 0.99999913 |
| 8 x 128 | 0.99999959 | 1.33e-4 | 0.99999876 |
| 2 x 128 | 0.99999959 | 1.33e-4 | 0.99999887 |
| 1 x 128 | 0.99999959 | 1.33e-4 | 0.99999881 |
| 1 x 32 | 0.99999959 | 1.33e-4 | 0.99999900 |

FASTEST's figures are the record's conformance check against the
bundle's reference cases, the same at every size; MLX's are its vectors
on the measured rows against each row's expected vector. Both are far
inside FASTEST's 0.999 floor.
