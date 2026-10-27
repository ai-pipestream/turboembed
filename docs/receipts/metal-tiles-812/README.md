# The 64 x 32 F16 tile against the 32 x 32 and 64 x 64 on the M2 (#812)

The receipt behind docs/metal.md, "The 64 x 32 tile". Code at `60128d6e`.

- `schedule.jsonl`: first line, the commit, the SHA-256 of
  `core/metal/kernels.metal`, of `bench/metal-peers/gemm_time.swift` and
  of its release binary (`swiftc -O`), the macOS version and build and the
  command. Then one line per shape per round: the tile order, start and
  end (UTC), the 1, 5 and 15-minute load and the GPU's "Device
  Utilization %" before and after, and the exit status (0 for all 200).
- `turbo.jsonl`: the harness's output, every timed sample kept, the
  round prefixed. Each figure is the GPU time of one dispatch, from the
  command buffer's GPU start to its end, twenty timed after one untimed.
- `summary.json`: per shape and tile over the five rounds, the least,
  the median of the rounds' medians, the mean of their means and the
  largest p99, and the 64 x 32 tile's median over the 32 x 32's, round
  by round.
- `kernels-check.log`: `metal_f16` at the same commit, every F16 tile
  within FASTEST's floor of the CPU, on the tiny bundle, on a model whose
  widths leave partial tiles, and on all-minilm-l6-v2 and
  bge-base-en-v1.5 at their longest rows.

Bare mode only: F16 x and w, F32 sums, F16 out, no bias, activation or
k split. Shapes (n, k) = (3h, h), (h, h), (4h, h) and (h, 4h) for
h = 384 and 768, at M = 128, 512, 1024, 4096 and 16384. Each shape runs
the three tiles in one process, the order rotated by shape and by
round. The 1-minute load was 2.3 to 2.8 throughout; desktop applications
held the GPU 0 to 92% busy, which is why the shapes under M = 4096 swing
from round to round.
