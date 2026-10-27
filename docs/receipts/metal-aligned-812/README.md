# The 32 x 32 F16 tile without range checks on the M2 (#812)

The receipt behind docs/metal.md, "The 32 x 32 tile without range
checks". Code at `da1472b0`.

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
  largest p99, and the aligned tile's median over the checked one's,
  round by round.
- `kernels-check.log`: `metal_aligned` and `metal_f16` at the same
  commit, and all-minilm-l6-v2 and bge-base-en-v1.5 at their longest
  rows (1 - cosine with the CPU 1.965e-6 and 4.635e-6, the same as
  before the change).

`h32n32` is the 32 x 32 tile with its range checks, as before;
`h32n32a` is the same kernel with them compiled out, which the backend
now runs for `h32n32` (and `auto` up to 1024 tokens) when the packed
tokens, n and k are multiples of 32; `h64` is `auto`'s tile past 1024
tokens. Bare mode only: F16 x and w, F32 sums, F16 out, no bias,
activation or k split. Shapes (n, k) = (3h, h), (h, h), (4h, h) and
(h, 4h) for h = 384 and 768, at M = 128, 512, 1024, 4096 and 16384,
the three tiles in one process per shape, the order rotated by shape
and by round.

Load: rounds 1 and 2 ran with the 1-minute load under 4; in rounds 3 to
5 builds outside the project took it to 55, and desktop applications
held the GPU up to 62% busy. Each round's median of the 40 paired ratios
(aligned over checked) is 0.935 to 0.943; the 16 pairs taken with the
load under 8 and the GPU under 20% busy before and after have a median
of 0.936, every one under 1.
