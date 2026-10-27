# The F16 linear layers alone against MLX on the M2 (#812)

The receipt behind docs/metal.md, "The linear layers alone against MLX".

- `schedule.jsonl`: first line, the source commit, the SHA-256 of
  `core/metal/kernels.metal`, of `bench/metal-peers/gemm_time.swift` and of
  its release binary (`swiftc -O`), of `bench/metal-peers/mlx_gemm.py`, the
  macOS version and build, and both commands. Then one line per side per
  shape per round: start and end (UTC), the 1, 5 and 15-minute load and the
  GPU's "Device Utilization %" before and after, and the exit status.
- `turbo.jsonl` and `mlx.jsonl`: each side's output, every timed sample
  kept, the round prefixed. Ours gives host and GPU time; MLX gives host
  time.
- `summary.json`: per shape and side over the five rounds, the least, the
  median of the rounds' medians, the mean of their means and the largest
  p99.
- `kernels-check.log`: `metal_f16` at the same commit, the timed kernels'
  F16 choices within FASTEST's floor of the CPU.

Five paired rounds, interleaved shape by shape with `GEMM_ONLY`, the side
that goes first alternating by shape and by round. Only `mode=bare` is
timed: F16 inputs, F32 sums, F16 output, no bias or activation, no split.
