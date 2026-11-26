# #812: the 64 x 32 F16 tile without range checks

Measured on the Apple M2 (10-core GPU, Metal, macOS 27) with MLX 0.29.3,
head `7ef2d86dd577a396dc3ef8733981d008d7be4645` against main
`7755201ffb4e918ba97b761147999c23ef3064c5`. Desktop apps were open.
Paths are scrubbed to `<scratch>`, `<head tree>`, `<base tree>`, `<head target>`, `<base target>` and
`<bundles>`.

## Files

- `e2e/`: one log per `turbo-bench record` run, 5 rounds of 8 sizes,
  main (`base`) and head each with MLX float16 in the record, the side
  order alternating by round and the size order reversed in even rounds.
  Each log has the load and GPU before and after; the record's file name
  carries the commit. `e2e/summary.md` is the table in docs/metal.md.
  The 1-minute load before each run was 1.7 to 2.7 and the GPU 0 to 6%
  busy; the 5-minute load was still falling from earlier builds.
- `gemm/`: bare mode, one shape a process, three rounds at 4096 and
  16384 tokens, h = 384 and 768, the side order rotating by shape:
  `base.jsonl` (main's `gemm_time.swift` and kernels, `auto`),
  `head.jsonl` (this tree's, `auto`, `h32n32a`, `h64n32a`), `mlx.jsonl`
  (`mlx_gemm.py`), `schedule.jsonl`, the three header lines and
  `summary.md` (median of the rounds' medians; MLX has host time only).
- `gemm/mlx-kernel-gpu.txt`: MLX's nt kernel from its metallib beside
  `h64n32a` and `h32n32a`, GPU-timed in one process on the same buffers,
  20 runs each, interleaved, median GPU ms; every output compared bit for
  bit with MLX's (no `BITDIFF`).
- `kernels-check.log`: `a_real_bundle_at_its_longest_rows_stays_inside_f16`
  in release on all-minilm-l6-v2 and bge-base-en-v1.5.
- `drivers/`: the scripts that made these files.
