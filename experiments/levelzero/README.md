# Level Zero kernel experiments

Standalone Level Zero programs for timing and checking kernels outside the
library, on an Intel GPU with the compute runtime. Each C file builds with
`gcc -O2 -o NAME NAME.c -lze_loader -lm`; each `.cl` file compiles to SPIR-V
with
`clang -cl-std=CL3.0 --target=spirv64 -O2 -mllvm --spirv-ext=+SPV_INTEL_subgroups -c -o K.spv K.cl`.
Timings are device kernel timestamps, the median over 200 launches after
200 warm ones.

- `gemm_bench.c`, `gemm_bench.sh`, `gemm_variants.cl`: F16 DPAS GEMM
  (`Y = X . Wt`, 2D block reads) with tile, group, k-step, split-K,
  prefetch and barrier variants selected by `-D` macros; the script takes
  `TM TN WM WN KSTEP LARGE_GRF shape:splits...`. `EVICT=1` copies 64 MB
  between launches so inputs come from memory, not cache.
- `gemm_16x32_pipelined.cl`, `gemm_32x64_pipelined.cl`: the same with the
  next step's operands loaded before this step's products (the 32 x 64 one
  wants `-ze-opt-large-register-file`).
- `sgemm_variants.cl`: F32 GEMM variants (shuffled or uniform A terms).
- `linear_bench.c`: times one kernel of `core/levelzero/encoder.cl`'s
  linear family (`linear_dpas`, `linear_dpas_to_half`, ...) by name.
- `layer_norm_bench.c`: times `linear_dpas_layer_norm` with random data.
- `ffn_bench.c`: runs the two-kernel feed-forward block and
  `linear_dpas_mlp` on the same data, compares the outputs and times both.
- `dpas_check.c`, `visa_dpas.cl`: a DPAS product checked against the CPU;
  `visa_dpas.cl` issues it as inline vISA, which needs the Khronos
  `llvm-spirv` translator (`clang -target spir64 -emit-llvm`, then
  `llvm-spirv --spirv-ext=+SPV_INTEL_inline_assembly,+SPV_INTEL_subgroups`).
- `tf32_layout.c`, `tf32_layout.cl`: shows which lane and element of the
  TF32 DPAS A operand land in which row and term.
- `launch_cost.c`, `launch_cost.cl`: host time of a chain of empty kernel
  launches on an in-order immediate list, with and without events.
- `igadis.c`: disassembles a raw GEN binary (such as oneDNN's
  `ONEDNN_JIT_DUMP=1` output) with the IGA library, for Xe2.
