# The CUDA backend

The `cuda` backend runs embed sessions on NVIDIA GPUs through the CUDA
runtime, with GEMMs of its own; it links cuBLAS to measure them against
and to check them in the tests. It is C++ and CUDA in `core/cuda/`,
compiled by `core/build.rs` with nvcc into a static library that libturbo
links, and the core reaches it only through its `turbo_backend` table
(`include/turbo/turbo_backend.h`). It is off by default: the `cuda`
feature of the `turbo` crate links it, and `turbo_version()` then says
`0.1.0 cuda cpu`. Its devices come before the CPU's.

## Requirements

- Linux, with the NVIDIA driver loaded and a GPU it lists.
- The CUDA toolkit, 12.x or 13.x: nvcc, the runtime's and cuBLAS's
  headers, and `libcudart.so` and `libcublas.so` in its library
  directory. The driver must support the toolkit's CUDA version (for
  13.x, a 580 driver or newer; for 12.x, 525 or newer).
- A host C++ compiler nvcc accepts (gcc or clang, C++17).
- A GPU whose architecture the build targets. The default is sm_89, an
  RTX 4080 (sm_89) for example. The highest architecture built also
  carries its PTX, which a newer GPU compiles when the library loads; an
  older one is listed, and its capability cell says UNSUPPORTED with the
  reason.

## Building

```
export TURBO_CUDA_ROOT=/usr/local/cuda        # the toolkit's directory
cargo build -p turbo --release --features cuda
```

| Variable | Meaning |
|---|---|
| `TURBO_CUDA_ROOT` | The toolkit's directory, with `bin/nvcc`, `include/` and the libraries in `lib64/`, `lib/`, `targets/<arch>-linux/lib/` or `lib/<arch>-linux-gnu/`. A distribution's packaged toolkit (nvcc in `/usr/bin`) works with `TURBO_CUDA_ROOT=/usr`. Unset: `CUDA_PATH`, then `CUDA_HOME`, then `/usr/local/cuda`. |
| `TURBO_CUDA_ARCH` | The SM architectures to compile for, comma separated as nvcc numbers them: `89`, or `86,89,120`. Default `89`. |
| `NVCC_CCBIN` | nvcc's own: the host compiler it runs, when the default is not one it accepts. |

At run time, `TURBO_CUDA_CUBLAS`, read when a session is made, hands the
GEMMs it names to cuBLAS, for measuring the backend's own against it: a
comma-separated list of `qkv`, `out`, `ffn1` and `ffn2` (the attention
output and the two feed-forward GEMMs), or `all`. cuBLAS's product then
goes through a kernel doing the same epilogue, and such a session runs
its launches one by one instead of as a graph. Unset, cuBLAS computes
nothing. `TURBO_CUDA_TILE`, read the same way, picks the GEMMs' output
tile for every GEMM: `64x64`, `128x64`, `128x128` or `128x128-16x8`
(128 × 128 over 128 threads of 16 × 8 outputs each for the FMA
kernels, where the other tiles give a thread 8 × 8; plain `128x128` on
the tensor cores). Unset, the F32 kernels take `128x128-16x8`, the F16
FMA kernels `128x64`, and the tensor-core kernels `128x128` for the QKV
and first feed-forward GEMMs and `128x64` for the other two. A tile shares the
work among the blocks at other points, so the vectors agree within the
precision's bound, not bit for bit; only the time should differ.
`TURBO_CUDA_ATTENTION=split`, read the same way, gives the sessions
that compute attention with FMAs the kernel that splits each query's
keys among four warps (a lane per query, 64 queries to a block, the
partial softmaxes merged in a fixed order), for measuring against the
default. `TURBO_CUDA_SK_STEPS`, read the same way, is the fewest k steps
a GEMM's block takes before the GEMM runs on fewer blocks (a count from
1 to 64; 4 when unset), for measuring how finely the work is shared.
Like the tile, it moves where the sums split, so the vectors agree
within the bound, not bit for bit.

A GEMM's block that finishes a tile waits for the blocks that computed
its other k steps. Should one never arrive (the device running a later
block before an earlier one could leave it waiting), the wait gives up
after about a second, and the run fails with `TURBO_E_RUNTIME` instead
of hanging; a session whose GEMM kernels fit fewer blocks to an SM than
they were built for logs a warning when it is made.

The library links the toolkit's shared `libcudart.so.<major>` and
`libcublas.so.<major>`, with the toolkit's library directory as its run
path. That run path covers this package's own library and tests only: a
binary elsewhere that links the `turbo` rlib with the feature finds the
libraries through `LD_LIBRARY_PATH` or a run path of its own. A machine
that runs it needs those libraries, from the toolkit or a CUDA runtime
install of the same major version; a build without the feature needs
none of them. Without a driver, or with no GPU, the
backend lists no device and the runtime is made as usual. With a driver
older than the runtime, it lists none and the runtime's log says why.

## What it does

- **Devices.** One per GPU the runtime counts, in its order.
  `device_info` is read on each call: name, memory total and free now
  (`cudaMemGetInfo`), `unified_memory` for an integrated GPU, the
  runtime's version, and the driver's as its kernel module version and
  the CUDA version it supports (`580.82.07, CUDA 13.0`). `arch`, the
  label benchmark records are filed under, is the device name's model,
  derived by one rule: the name's words without the vendor's and brand's
  (NVIDIA, GeForce, Tesla, Quadro, GPU, Generation), up to a memory size
  or form factor (80GB, PCIe, SXM4, HBM3, NVL), in lower case letters and
  digits. `NVIDIA GeForce RTX 4080` is `rtx4080`, `NVIDIA A100-SXM4-80GB`
  is `a100`, `NVIDIA GeForce RTX 4070 Ti SUPER` is `rtx4070tisuper`,
  `NVIDIA RTX 6000 Ada Generation` is `rtx6000ada`.
- **Capability.** Embed is EXPERIMENTAL at every precision, honoring
  every field of `turbo_embed_options`. MODEL and EXACT compute in F32;
  FASTEST computes in F16: its GEMMs take F16 weights and activations and
  accumulate in F32 on the tensor cores, attention takes F16 queries,
  keys and values the same way, and everything else (the hidden states,
  residuals, LayerNorm, softmax, pooling) stays F32. The cell says F32
  or F16 accordingly. A model stored in F16 or BF16 computes in F32 from
  a converted copy at EXACT; its session at
  MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3), as on the
  CPU. A model with a GEMM weight past F16's range (65504) computes in
  F32 at FASTEST too, with a warning in the log, and
  `turbo_session_get_info` reports F32 for it.
- **Contexts.** A context is a stream and a cuBLAS handle on its device,
  with a 32 MiB cuBLAS workspace of its own, so a GEMM `TURBO_CUDA_CUBLAS`
  hands to cuBLAS never allocates.
  The stream, the handle and its workspace are used under the context's
  lock. The calling thread's current device is set for each call and put
  back after it.
- **Buffers.** `DEVICE` is `cudaMalloc` memory, `PINNED` page-locked host
  memory (`cudaMallocHost`), `HOST` pageable memory aligned to 64 bytes,
  `SHARED` managed memory (`cudaMallocManaged`). Import takes a
  `TURBO_HANDLE_CUDA_PTR` (aux the device ordinal, which must be the
  context's) as `DEVICE` or `SHARED`, and a `TURBO_HANDLE_HOST_PTR` as
  `HOST`, or as `PINNED` or `SHARED` when the memory is page-locked or
  managed; the driver is asked what the memory is. Export gives the
  CUDA pointer of `DEVICE` and `SHARED` buffers and the host pointer of
  `HOST`, `PINNED` and `SHARED` ones. Any other kind is
  `TURBO_E_UNSUPPORTED`, naming it.
- **Models.** Loading copies every tensor, in the dtype it is stored in,
  into one device allocation, each layer's Q, K and V weights back to
  back so one GEMM makes the three projections. An F16 or BF16 model's
  F32 copy is made on the device by its first session (an F16 session
  reads its biases, LayerNorms and embeddings from it too), shared by
  every later one, and freed with the model. An F32 or BF16 model's GEMM
  weights are rounded to F16 into one more allocation by its first
  FASTEST session, laid out the same way and shared the same way; an F16
  model's FASTEST sessions read its own weights. These copies are made
  by `turbo_session_create`, never by a run.
- **Sessions.** Every byte a run touches is allocated when the session is
  made, for its `max_batch` rows of `max_seq` tokens: device scratch, the
  output buffer and page-locked staging for the rows; and the run is
  captured into a CUDA graph, instantiated and uploaded to the device.
  `embed_write` hands the rows over laid out at the run's own width,
  each array's rows back to back and the arrays one after another, so
  only the entries the caller passed are sent. When every array is in
  page-locked (a `PINNED` buffer's) or managed memory with its rows back
  to back, each goes to the device in one contiguous copy from there,
  and the write waits for them, since that memory is the caller's again
  when the call returns. Otherwise the write copies them into the
  session's page-locked staging on the host, and the run's first kernel
  reads them from there over the bus: nothing is queued on the stream
  ahead of the run's graph.
  Rows in device memory are refused (`TURBO_E_INVALID_ARGUMENT`): the core
  reads and checks every row on the host before the backend sees it.
  The run computes the rows packed, as the CPU does: each row's
  positions up to its last live token, one row after another, so padding
  past that token is never computed by any GEMM, LayerNorm, GELU or
  attention. Positions are the row's own column indices; attention skips
  masked keys, and no pooling reads past the last live token, so no
  vector changes. It waits for the stream before it returns, and leaves
  the vectors in the session's `DEVICE` buffer: `turbo_result_buffer`
  hands out that memory, and `turbo_result_read` copies it back on the
  context's stream.
- **Pipeline.** A run is one CUDA graph launch and one wait at its end.
  The graph holds 4 + 7 × layers kernels (46 for MiniLM's 6 layers),
  captured when the session is made and never again: nothing about a
  run's shape is a launch argument. The first kernel, `fetch_rows`,
  brings the rows from the staging when the write left them there. The
  second, `pack_rows`, reads the mask and writes the run's packing to
  device memory: each row's length (1 + its last live position) and
  first packed token, the packed token count, each token's row, a key
  bias of -1e30 for masked tokens inside a row, and the rows ordered by
  length, longest first, with the attention work items each starts. The
  two kernels' arguments, the rows' shape, the embed options and the
  entries to fetch, are set in the graph
  (`cudaGraphExecKernelNodeSetParams`) before a launch where they differ
  from the last one's; nothing is copied for them. Every later kernel
  reads the token count there and is launched for the session's
  `max_batch` × `max_seq` tokens, capped at what the device holds at
  once, its blocks looping over the work there is and leaving when there
  is none. Then the embedding lookup and its LayerNorm, a warp per
  packed token. Per layer:
  1. one GEMM of the Q, K and V projections together, `[tokens, hidden]`
     by `[3 × hidden, hidden]`, their biases added in its epilogue, each
     head's written apart (`[3][heads][tokens][head_dim]`) so attention
     reads a row's keys contiguously;
  2. attention, rows longest first: the row's keys and values for the
     head go through shared memory, the whole row at once up to a chunk
     (64 keys for the FMA kernel, 256 on the tensor cores), the softmax
     carried from one block of keys to the next by its running largest
     score (flash attention's rescaling), with no mask read but the key
     bias of a row that has masked tokens. EXACT and MODEL take 32
     queries of one head of one row to a block of 128 threads and compute
     QKᵀ and then PV as register tiles, each thread 4 queries by 4 keys
     and then 4 queries by head_dim / 16 values, every sum in the order
     of the head's values or the keys' positions. FASTEST with heads of
     32 or 64 computes QKᵀ and PV with `mma.sync` on the tensor cores, 64
     queries to a block, F32 accumulators, softmax in F32;
  3. the attention output GEMM;
  4. its bias, the residual and LayerNorm in one kernel, a warp per token
     holding its row in registers (writing an F16 copy too for an F16
     session);
  5. the feed-forward input GEMM, its bias and GELU (erf) in its
     epilogue;
  6. the feed-forward output GEMM;
  7. its bias, the residual and LayerNorm, as in 4.

  Last, one kernel pools each row, a block of 384 threads per row (mean
  over its mask, its first token, or its last live one), cuts to
  `output_dim` and normalizes.
  The GEMMs are the backend's own, their bias, GELU and head-major
  layout applied in the epilogue. At FASTEST they take F16 operands with
  `mma.sync.m16n8k16` and F32 accumulators, 32 values of k to a step
  through a `cp.async` pipeline, eight warps to a block and two blocks
  to an SM: 128 × 128 tiles over two stages, each warp 32 × 64, for the
  QKV and first feed-forward GEMMs, and 128 × 64 tiles over three, each
  warp 32 × 32, for the attention output and second feed-forward GEMMs,
  whose outputs are a third or a quarter as wide. The outputs are staged
  through shared memory to be stored 16 bytes at a time. At MODEL and
  EXACT they take F32 operands with F32 FMAs (no TF32), 128 × 128
  tiles over 128 threads, each thread 16 × 8 outputs (so it reads 24
  values from shared memory per 128 FMAs, where 8 × 8 reads 16 per 64),
  one block to an SM, 16 values of k to a step through a three-stage
  `cp.async` pipeline (see `TURBO_CUDA_TILE` for the other tiles).
  Devices before sm_80 take the FMA kernels at every precision, F16 at
  FASTEST with 8 × 8 outputs of a 128 × 64 tile. The token count changes with every batch, so no
  fixed tiling fills the device; each GEMM is scheduled stream-K
  instead. It launches as many blocks as the device holds at once and
  gives each an equal, contiguous share of the work, counted as tiles ×
  steps of k. A tile whose steps fall to two or more blocks is finished
  by the block holding its last step, which adds the others' partial
  products from a workspace, in a fixed order, and runs the epilogue.
  Each block works through its share from the end, so the partial
  products it owes come first, and a block only waits on blocks started
  before it. Every kernel of the graph asks for the same split of the
  SMs' memory, the most shared memory, so the device need not change it
  between kernels.
- **Session limits.** Beyond the model's own, a `max_batch` over 65535
  is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 1). A row may be as long
  as the model's positions: attention streams longer rows' keys through
  shared memory a chunk at a time. A model is refused when it is loaded
  (`TURBO_E_UNSUPPORTED`) for a head wider than 64 values, a hidden width
  over 2048, or a hidden or intermediate width not a multiple of 8.
- **Numerics.** An F32 session is F32 throughout: every GEMM output is
  a sum, in a fixed order, of chains of F32 FMAs over consecutive steps
  of k. An F16 session rounds its GEMMs' and attention's inputs to F16
  and accumulates in F32. The arithmetic follows the CPU encoder where
  order matters: LayerNorm takes the mean, then the variance about it
  (in F32 here, F64 on the CPU), softmax subtracts the largest live
  score (a running one, rescaling what was summed before it, where the
  CPU takes the row's largest first), mean pooling sums in position
  order, the L2 norm is summed in F32 and floored at 1e-12. No reduction
  uses atomics. Where the GEMMs split k among blocks depends on the
  packed token count and the number of blocks the device holds, so the
  same rows in the same batch on the same device give the same bits,
  and results agree across batches and devices within the precision's
  bound. (The packing counts rows into length bins with integer atomics;
  that changes which block computes a row, never what it computes.)
  Against the fp32 reference, F32 must reach cosine 0.9999 and a largest
  absolute difference of 1e-4, F16 cosine 0.999 (docs/conformance.md).
- **What a result reports.** Stages: tokenize on the host for text,
  upload, lookup, encode and pool on the device, normalize fused into the
  pooling kernel when it runs, and no download: the vectors stay where
  they are. `h2d_bytes` is the rows sent by the write, 4 bytes per token
  for each of ids, mask and (when given) types. `d2h_bytes` is 0 after the
  run and grows by each read. `host_allocs` and `device_allocs` are what
  the backend allocated on the running thread during the run, counted
  where it allocates; a run allocates nothing, cold or warm. The first
  run of a shape may still load kernels the driver has not loaded yet
  (CUDA loads modules lazily); `core/tests/cuda.rs` holds warm runs to an
  unchanged free-memory figure too.

## Testing on a GPU machine

With `TURBO_CUDA_ROOT` set as above, from the workspace root.
`TURBO_TEST_REQUIRE_CUDA=1` makes every test in `core/tests/cuda.rs` that
needs a device fail when the backend lists none, where without it the
test passes with a line saying it was skipped; set it on a GPU machine,
so a run that found no GPU cannot pass.

```
export TURBO_TEST_REQUIRE_CUDA=1

# Everything, with the CUDA-only tests in core/tests/cuda.rs:
cargo test -p turbo --features cuda

# The CUDA-only tests, with what they print (the device, the largest
# difference from the f64 encoder and the CPU), in release too for the
# largest shape:
cargo test --release -p turbo --features cuda --test cuda -- --nocapture

# Conformance on the small sealed bundle (docs/conformance.md):
TURBO_TEST_DEVICE=cuda cargo test --release -p turbo --features cuda --test conformance -- --nocapture
# ... at FASTEST (F16, held to the F16 bound) and EXACT:
TURBO_TEST_DEVICE=cuda TURBO_TEST_PRECISION=fastest \
    cargo test --release -p turbo --features cuda --test conformance -- --nocapture
TURBO_TEST_DEVICE=cuda TURBO_TEST_PRECISION=exact \
    cargo test --release -p turbo --features cuda --test conformance -- --nocapture

# On a real bundle, made with turbo-bundle (bundle/README.md): conformance,
# and one run at the bundle's max_batch x max_seq against the CPU.
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=cuda \
    cargo test --release -p turbo --features cuda --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features cuda --test cuda -- --include-ignored --nocapture
```

`TURBO_TEST_DEVICE=cuda` picks the first device the cuda backend lists,
and the conformance test fails when it lists none. The tests in
`core/tests/cuda.rs` that need no device (the table, the arch labels)
run everywhere the feature builds.

## Recording a benchmark

Embed stays EXPERIMENTAL here until a benchmark record backs the cell:
one made by `turbo-bench` on the GPU, with TensorRT and
text-embeddings-inference measured on the same token rows
(docs/benchmarks.md, which has the command).
