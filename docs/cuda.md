# The CUDA backend

The `cuda` backend runs embed sessions on NVIDIA GPUs through the CUDA
runtime and cuBLAS. It is C++ and CUDA in `core/cuda/`, compiled by
`core/build.rs` with nvcc into a static library that libturbo links, and
the core reaches it only through its `turbo_backend` table
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
- **Capability.** Embed is EXPERIMENTAL at every precision, computing in
  F32, honoring every field of `turbo_embed_options`. A model stored in
  F16 or BF16 computes in F32 from a converted copy at EXACT and FASTEST;
  its session at MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3),
  as on the CPU.
- **Contexts.** A context is a stream and a cuBLAS handle on its device,
  with a 32 MiB cuBLAS workspace of its own, so a GEMM never allocates.
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
  into one device allocation. An F16 or BF16 model's F32 copy is made on
  the device by the first session that computes in F32, shared by every
  later one, and freed with the model.
- **Sessions.** Every byte a run touches is allocated when the session is
  made, for its `max_batch` rows of `max_seq` tokens: device scratch, the
  output buffer and page-locked staging for the rows. `embed_write` sends
  the rows to the device, from the caller's memory when it is page-locked
  (a `PINNED` buffer's) and through the staging otherwise, and waits for
  them. Rows in device memory are refused (`TURBO_E_INVALID_ARGUMENT`):
  the core reads and checks every row on the host before the backend
  sees it. The run computes on the device over the written `[batch, seq]`
  grid: the embedding lookup and its LayerNorm in one kernel; per layer
  the Q, K and V projections, the attention output and the feed-forward
  layers as `cublasSgemm`, one attention kernel (scaled dot products over
  the keys whose mask is 1, softmax from the largest score), residual and
  LayerNorm kernels, and GELU with erf; then one kernel pools (mean over
  the mask, the first token, or the last live one), cuts to `output_dim`
  and normalizes. It waits for the stream before it returns, and leaves
  the vectors in the session's `DEVICE` buffer: `turbo_result_buffer`
  hands out that memory, and `turbo_result_read` copies it back.
- **Session limits.** Beyond the model's own, two refusals come from
  the device, both `TURBO_E_UNSUPPORTED_OPTION`: a `max_batch` over 65535
  names field 1, since the kernels run one block per row and a grid
  dimension holds 65535; a `max_seq` whose attention scores do not fit
  the shared memory the device gives one block (about 4 bytes per token
  plus the head's width; 99 KiB on sm_89, so about 25000 tokens) names
  field 2.
- **Numerics.** F32 throughout, with TF32 off: the cuBLAS handle's math
  mode is `CUBLAS_DEFAULT_MATH`, which computes an F32 GEMM in F32. The
  arithmetic follows the CPU encoder where order matters: LayerNorm sums
  its mean and variance in F64, softmax subtracts the largest live score,
  mean pooling sums in position order, the L2 norm is summed in F64 and
  floored at 1e-12. Cosine against the fp32 reference must reach 0.9999.
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
