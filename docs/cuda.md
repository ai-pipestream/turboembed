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
the tensor cores), and for the tensor cores `128x128-4w` (four warps of
64 × 64), `256x128` (eight such warps; `128x128-4w` for a GEMM with an
F32 output) or `8w`, the eight-warp tiles, FASTEST's F32-sums shapes: `128x128`
over warps of 32 × 64 for the QKV and first feed-forward GEMMs and
`128x64` for the other two (the FMA kernels take `128x64` for these
three). FASTEST's default is `ctk`, F16 sums over a block's whole k on
CUTLASS's mainloop, described with the `f16k` tiles below. F16 on the tensor cores also takes `sw`, `sw8w` and `sw256`, the
swizzled kernel: stage rows of 64 bytes with 16-byte chunk c of row r
at c ^ ((r >> 1) & 3) in place of rows padded to 80 bytes, so three
stages of 128 × 128 take 48 KB and two blocks share an SM; the loads
running STAGES - 1 steps ahead across the end of a tile, so the next
tile's first stages load while a tile is finished; and the epilogue from
registers (F32 a float2 a lane, F16 gathered by shuffles into 16-byte
stores), with no pass through shared memory. On warps of 64 × 64 and
32 × 96 the mainloop is software-pipelined: the fragments of the next
16 values of k are read while the MMAs of the current 16 run, the next
step's first after a step's last, so a step's barrier comes before its
last MMAs rather than before its first reads. The eight-warp mix's
shapes wait at each step's barrier instead: there the pipeline measured
slower, with or without registers to spare.
`sw` is 128 × 128 over four warps of 64 × 64 for every GEMM; `sw256` the same but 256 × 128 over eight such warps
for the first feed-forward GEMM, one block to an SM; `sw8w` the
eight-warp mix's shapes at three and four stages; `swrow` is `sw8w` but
the attention output and second feed-forward GEMMs on 64 × 384 tiles,
whole rows, over eight warps of 32 × 96, one block to an SM: their
epilogue adds the bias and the residual and runs the LayerNorm on the
rows in registers (each lane's values of a row summed, then the quad's
by shuffles, then the four warps across the row in warp order through
shared memory; the mean, then the variance about it), writing the F32
hidden states and their F16 copy once, with no LayerNorm kernel after
it. It sums in another order than the separate kernel, so its vectors
agree with the default's within FASTEST's bound, not bit for bit; a
hidden width over 384 takes `sw8w`. The other precisions take `128x64`
for these. Unset, the FMA kernels and TF32 take `128x64`,
and F16 on the tensor cores `8w`. On an RTX 4080 at 32 × 256, `8w` is faster than the
four-warp tiles (`128x128-4w` with `256x128` for the first feed-forward
GEMM): about 0.73 against 0.83 ms on mixed rows and 3.5 against 3.9 ms
on full rows. EXACT is about 4% slower with `128x128-16x8` than with
`128x64`.
A tile shares the
work among the blocks at other points, so the vectors agree within the
precision's bound, not bit for bit; only the time should differ.
`TURBO_CUDA_ATTENTION=split`, read the same way, gives the sessions
that compute attention with FMAs the kernel that splits each query's
keys among four warps (a lane per query, 64 queries to a block, the
partial softmaxes merged in a fixed order), for measuring against the
default. FASTEST's attention on the tensor cores (heads 32 or 64 wide)
runs 128 queries to a block of eight warps, so each chunk of keys and
values in shared memory serves 128 queries; the keys and values go 64
at a time through three buffers filled by `cp.async`, the next two
chunks loading while this one's products and softmax run, with one
barrier to a chunk, and a block loads its next item's queries and first
two chunks during an item's last. A chunk whose 64 keys are all the
row's and none masked (every chunk of a full row, all but the last of
others) takes a path that tests no key against the row's end or the
mask. The softmax is taken in base 2: the running maximum is of the
unscaled scores, and each probability is `ex2.approx` of one fused
multiply-add, the score times the scale times log2 e less the
maximum's. `ex2.approx` is within 2 ulp of F32, below the F16 rounding
the probabilities take for P V. `TURBO_CUDA_ATTENTION=exact` keeps the
earlier softmax, the scores scaled before the maximum is taken and
`exp2f` for each probability, which gives the earlier bits; the default
agrees with it within FASTEST's bound. Heads of 32 fit two blocks to an
SM, heads of 64 one. `TURBO_CUDA_ATTENTION=fa32`, for measuring, gives
heads of 32 the same kernel with two tiles of 16 queries to a warp over
four warps, so each fragment of keys and values read from shared memory
feeds four products in place of two, at two blocks of four warps to an
SM (shared memory holds it to two). It gives the default's bits; heads
of 64 keep the default. On an RTX 4080 SUPER at 32 x 256 full rows the
kernel, with two buffers and `exp2f`, took 324 µs a pass against 398
for the earlier kernel of 64 queries to four warps, which
`TURBO_CUDA_ATTENTION=64` still gives; the two round differently, so
their vectors agree within FASTEST's bound, not bit for bit. `TURBO_CUDA_LAYER_NORM=fused`, read the same way, has the
N = hidden GEMMs' epilogue run the LayerNorm in place of the default's
kernel of its own after the GEMM (see Pipeline; the bits are the same
either way, and on an RTX 4080 the separate kernel is faster).
`TURBO_CUDA_F16_ACCUMULATE=1`, read the same way, gives FASTEST's GEMMs
on the tensor cores F16 accumulators (`mma.sync.m16n8k16` with an F16
C and D), `8w`'s tiles (`sw8w`'s with `TURBO_CUDA_TILE=sw8w`, and
whatever other tile it names): each 64 terms
of k are summed in F16, and each such sum is added into the F32
accumulators, in k's order, the chunks counted from k 0 so a tile's
sums do not depend on which blocks share it. F16 accumulation is twice
the tensor cores' F32 rate on GeForce cards. It is off by default; the
CUDA tests hold it to FASTEST's bound, cosine 0.999, and print the
cosine and largest absolute difference they measure.
`TURBO_CUDA_TILE=f16k` goes further: FASTEST's GEMMs sum
in F16 over the whole of a block's k, with no F32 accumulators but the
stream-K partial products and their total (TensorRT's F16 GEMMs), on the
swizzled kernel at 128 × 128 over four warps of 64 × 64, four stages, one
block to an SM; `f16k3` is the same at three stages, two blocks to an SM;
`f16krow` is `f16k3` but the attention output and second feed-forward
GEMMs as `swrow` takes them, 64 × 384 whole rows with the residual and
the LayerNorm in their epilogue (`f16k3`'s tile for hidden states wider
than 384); `f16k256` is `f16k3` but QKV and GELU at 256 × 128 over
eight warps of 64 × 64, one block to an SM.
`TURBO_CUDA_TILE=ct` runs the GEMMs' k loop on CUTLASS's sm80 threadblock
mainloop (`MmaMultistage` and its iterators, the headers under
`core/cuda/cutlass/`, BSD-3-Clause, the README's Licence section) inside a kernel of
this backend's: 128 × 128 × 32 tiles at three stages over four warps of
64 × 64, two blocks to an SM, F16 operands and F32 sums, with the
swizzled kernel's tile schedule, partial products and epilogues; `ctk`
is `ct` with F16 sums over the whole of a block's k, as `f16k3`, and
gives `f16k3`'s bits. Both need K a multiple of 8 (the mainloop's loads
are 16 bytes) and take the `8w` and `f16k3` tiles otherwise. `ctk` is
FASTEST's default (`f16k3` where K is not a multiple of 8): on an RTX
4080 SUPER it measured 12% faster than `f16k3` and 35% faster than `8w`
on dense input at a reference cosine of 0.999998, three orders of
magnitude inside FASTEST's bound. `ct` and `ctk` are candidates the
tuner measures; the other whole-k tiles are not.
Each F16 sum rounds to 11 bits all along k, so the error grows with k: on
uniform operands in [-1, 1] the CUDA tests print it against cuBLAS for
F32 sums, sums over 64 and whole-k sums side by side, and hold the last
within 1e-2 of the largest value. A tile's sums depend on where the
blocks sharing it split its k, so its bits depend on the grid (the same
from run to run on one device). The other precisions take `128x64`.
`TURBO_CUDA_POOL=columns` gives the pooling of a thread per column, for
measuring against the default. `TURBO_CUDA_SK_STEPS`, read the same way, is the fewest k steps
a GEMM's block takes before the GEMM runs on fewer blocks (a count from
1 to 64; 4 when unset), for measuring how finely the work is shared, or
`tiles`: whole tiles to a block, so no tile is split between blocks and
no block waits on another's partial product (the blocks then share the
tiles, not the k steps, evenly). Like the tile, it moves where the sums
split, so the vectors agree within the bound, not bit for bit.

A GEMM's block that finishes a tile waits for the blocks that computed
its other k steps. Should one never arrive (the device running a later
block before an earlier one could leave it waiting), the wait gives up
after about a second, and the run fails with `TURBO_E_RUNTIME` instead
of hanging; a session whose GEMM kernels fit fewer blocks to an SM than
they were built for logs a warning when it is made.

### Kernel choices

A session's kernels are chosen per bin of packed tokens (up to 256,
1024, 4096, 16384, and past 16384), and `turbo_session_get_info`
reports them in `choices` as one line, the bins the session has first:

```
le256:qkv=8w/sk4,out=8w/sk4,ffn1=8w/sk4,ffn2=8w/sk4,attn=mma128,ln=separate;le1k:...;pool=groups;forced=
```

Each GEMM (`qkv`, `out`, `ffn1`, `ffn2`) names its tile as
`TURBO_CUDA_TILE` spells it (`acc16-8w` and `acc16-sw8w` being the
F16 accumulators' eight-warp and swizzled eight-warp tiles), then its
stream-K (`sk<steps>` or `tiles`), then `/tf32` when it computes in TF32;
`attn` is `fma-tiled`, `fma-split`, `mma64`, `mma128`, `mma128-exact`
or `mma128-fa32`; `ln` is
`separate` or `fused`; `pool` is `groups` or `columns`. The names are of
the kernels that run: a tile the session's operands or device do not
take is reported as the one that runs in its place. `forced=` lists the
knobs (`tile`, `sk`, `tf32`, `attn`, `ln`, `pool`) the environment
fixed; `tuned` says `default`, `forced` when every knob was or a tuned
session's every GEMM tile was, and `measured` or `cache` for a tuned
session (below).

`TURBO_CUDA_CHOICES`, read when a session is made after the switches
above, forces the items it names, over them: the line a session
reported gives a session of the same kernels, so the same bits. It may
name fewer items (`all:qkv=sw8w/tiles` forces the first GEMM of every
bin, a GEMM's stream-K may be left out) and leaves the rest as they
were. A bin the session does not have is left out, with a DEBUG line,
so one line forces sessions of any size. An unknown item or value is
`TURBO_E_INVALID_ARGUMENT` naming it.

A kernel is allowed a precision by the numeric class it computes in:
F32 FMAs at EXACT and MODEL; at FASTEST F16 operands with F32 sums and,
by the decision of 2026-09-26 that made `ctk` the default, F16 sums
within a chunk (the `acc16-` tiles, and the `f16k` and `ctk` tiles, whose
chunk is a block's whole k), and F32 FMAs for a model past F16's range.
TF32 is in no precision's set until a decision adds it; a kernel of it
forced through `TURBO_CUDA_CHOICES` is `TURBO_E_UNSUPPORTED_OPTION`
naming field 3 and the kernel, unless its experiment's switch is set for
the session (`TURBO_CUDA_TF32=1` at MODEL), which widens that session's
set. A session computes in its precision's classes, and in another only
where a kernel it chose computes in it: a switch the line overrides
widens nothing.

### Autotuning

A session made with `turbo_session_desc.tuning` ON or RETUNE (or
`TURBO_AUTOTUNE=on` or `retune`, when the desc leaves it to the
environment) times its GEMMs' kernel variants when it is made and runs
the fastest; docs/autotune.md has the switches, the cache and what
holds across backends. It is off by default.

What is timed: each token bin's four GEMMs, each over the tiles its
precision's classes allow: at FASTEST on tensor cores `ctk`, `8w`,
`sw8w`, `ct`, `acc16-8w` and `acc16-sw8w`; on the FMA kernels `128x64`,
`128x128-16x8`, `128x128` and `64x64`; with `TURBO_CUDA_TF32=1` at
MODEL, `128x64/tf32` and `128x128/tf32` too. The other tiles, and the
`f16k` tiles, are forced only. A GEMM whose tile a switch forces is not
timed. Stream-K, attention, the LayerNorm and the pooling keep their
defaults.

How: after the session's memory is allocated and cleared, under the
context's lock, on its stream and its own buffers. Each bin, those
nearest a mixed batch first (`le4k`, `le1k`, `le16k`, `le256`,
`gt16k`), gets rows of its upper edge's tokens (the session's size when
smaller, 32768 at most past 16384) in the lengths of a mixed batch, ids 1
and types 0; the first runs the whole encoder once, which loads the
modules and brings the clocks up. Each variant of a GEMM of the first
layer is launched once untimed, then timed five times with event pairs,
and five more while the times are more than 10% apart, up to 15. The
least time ranks, and a variant replaces the incumbent (the cached
choice, else the default) only when 5% faster, so measuring again on
the same device keeps the choice. When the first incumbent's five times
have a median more than 25% above their least, it is timed again, up to three times in all,
since a cold device's clocks may still be coming up; still apart, the
device is shared or throttling: the session keeps its incumbents, is
not cached, and the log says so. A
variant that cannot launch here is not timed, and an INFO line says so
once. The graphs are captured from the choices after the memory is
cleared again.

The budget is `tuning_budget_ms`, else `TURBO_AUTOTUNE_BUDGET_MS`, else
150 ms, or 750 ms with a disk cache (`TURBO_AUTOTUNE_CACHE`); past it,
what is left is not timed, and DEBUG lines name it. The context's lock
is held that long, so the context's other sessions wait: a server that
makes sessions while it serves makes them on a context of their own, or
with tuning off. A session whose GEMMs cuBLAS computes
(`TURBO_CUDA_CUBLAS`) is not tuned and takes no cached choice: it
reports `default`, and an INFO line says why. Nor is a session whose
every GEMM's tile is forced (`TURBO_CUDA_TILE`, or `TURBO_CUDA_CHOICES`
naming each), since the tuner times only tiles: it reports `forced`,
and an INFO line says so.

A measured session reports `tuned` MEASURED and `tune_ms`, and logs at
INFO `cuda device 0: kernels chosen in <ms> ms for <n> token bins:
<choices> (<bin>: the GEMMs of a layer <ms> ms, the incumbents <ms> ms;
...; <n> not timed)`; each variant's least and median time is a DEBUG
line. A session of the same device, driver, build, bundle, precision
and size then takes the choice unmeasured, `tuned` CACHE, unless it
forces or widens something: only a measured session with nothing
forced and no experiment's class run is cached, and an INFO line says
why another measured session was not.

What the first tuned session costs, on an RTX 4080 with
all-MiniLM-L6-v2 at 32 rows of 256 tokens (bins up to 16384) and the
default 150 ms budget without a disk cache:

| Precision | `tune_ms` | Not timed |
|---|---|---|
| FASTEST | 14 ms | 0 |
| EXACT | 100 ms | 0 |

Measured with:

```sh
TURBO_TEST_REQUIRE_CUDA=1 TURBO_TEST_BUNDLE=<all-MiniLM-L6-v2 bundle> \
    cargo test --release -p turbo --features cuda --test cuda the_first_tuned_session_s_cost \
    -- --include-ignored --nocapture
```

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
  every field of `turbo_embed_options`. MODEL and EXACT compute in F32
  with F32 FMAs. `TURBO_CUDA_TF32=1`, read when a session is made, puts
  MODEL's GEMMs on the tensor cores of sm_80 and newer instead: they
  round their F32 operands to TF32 and accumulate in F32
  (`mma.sync.m16n8k8`); EXACT stays F32 FMAs throughout. It is off by
  default until measured: on an RTX 4080 the tensor cores' TF32 peak is
  the F32 FMAs' peak. Either way the session reports F32. The CUDA
  tests hold TF32 to F32's cosine, 0.9999, but not to its largest
  absolute difference of 1e-4 (docs/conformance.md), which TF32's
  10-bit operands need not reach.
  Attention at MODEL is EXACT's FMA kernel; a TF32 attention would be a
  follow-up. FASTEST computes in F16: its GEMMs take F16 weights and activations and
  accumulate in F32 on the tensor cores, attention takes F16 queries,
  keys and values the same way, and everything else (the hidden states,
  residuals, LayerNorm, softmax, pooling) stays F32. The cell says F32
  or F16 accordingly. A model stored in F16 or BF16 computes in F32 from
  a converted copy at EXACT; its session at
  MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3), as on the
  CPU. A model with a GEMM weight past F16's range (65504) computes in
  F32 at FASTEST too, with a warning in the log, its GEMMs as MODEL's
  (TF32 with `TURBO_CUDA_TF32=1`), and `turbo_session_get_info`
  reports F32 for it.
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
  captured into a CUDA graph, instantiated and uploaded to the device:
  one graph per bin of packed tokens (up to 256, 1024, 4096, 16384 and
  more; the bins past `max_batch` × `max_seq` do not exist for the
  session) whose kernel choices differ, all bins sharing one graph when
  they choose alike, as they do unless their choices were forced apart.
  With more than one graph, `embed_write` counts the packed tokens on
  the host and the run launches its bin's graph; with one, it counts
  nothing.
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
  The graph holds 4 + 5 × layers kernels (34 for MiniLM's 6 layers;
  4 + 7 × layers with the LayerNorms apart, see below),
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
  3. the attention output GEMM, with its bias, the residual and
     LayerNorm in its epilogue: each tile's finishing block adds the
     bias and the residual into the hidden states and counts the tile
     done for its block of rows; the block that finishes the last of a
     block of rows' tiles then normalizes those rows, a warp per row
     holding it in registers, two rows at a time (writing an F16 copy
     too for an F16 session). The sums are those of the separate
     LayerNorm kernel in the same order, so the bits are the same (at
     FASTEST, those of the F32 residual stream, below);
  4. the feed-forward input GEMM, its bias and GELU (erf) in its
     epilogue, `erff`, or where it writes F16 (FASTEST) erfc from a fit
     with `TURBO_CUDA_GELU=poly` (below);
  5. the feed-forward output GEMM, with its bias, the residual and
     LayerNorm, as in 3.

  That fused epilogue runs only with `TURBO_CUDA_LAYER_NORM=fused` and
  hidden widths up to 512. By default, the attention output and
  feed-forward output GEMMs take the product alone and then a kernel of
  their own for the bias, residual and LayerNorm, a warp per token.
  The fused one is slower because of its end: one block normalizes all
  of a block of rows (128) with its four or eight warps, two rows at a
  time, each pair waiting on its reads from L2 and on its reductions,
  while the separate kernel spreads the rows over every SM. That tail
  sits on each launch's critical path and costs more than the launch it
  saves: on an RTX 4080's mixed run, 0.88 ms at FASTEST against 0.73,
  1.98 at EXACT (four warps to a block) against 1.69.

  Last, one kernel pools each row, a block of 384 threads per row: a
  thread per four columns, and for the mean up to eight groups of such
  threads, each adding a contiguous run of the row's live tokens in
  position order with eight 16-byte loads in flight, the groups' sums
  then added in group order. It cuts to `output_dim` and normalizes,
  writing the vector once. `TURBO_CUDA_POOL=columns`, and hidden
  widths past 1536, take instead the kernel of a thread per column,
  adding every token in turn.
  The GEMMs are the backend's own, their bias, GELU and head-major
  layout applied in the epilogue. At FASTEST they take F16 operands with
  `mma.sync.m16n8k16` and F32 accumulators, 32 values of k to a step
  through a `cp.async` pipeline, eight warps to a block: 128 × 128 tiles
  of warps of 32 × 64 at two stages for the QKV and first feed-forward
  GEMMs, the widest, and 128 × 64 tiles of warps of 32 × 32 at three
  for the other two, two blocks to an SM either way. The outputs are staged
  through shared memory to be stored 16 bytes at a time. At MODEL with
  `TURBO_CUDA_TF32=1`, on sm_80 and newer, the same kernel takes F32
  operands, 16 values of k to
  a step (the same 64 bytes of each row), rounds each fragment it reads
  from shared memory to TF32 (`cvt.rna.tf32.f32`) and multiplies with
  `mma.sync.m16n8k8`, F32 accumulators, 128 × 64 tiles over eight warps
  of 32 × 32 and three stages, two blocks to an SM, for every GEMM
  (`TURBO_CUDA_TILE=128x128` for the other; its F32 output tile fits
  one block to an SM). Otherwise, at MODEL and EXACT, they take F32
  operands with F32 FMAs (no TF32), each thread 8 × 8 outputs of a 128 × 64 tile, 16 values of
  k to a step through a three-stage `cp.async` pipeline (see
  `TURBO_CUDA_TILE` for the other tiles). Devices before sm_80 take the
  FMA kernels at every precision.
  The token count changes with every batch, so no
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
  and accumulates in F32. Its GELU takes `erff`. `TURBO_CUDA_GELU=poly`,
  read when a session is made, gives an F16 output's GELU erf from a fit
  instead, for measuring against the default (`erf` names the default):
  0.5 x (2 - erfc(z)) for x >= 0 and 0.5 x erfc(z) below, z = |x| /
  sqrt 2, so the negative side has no cancellation, with erfc(z) =
  t P(t) exp(-z^2), t = 1 / (1 + p z), the form of Abramowitz and Stegun
  7.1.26 with a sixth term, fit to erfc's relative error (6.5e-7 for z
  up to 4). It is within 2.4e-7 of GELU for |x| <= 8, and rounded to F16
  it is GELU rounded to F16 for every F16 value there but ten that lie
  within 2e-3 of an ulp of a tie (tests/cuda_gelu.rs, on the host). An
  exponential, a reciprocal and multiply-adds, without `erff`'s branch;
  on an RTX 4080 SUPER at 32 x 256 full rows it takes the same time as
  `erff` on the default tiles and 5% less on `sw8w`'s GELU launch, whose
  `erff` instance spills registers. At FASTEST the hidden states between
  one LayerNorm and the next are kept in F16 alone, the copy the GEMMs
  read: each LayerNorm adds the GEMM's F32 product and its bias to the F16
  residual in F32, normalizes, and writes F16; the F32 hidden states are
  written after the last layer, for the pooling. `TURBO_CUDA_RESIDUAL=f32`,
  read when a session is made, keeps the F32 residual stream as well, the
  earlier bits, for measuring against the default (`f16` names the
  default); the LayerNorm epilogues (`TURBO_CUDA_LAYER_NORM=fused`, `swrow`,
  `f16krow`) read the F32 stream and keep it. On the tensor cores the
  attention output and feed-forward output GEMMs write their product
  with its bias in F16 (1.5 KB a token in place of 3), which their
  LayerNorm kernel reads in place of the F32 product and the bias;
  `TURBO_CUDA_PRODUCT=f32`, read the same way, keeps the F32 product and
  the kernel's bias, the earlier bits (`f16` names the default). A GEMM
  cuBLAS runs (`TURBO_CUDA_CUBLAS`) keeps the F32 product. The arithmetic follows the
  CPU encoder where order matters: LayerNorm takes the mean, then the
  variance about it
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

# Conformance again with its sessions tuned (tests/common's session_desc
# asks for tuning ON), at each precision as above:
TURBO_TEST_TUNING=1 TURBO_TEST_DEVICE=cuda \
    cargo test --release -p turbo --features cuda --test conformance -- --nocapture
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
