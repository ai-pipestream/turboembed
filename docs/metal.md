# The Metal backend

The `metal` backend runs embed sessions on Apple GPUs through Metal. It is
Objective-C++ and the Metal Shading Language in `core/metal/`: the host
side is compiled by `core/build.rs` with the Xcode command line tools'
clang into a static library that libturbo links, and the kernels,
`core/metal/kernels.metal`, are built into the library as source, which
Metal compiles for the device when a context is made. The core reaches
the backend only through its `turbo_backend` table
(`include/turbo/turbo_backend.h`). It is off by default: the `metal`
feature of the `turbo` crate links it, and `turbo_version()` then lists
`metal` before `cpu` (`0.1.0 metal cpu`). Its devices come before the CPU's.

## Requirements

- macOS 14 or later on Apple silicon (the M1 and later, Metal's Apple7
  family and up), where the GPU shares the host's memory. The linear
  layers and attention use SIMD-group matrices, which that family has,
  and read their elements the way Metal 3.1, which macOS has from 14 on,
  allows.
- A model whose hidden and intermediate widths are multiples of 8. A
  model that is not is refused at load with `TURBO_E_UNSUPPORTED`.
- The Xcode command line tools, for clang, the macOS SDK and `xcrun`.
  Xcode itself is not needed, nor is its offline `metal` compiler.

A Mac whose GPU has memory of its own (an AMD GPU in an Intel Mac), an
Intel Mac's integrated GPU, or a Mac on macOS before 14, is listed, its
capability cell says UNSUPPORTED with the reason, and a context on it is refused with
`TURBO_E_UNSUPPORTED` and the same reason.

## Building

```
cargo build -p turbo --release --features metal
```

The build fails on any target but macOS, arm64 or x86_64. The host side
is compiled for macOS 11 and later, and links the Metal and Foundation
frameworks and libc++, which every macOS has.

## What it does

- **Devices.** One per device `MTLCopyAllDevices` reports, probed once
  per process so an ordinal names the same device on every call.
  `device_info` is read on each call: the name Metal gives, `IGPU` with
  `unified_memory` on Apple silicon, `memory_total` the working set Metal
  recommends for the device, and `memory_free` what is left of it after
  this process's Metal allocations and no more than the host has free
  (free and inactive pages); on a GPU with its own memory, where Metal
  says nothing of other processes, it is 0 for unknown. `arch`, the label
  benchmark records are filed under, is the chip in lower case without
  spaces: `Apple M2` is `m2`, `Apple M2 Pro` is `m2pro`. `vendor` is the
  name's first word. `runtime_version` is the SDK the build compiled
  against (`Metal, macOS SDK 27.0`); `driver_version` is the macOS
  version and build, since Metal ships with it (`macOS 27.0 (26A428)`).
- **Capability.** Embed runs at every precision, honoring every field
  of `turbo_embed_options`: in F32 at EXACT and MODEL, and in F16 at
  FASTEST, whose linear layers take F16 operands (Numerics). A model
  stored in F16 or BF16 computes from an F32 copy at EXACT and FASTEST;
  its session at MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3),
  as on the CPU. A model with a linear layer's weight past F16's range
  (65504) computes in F32 at FASTEST too, as the CUDA backend does, with a
  warning in the log, and `turbo_session_get_info` reports F32 for it. A cell is SUPPORTED where a benchmark record in
  `benchmarks/records/` backs it (docs/benchmarks.md): on the M2, the
  MODEL and EXACT records against TEI's router built natively with
  Metal. Elsewhere it is EXPERIMENTAL, and so is the M2's FASTEST cell
  until a record of FASTEST in F16 backs it (Speed on the M2 has its
  times). On the M2, one 16-token request at a time runs in 1.69 ms
  p50 against TEI's 10.1 ms. For 32-row batches the ratio is about 50x
  on mixed rows and 8-9x on dense, because TEI's router runs a 32-row
  request as about 3 sequential sub-batches: its own `x-inference-time`
  p50 was 609 ms on mixed and 1618 ms on dense, so its time is its
  forward passes on candle Metal (77 to 1067 ms per sub-batch), with
  HTTP and queueing under 25 ms of it. That forward grows steeply with
  the padded batch, 3 ms for one 12-token row against 400-900 ms for a
  sub-batch of about 10 rows padded to 256, so the 32-row ratio measures
  TEI's forward at that shape, not its router.
- **Contexts.** A context is a command queue on its device and the
  kernels, compiled from their source without fast math, so `exp`,
  `sqrt` and division are the precise ones: with the macOS 15 SDK or
  later and on macOS 15 or later, by `MTLMathModeSafe`, else by turning
  fast math off. The first context on a device compiles them, and every
  later one in the process uses that compilation. The context's queue,
  for widening weights and reading `DEVICE` buffers, is used under the
  context's lock; runs go on their sessions' own queues.
- **Buffers.** All four placements are the one memory. `DEVICE` is a
  private Metal buffer, with no host address; `PINNED` and `SHARED` are
  shared Metal buffers, one address for the host and the device; `HOST`
  is host memory aligned to 64 bytes, with no Metal buffer. Import takes
  a `TURBO_HANDLE_MTL_BUFFER` (handle the `id<MTLBuffer>`, aux its
  `id<MTLDevice>`, which must be the context's) as `DEVICE` when the
  buffer is private and as `PINNED` or `SHARED` when it is shared, and a
  `TURBO_HANDLE_HOST_PTR` as `HOST`, or as `PINNED` or `SHARED` when it is
  whole pages, which Metal then maps without a copy. Export gives the
  Metal buffer of every placement but `HOST`, and the host pointer of
  every one but `DEVICE`. Reading a `DEVICE` buffer back goes through one
  shared staging buffer per context, kept at the size of the largest
  read. Any other kind is `TURBO_E_UNSUPPORTED`, naming
  it.
- **Models.** Loading copies nothing: the pages the core holds the
  weights in are mapped as shared Metal buffers, one per run of tensors
  that lie together (a weights file's), and each tensor is read where it
  is. Metal documents mapping for memory from `vm_allocate` or `mmap`;
  the core's weights are an aligned heap allocation, which Metal maps
  on the macOS versions tested (`core/tests/metal.rs` checks that nothing
  is copied, so a release that stops mapping it fails that test). A run of
  tensors Metal will not map is copied once into a shared buffer of its
  own, and the context's log says how many bytes. An F16 or BF16
  model's F32 copy is made on the device by the first session that
  computes in F32, shared by every later one, and freed with the model.
  The first FASTEST session makes the linear layers' weights in F16 the
  same way, narrowed on the device from the F32 ones, shared by every
  later FASTEST session and freed with the model; an F16 model's are
  read where they are. The narrowing kernel flags a finite weight that
  F16 cannot hold, and then the copy is dropped and FASTEST runs the F32
  kernels for that model.
- **Sessions.** Every byte a run touches is allocated when the session is
  made, for its `max_batch` rows of `max_seq` tokens: private scratch,
  shared memory for the rows, the shared buffer the vectors are written
  to, and a command queue of the session's own, so sessions on one
  context run at the same time, each on the GPU cores the others leave
  free. `embed_write` packs the rows into the session's shared memory,
  where the GPU reads them: each row's columns up to its last live token,
  one row after another, with each token's column for its position
  embedding. Padding past a row's last live token is never computed; no
  output depends on it. The run encodes the encoder into one compute pass
  over the packed tokens: the embedding lookup and its LayerNorm in one
  kernel; per layer the Q, K and V projections in one dispatch, with
  their biases added as they are stored; attention; the attention
  output; a residual and LayerNorm; the feed-forward input with its bias
  and GELU added as it is stored; the feed-forward output; a residual and
  LayerNorm; then one kernel pools (mean over the mask, the first token,
  or the last live one), cuts to `output_dim` and normalizes. The linear
  layers run one of two GEMM kernels. Up to 128 packed tokens, where
  reading the weights is most of the work, and for an output width that
  is a multiple of 16, a threadgroup computes a 32 x
  16 tile of the output over all of k, its four SIMD groups each taking
  every fourth step of 8 along k and reading straight from device
  memory, and sums their four parts as it stores: many small
  threadgroups keep many weight reads in flight. Above that, four SIMD
  groups per 32 x 64 tile of the output, each 16 x 32 of it in SIMD-group
  matrices, go over k in steps of 16 staged in threadgroup memory, and a
  layer that would still leave most of the GPU idle has its k split
  across more threadgroups, the residual kernel summing the parts. At
  FASTEST the same two shapes run with F16 weights: the few-token kernel
  reads them straight from device memory and stages each group's 32 rows
  of x as F16; past 128 tokens a threadgroup's tile is 32 x 64, 64 x 64
  or 128 x 64 (Kernel choices), staged in steps of 32 along k as F16, 16
  bytes a thread at a time, so its operands take half the threadgroup
  memory of the F32 kernel's. The LayerNorms also write the residual
  stream in F16, which the Q, K and V projections and the feed-forward
  input read, and Q, K, V, the attention context and the feed-forward
  block's middle are F16 between kernels. Attention takes
  32 queries of a row and one head per threadgroup, and goes through the
  row's keys in chunks of 32 staged in threadgroup memory, with a running
  softmax; at FASTEST it stages K and V in F16, holds its queries in
  registers and runs the softmax in base 2. A head width that is not a
  multiple of 8, or is over 64, runs a narrower kernel, one query at a
  time, in F32 at every precision. The run waits for the pass to
  complete before it returns, and leaves the vectors in the session's
  `SHARED` buffer, which `turbo_result_buffer` hands out and
  `turbo_result_read` copies from.
- **Session limits.** Beyond the model's own, a session whose scratch is
  more than one Metal buffer holds is `TURBO_E_UNSUPPORTED_OPTION` naming
  field 1. For a model whose heads run the narrow attention kernel, a
  `max_seq` whose scores do not fit the threadgroup memory the device
  gives (4 bytes per token; 32 KiB on Apple GPUs, so about 8000 tokens)
  names field 2. The device is a GPU, so `cpu_threads` (field 6 of
  `turbo_session_desc`) is 0 alone: any other value is
  `TURBO_E_UNSUPPORTED_OPTION` naming field 6.
- **Range pooling.** `turbo_embed_write_ranges` packs the rows as
  `embed_write` does and copies the ranges, three `uint32` each as
  `turbo_token_range` lays them out, into a region of the same shared
  memory after the attention blocks, sized for the session's
  `max_ranges` when the session is made. The GPU reads them there, so
  nothing else is copied and nothing crosses. The run encodes every row
  once, as for `embed_write`, and then a second pooling kernel, one SIMD
  group per range, writes one vector per range in range order: the mean
  of the final hidden states over the range's columns whose mask is 1,
  cut to `output_dim`, then normalized when `normalize` says so. A range
  stops at its row's last live token, since the columns after it are
  padding that is never computed. The sum runs in position order with the
  row pooling's arithmetic, so a range over a whole row gives that row's
  vector bit for bit. The session's output buffer holds
  max(`max_batch`, `max_ranges`) vectors of `hidden` F32, made with the
  session, so a range run allocates nothing either.
- **Numerics.** F32 throughout. Apple GPUs have no F64, so where the CPU
  encoder sums in F64 this backend sums in F32: LayerNorm takes the mean,
  then the variance around it, in two passes, and the L2 norm is summed
  in F32 and floored at 1e-12. Softmax works from the largest live score,
  kept as it goes through the keys, and mean pooling sums in position
  order, as on the CPU. Metal has no `erf`; GELU's comes from erfc's
  Chebyshev fit in Numerical Recipes, with relative error under 1.2e-7.
  Cosine against the fp32 reference must reach 0.9999. At FASTEST the
  linear layers take F16 operands, the weights narrowed once and x
  rounded to F16 (the residual stream as the LayerNorm writes it, the
  rest as the kernel before stores it), and sum in F32 SIMD-group
  matrices; the bias and GELU are added in F32 and the result rounded to
  its F16 or F32 store. For head widths the matrix kernel takes,
  attention's scores and its product with V take F16 operands, the
  queries scaled by 1 / (sqrt(head width) ln 2) in F16 so the softmax is
  base 2 on the precise `exp2`, and the softmax, its running max and sum,
  and every sum are F32. The residual stream itself, LayerNorm and pooling
  are F32 as at EXACT. Cosine must reach 0.999, FASTEST's floor on every
  backend; all-MiniLM-L6-v2's reference cases (up to 256 tokens) reach
  1 - 4.5e-7. Q, K, V, the context and GELU's output are stored in F16
  with no guard for values past its range, and a larger model at 512
  tokens has not been checked yet.
- **What a result reports.** Stages: tokenize on the host for text,
  no upload (the rows were copied into the session's shared memory by
  the write, as on the CPU), lookup, encode and pool on the device,
  normalize fused into the pooling kernel when it runs, and no download. With one memory nothing crosses to a
  device: `h2d_bytes` is 0, and `d2h_bytes` is 0 after the run and grows
  by each read, as turbo.h counts reads. `host_allocs` and
  `device_allocs` are what the backend allocated on the running thread
  during the run, counted where it allocates; a run allocates nothing,
  cold or warm. The command buffer and encoder each run submits are
  Metal's own objects, made by Metal for every submission, and are not
  counted.

## Kernel choices

At FASTEST a session chooses one item, which tile its linear layers run
in past 128 packed tokens, and reports it in `turbo_session_get_info`'s
`choices` (docs/autotune.md):

```
linear=h64;forced=
```

`linear=h64` is the 64 x 64 tile, each of its four SIMD groups 32 x 32
of it; `linear=h32` the 32 x 64 tile, each of four groups 16 x 32; and
`linear=h128` the 128 x 64 tile, each of eight groups 32 x 32, on a
device that runs threadgroups of 256. All three are F16 operands with
F32 sums (`TURBO_NUMERIC_F16_F32ACC`); the tile can change how many
parts a small layer's k is split into, so a different choice moves the
bits within FASTEST's floor. `forced=` names `linear`
when the environment fixed it. EXACT and MODEL have one path, the F32
kernels, and report no choices, as does FASTEST for a model with a weight
past F16's range. A session nothing forces or tunes runs
`h64`.

`TURBO_METAL_CHOICES`, read when a session is made, forces the item it
names (`linear=h32`, `linear=h64`, `linear=h128`, or `linear=f32` at
EXACT and MODEL);
a line a session reported, `forced=` and all, forces the same kernels
back. A kernel outside the precision's numerics (`f32` at FASTEST, an F16
tile at EXACT or MODEL), or `h128` on a device that does not run it, is
`TURBO_E_UNSUPPORTED_OPTION` naming field 3, and so is an F16 tile forced
on a model with a weight past F16's range; any other item is
`TURBO_E_INVALID_ARGUMENT`.

### Autotuning

A FASTEST session made with `turbo_session_desc.tuning` ON or RETUNE
(or `TURBO_AUTOTUNE=on` or `retune`) times each tile when it is made and
runs the fastest; docs/autotune.md has the switches, the cache and what
holds across backends. It is off by default, and a forced choice is
never timed.

How: on the session's own memory and queue, with rows of `max_seq`
tokens, ids 1, as many as fill 8192 tokens (all of the session's rows
when it holds fewer). Each tile, the incumbent (the cached choice, else
`h64`) first, runs the whole encoder once untimed, then five times timed
from the host, each run waited for. The least time ranks, and the
fastest challenger replaces the incumbent only when 5% faster. A tile whose median is more than 25%
above its least is timed again, up to three times in all; still apart,
the GPU is shared or throttling, and the session keeps the incumbent,
reports `default`, is not cached, and an INFO line says so. The budget
counts the timed runs alone; past it, what is left is not timed. A
measured session reports `measured` and logs at INFO `metal device <n>:
linear layers chosen in <ms> ms: <choice> (<bin>, a run: h64 <ms> ms,
h32 <ms> ms, h128 <ms> ms)`.

## Speed on the M2

all-minilm-l6-v2 (bundle manifest 3c305237), release builds,
`turbo-bench record --device metal --warmup 5 --iterations 50` for
turbo and `bench/metal-peers/mlx_bench.py` (MLX 0.29.3, the same encoder
written on MLX, `mx.compile`d, 5 untimed then 50 timed runs) for MLX.
Five rounds, each running every cell once in turn: this backend at
FASTEST (F16) and EXACT, the backend before FASTEST computed in F16 (F32
at FASTEST, built from the same main with the macOS build fix), MLX in
float16 and float32. The machine was in use, not idle (load averages from
41 down to 5 over the rounds), so the table gives the median of the five
p50s and, after the slash, the least p50, in ms. MLX runs dense rows
only; it computes a padded batch, so mixed rows would time its padding.
`mlx_bench.py` casts every weight, LayerNorm's included, to the dtype it
runs in, so MLX float16 is F16 throughout, a looser contract than
FASTEST's F32 residual stream, LayerNorm and sums.

| Rows | FASTEST, F16 | FASTEST before, F32 | EXACT | MLX float16 | MLX float32 |
|---|---|---|---|---|---|
| 32 x 256 dense | 109.2 / 109.1 | 147.1 / 139.9 | 180.2 / 140.1 | 121.5 / 100.9 | 183.3 / 145.4 |
| 32 x 256 mixed | 22.5 / 19.6 | 26.6 / 23.9 | 26.8 / 24.1 | | |
| 8 x 128 dense | 13.9 / 13.5 | 17.9 / 17.0 | 17.7 / 17.0 | 11.8 / 11.6 | 16.5 / 15.7 |
| 1 x 128 dense | 2.78 / 2.74 | 3.21 / 3.04 | 3.18 / 3.15 | 2.48 / 2.45 | 2.85 / 2.81 |

At 32 x 256, FASTEST in F16 is 1.28x the F32 path by the least p50 and
1.35x by the median, and 0.93x MLX float16's best p50; MLX float16 is
faster at 8 and 1 rows. F16 halves the bytes the linear layers and
attention move and stage, but the M2's GPU does F16 and F32 multiply-adds
at one rate, so the gain is the memory traffic, not the arithmetic. On
those 32 x 256 rows the encoder's linear layers take about 78 ms and
attention about 27 ms; MLX's own matrix products for the same four
shapes take about 59 to 72 ms on the same machine. Vectors at FASTEST
reach 1 - 4.5e-7 cosine against the bundle's reference. EXACT runs the
same F32 kernels as before this change: its least p50 at 32 x 256 (140.1)
matches the F32 FASTEST path's (139.9), and its higher median is the
machine's load in the later rounds.

## Testing

From the workspace root. `TURBO_TEST_REQUIRE_METAL=1` makes every test in
`core/tests/metal*.rs` that needs a device that runs embed fail when there
is none, where without it the test passes with a line saying it was
skipped; set it on Apple silicon, so a run that found no GPU cannot pass.

```
export TURBO_TEST_REQUIRE_METAL=1

# Everything, with the Metal-only tests: core/tests/metal.rs (devices,
# buffers, weights), metal_runs.rs (the encoder's vectors), metal_ranges.rs
# (late pooling, limits, the largest shape) and metal_f16.rs (FASTEST's
# F16 linear layers and their choices):
cargo test -p turbo --features metal

# The runs and FASTEST's tests, with what they print (the largest
# difference from the f64 encoder and the CPU, each choice's lowest
# cosine with the CPU), in release:
cargo test --release -p turbo --features metal --test metal_runs --test metal_f16 -- --nocapture

# Conformance on the small sealed bundle (docs/conformance.md):
TURBO_TEST_DEVICE=metal cargo test --release -p turbo --features metal --test conformance -- --nocapture

# On a real bundle, made with turbo-bundle (bundle/README.md): conformance,
# and one run at the bundle's max_batch x max_seq against the CPU.
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=metal \
    cargo test --release -p turbo --features metal --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features metal --test metal_ranges -- --include-ignored --nocapture
```

`TURBO_TEST_DEVICE=metal` picks the first device the metal backend lists,
and the conformance test fails when it lists none. The tests in
`core/tests/metal.rs` that need no device that runs embed (the table, the
device listing) run everywhere the feature builds.
