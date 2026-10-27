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
  as on the CPU. An encoder layer with a linear layer's weight past
  F16's range (65504) computes in F32 at FASTEST, as CUDA computes such
  a model, with a warning in the log naming how many layers; the other
  layers run in F16. When every layer has one, the session computes in
  F32 and reports it in `compute_dtype`, and a `TURBO_METAL_CHOICES`
  line that forces an F16 tiling is refused
  (`TURBO_E_UNSUPPORTED_OPTION`, field 3). A cell is SUPPORTED where a benchmark record in
  `benchmarks/records/` backs it (docs/benchmarks.md): on the M2, the
  records against TEI's router built natively with Metal, FASTEST's
  against TEI 1.8.3 in float16. Elsewhere it is EXPERIMENTAL. At FASTEST,
  32 rows of 256 tokens take 18.1 ms p50 on mixed rows against TEI's
  623 ms round trip (float16), and 100.2 ms on dense rows against TEI's
  1083 ms; Speed on the M2 has the comparison with MLX. At MODEL and
  EXACT on the M2, one 16-token request at a time runs in 1.69 ms p50
  against TEI's 10.1 ms. For 32-row batches the ratio there is about
  50x on mixed rows and 8-9x on dense, because TEI's router runs a
  32-row request as about 3 sequential sub-batches: its own
  `x-inference-time` p50 was 609 to 723 ms on mixed and 1471 to 1549 ms
  on dense, so its time is its
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
  read where they are. The narrowing kernel flags each encoder layer
  with a finite weight that F16 cannot hold, and that layer's F16
  weights go unused; when every layer is flagged, the copy is dropped.
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
  of x as F16, up to 64 tokens, where it is the faster in F16; past that
  a threadgroup's tile is 32 x 32, 32 x 64, 64 x 64 or 128 x 64 (Kernel
  choices), staged in steps of 32 along k as F16, 16 bytes a thread at a
  time, so its operands take half the threadgroup memory of the F32
  kernel's. In the wider tiles the weights' part is staged transposed, so
  their SIMD-group matrices load without a transpose; the 32 x 32 tile
  stages it as it lies and loads each lane's two values of a fragment
  itself. The LayerNorms also write the residual
  stream in F16, which the Q, K and V projections and the feed-forward
  input read, and Q, K, V, the attention context and the feed-forward
  block's middle are F16 between kernels. Attention takes
  32 queries of a row and one head per threadgroup, and goes through the
  row's keys in chunks of 32 staged in threadgroup memory, with a running
  softmax; at FASTEST it stages K and V in F16 (K transposed, so its
  SIMD-group matrices load without a transpose) and each chunk's mask as
  an additive bias, keeps its queries in registers and runs the softmax
  in base 2, with each group's scores and probabilities in threadgroup
  memory or in the SIMD-group matrices' registers (Kernel choices). At
  FASTEST a head width over 64 or not a multiple of 8, up to 128, runs
  the same F16 kernel built for heads of up to 128, with the head padded
  to a multiple of 8 by zeros in threadgroup memory and, when its width
  is not a multiple of 4, read from device memory a value at a time. At
  EXACT and MODEL, in a FASTEST layer computed in F32, and past 128 at
  any precision, such a head runs a narrower kernel in F32: each query's
  scores one key per lane of a SIMD group, the block's 32 queries shared
  out over as many SIMD groups as the threadgroup memory holds rows of
  the longest row's scores (up to 32; one at a time ran long enough at
  large shapes for macOS to stop the command buffer, #1406). The run waits for the pass to
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
  1 - 4.5e-7, and bge-base-en-v1.5's (12 layers, 768 wide) 1 - 3.1e-7.
  Q, K, V, the context, GELU's output and the residual stream's copy
  are stored in F16, whose largest finite value is 65504. On the M2, 8
  rows of up to 512 tokens of bge-base (text, rows that end early, and
  ids spread over the vocabulary) reach 1 - 1.2e-6 cosine with the CPU at
  EXACT, and the largest magnitudes FASTEST stores are 22.5 (Q), 12.4
  (K), 10.3 (V), 3.3 (the context), 51.8 (GELU's output) and 102.6 (the
  residual stream's copy): under 0.2% of F16's range. all-MiniLM-L6-v2
  at 256 tokens stays under 31. The test that measures them,
  `a_real_bundle_at_its_longest_rows_stays_inside_f16`, fails a model
  whose largest stored magnitude reaches 65504.
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

## DeBERTa-v2

A model of `FAMILY_DEBERTA_V2` (docs/bundle.md; GLiNER's encoder, #883)
loads and runs at MODEL and EXACT, in F32; FASTEST is refused with
`TURBO_E_UNSUPPORTED_OPTION` naming field 3, and never runs in F32 in its
place. The linear layers, the LayerNorms after them and the pooling are
the BERT encoder's; three kernels differ (kernels.metal, DeBERTa-v2),
each held to the CPU encoder (core/src/deberta), the reference:

- `deberta_embed`: the word row alone, LayerNormed, or 0 for a masked
  position; no position or type rows.
- The relative rows, made once a model by its first session: the
  relative table LayerNormed (`deberta_layer_norm_rows`), then put
  through each layer's query and key projections with their biases
  (`deberta_relative`, a thread an output, the sum in the CPU's order),
  `[2 * position_buckets, 2 * hidden]` a layer. A session holds the
  bucket of every distance its rows can span (HF's log buckets, in F32
  logarithms, as the CPU takes them).
- `deberta_attention`: a threadgroup per block of 32 queries of a row and
  head, as many SIMD groups as the threadgroup memory holds rows of the
  longest row's scores, each taking every groups-th query, a key a lane.
  A live query's score of a live key is (q . k + q . pos_key[b] +
  k . pos_query[b]) / sqrt(3 head_dim), b the bucket of i - j, over live
  keys; a masked query attends evenly to every position of its row.

`core/tests/metal_deberta.rs` holds the device to the CPU at MODEL on the
shapes of `core/tests/common/deberta.rs` (the tiny model, one at
GLiNER-small's size, heads of 32 and 128, and steep buckets), at seven
row patterns and token ranges: on the M2 every value is within 8.3e-6 of
the CPU's, against a bound of 1e-4. Dropping the content-to-position term
fails it (3.3e-4 on the tiny model).

## Kernel choices

At FASTEST a session chooses two items, which tile its linear layers
run in past 64 packed tokens and, for heads up to 128,
which kernel runs attention, and reports them in
`turbo_session_get_info`'s `choices` (docs/autotune.md):

```
linear=auto;attn=auto;forced=
```

`linear=h64` is the 64 x 64 tile, each of its four SIMD groups 32 x 32
of it; `linear=h32` the 32 x 64 tile, each of four groups 16 x 32;
`linear=h128` the 128 x 64 tile, each of eight groups 32 x 32, on a
device that runs threadgroups of 256; and `linear=h32n32` the 32 x 32
tile, after the fragment handling of MLX's steel GEMM: each of four
groups holds four 8 x 8 fragments spread across the tile, 16 rows and 16
columns apart, loaded lane by lane from threadgroup memory as F32, with
the weights staged as they lie, so nothing is transposed on the way.
`linear=auto` picks per run: the 32 x 32 tile up to 1024 packed tokens,
where its extra threadgroups fill the GPU, and the 64 x 64 tile above.
On the M2 (all-minilm-l6-v2, six alternating rounds, FASTEST against
main's `auto`, which ran the 32 x 64 tile up to 1024 tokens) the 32 x 32
tile takes 13.8% off a run at 1 x 128 tokens, 8.8% at 2 x 128, 7.2% at
4 x 128 and 8.5% at 8 x 128; from 2048 tokens up it was level with or
slower than the 64 x 64 tile, so `auto` keeps that there. All are F16
operands with F32 sums (`TURBO_NUMERIC_F16_F32ACC`); the tile can change
how many parts a small layer's k is split into, so a different choice
moves the bits within FASTEST's floor.

`attn=staged` passes each SIMD group's 8 x 32 scores and probabilities
through threadgroup memory, a query's row to four lanes; `attn=registers`
keeps them in the SIMD-group matrices' registers, each lane holding two
columns of a row in each 8 x 8 block, so it needs 6 KiB less threadgroup
memory and no barrier between the scores and the product with V.
`attn=auto` picks per run: staged up to 512 packed tokens and registers
above. On the M2 (all-minilm-l6-v2, six alternating rounds) registers is
3 to 4% slower at 2 and 4 rows of 128 tokens, 0.2 to 0.7% faster at 8,
16 and 32 rows of 128, and 1.6% faster at 32 rows of 256. Both are F16
operands with F32 sums; they add a row's scores in different orders, so
the two move the bits within FASTEST's floor.

`forced=` names `linear`, `attn` or both when the environment fixed them.
EXACT and MODEL have one path, the F32 kernels, and report no choices. A
session nothing forces or tunes runs `auto` for both.

`TURBO_METAL_CHOICES`, read when a session is made, forces the items it
names (`linear=h32`, `linear=h64`, `linear=h128`, `linear=h32n32`,
`linear=auto`, or
`linear=f32` at EXACT and MODEL; `attn=staged`, `attn=registers` or
`attn=auto`), and a session with any item forced is not tuned; a line a
session reported, `forced=` and all, forces the same kernels back. A
kernel outside the precision's numerics (`f32` at FASTEST, an F16 tile
or any `attn` at EXACT or MODEL), `attn` for heads the F16 attention
does not take, or `h128` on a device that does not run it, is
`TURBO_E_UNSUPPORTED_OPTION` naming field 3; any other item is
`TURBO_E_INVALID_ARGUMENT`.

### Autotuning

A FASTEST session made with `turbo_session_desc.tuning` ON or RETUNE
(or `TURBO_AUTOTUNE=on` or `retune`) times each choice at each token bin
it can hold when it is made, and runs the incumbent unless another
choice is faster at every one; docs/autotune.md has the switches, the
cache and what holds across backends. It is off by default, and a forced
choice is never timed. The tuner times the linear layers' tiles alone:
attention runs `attn=auto` while it does, since the two attention
kernels are closer than the 5% a challenger must win by.

How: on the session's own memory and queue, at 256, 1024, 4096 and 8192
packed tokens, each that the session's `max_batch` x `max_seq` holds,
and at its capacity (up to 8192) when that lies past the last of them:
rows of `max_seq` tokens or fewer, ids 1, as many as make the count. The
counts go smallest first. At each, the incumbent (the cached choice,
else `auto`) and each challenger still in run the whole encoder once
untimed, then up to five times timed from the host, each run waited for:
as many as the budget left shares out over the counts and choices left,
judged by the incumbent's untimed run, and two at least. A challenger
stays in only while its least time is 5% under the incumbent's at every
count so far, since it must be at all of them to win, so the larger,
costlier counts time only the incumbent and the challengers left, and
none left settles the choice there. A challenger replaces the incumbent
only when both were timed at every count and it is 5% faster at each; of
those, the one with the least total wins. So `auto`, which picks its tile
per run, stays unless one fixed tile is the faster at every size the
session runs. A choice whose median is more than 25% above its least is
timed again, up to three times in all; still apart, the GPU is shared or
throttling, and the session keeps the incumbent, reports `default`, is
not cached, and an INFO line says so. The budget counts the timed runs
alone; past it, what is left is not timed, so a count it cuts off leaves
the incumbent in place. A measured session reports `measured`, gives its
timings as `<bin>/linear/<choice>=<least ms>` lines, one per count and
choice timed, and logs at INFO `metal device <n>: linear layers chosen in
<ms> ms: <choice> (a run at le256: auto <ms> ms h32 <ms> ms ...; le1k:
...)`, ending with why any larger count was not timed.

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

### The F16 tile

The table above is FASTEST's first F16 kernels. Since then the weights'
part of each tile is staged transposed, the k loop is unrolled, the tile
is chosen per run (`linear=auto`), and the small kernel runs only up to
64 tokens. The same timing, all-minilm-l6-v2 dense rows, five rounds on
an M2 with load averages from 3.6 down to 2.1, median p50 / least p50 in
ms, FASTEST before these changes against after, and MLX 0.29.3 float16:

| Rows | FASTEST before | FASTEST | MLX float16 |
|---|---|---|---|
| 32 x 256 | 109.4 / 109.4 | 100.6 / 100.4 | 101.5 / 101.5 |
| 8 x 128 | 13.71 / 13.68 | 12.39 / 12.38 | 11.68 / 11.67 |
| 2 x 128 | 4.79 / 4.77 | 3.94 / 3.92 | 3.79 / 3.78 |
| 1 x 128 | 2.79 / 2.78 | 2.70 / 2.67 | 2.41 / 2.41 |
| 1 x 32 | 1.28 / 1.27 | 1.27 / 1.26 | 1.95 / 1.95 |

FASTEST is level with MLX float16 at 32 x 256 and ahead at 1 x 32, and
MLX is 4% to 12% faster from 2 to 8 rows of 128. On those rows the
linear layers are about three quarters of the time; at 8 x 128 they
run at about 2.2 TFLOPS, against MLX's 3 TFLOPS for its larger products.

### The 32 x 32 tile

`linear=h32n32`, which `auto` runs up to 1024 packed tokens (Kernel
choices). all-minilm-l6-v2 dense rows, `turbo-bench record --device metal
--precision fastest --warmup 5 --iterations 50` and `mlx_bench.py` float16,
six rounds on an M2, each started once the 5-minute load average was
under 3 (the 1-minute average reached 8 in the first rounds), the order of
the two trees alternating by round; median p50 / least p50 in ms, main
before the tile (`auto` on the 32 x 64 tile up to 1024 tokens) against
after:

| Rows | FASTEST before | FASTEST | MLX float16 |
|---|---|---|---|
| 32 x 256 | 102.53 / 99.87 | 101.84 / 99.83 | 107.54 / 105.52 |
| 32 x 128 | 46.45 / 45.51 | 46.98 / 45.46 | 47.72 / 46.39 |
| 16 x 128 | 24.25 / 22.98 | 23.72 / 23.32 | 23.68 / 22.68 |
| 8 x 128 | 12.61 / 12.50 | 11.54 / 11.50 | 12.14 / 11.82 |
| 4 x 128 | 6.67 / 6.52 | 6.19 / 5.99 | 6.47 / 6.34 |
| 2 x 128 | 3.88 / 3.67 | 3.54 / 3.41 | 3.77 / 3.72 |
| 1 x 128 | 2.68 / 2.65 | 2.31 / 2.30 | 2.48 / 2.45 |

From 1 to 8 rows of 128 FASTEST is now ahead of MLX float16; from 2048
tokens up the kernels are the same as before, so those rows differ by
noise alone, and MLX's larger products still run at about 3 TFLOPS
against our 2.5.

### The linear layers alone against MLX

`bench/metal-peers/gemm_time.swift` compiles `core/metal/kernels.metal`
as the backend does (safe math) and times one dispatch of a linear-layer
kernel per command buffer; `bench/metal-peers/mlx_gemm.py` times one MLX
0.29.3 `x @ w` per `mx.eval` (version refused if another; the SHA-256 of
`mlx/lib/libmlx.dylib` and `mlx.metallib` recorded). Each is twenty timed
runs after one untimed, every sample kept, with least, mean, median and
nearest-rank p99. The comparison is `mode=bare` on both sides: F16
inputs, F32 sums (our SIMD-group matrices are `simdgroup_float8x8`, as
MLX's steel GEMM sums float16 products in float32) and F16 output, no
bias or activation, no k split, and Q, K and V as one product of n = 3h.
Our weights are `[n, k]`, MLX's `[k, n]`, so any difference includes the
layout. Both figures are host wall time from submission to completion,
so each includes its side's launch and synchronization; our GPU time is
in the raw files and is not compared with MLX's host time.

Receipt: `docs/receipts/metal-gemm-812/` (`turbo.jsonl`, `mlx.jsonl`,
`schedule.jsonl`, `summary.json`, `kernels-check.log`), at source commit
`618fbee963d4`. Five paired rounds on the M2, interleaved shape by shape
(`GEMM_ONLY`), the side that goes first alternating by shape and round,
so neither side's runs sit at another GPU temperature; `schedule.jsonl`
gives each run's start and end, the 1, 5 and 15-minute load and the
GPU's utilization before and after, the macOS build, the binary's and
scripts' SHA-256 and the commands. The 1-minute load before a run was
3.2 to 4.2 and the GPU 0 to 53% busy (desktop apps). `kernels-check.log`
is `metal_f16` at the same commit, every F16 choice within FASTEST's
floor of the CPU.

Ours (`linear=auto`'s tile) and MLX, least / median of medians / mean of
means / largest p99 over the five rounds, in ms, the ratio of the
medians, and our fastest tile by median:

| h | M | shape | ours | MLX | ours / MLX | our fastest tile |
|---|---|---|---|---|---|---|
| 384 | 128 | qkv | 0.191 / 0.224 / 0.261 / 1.084 | 0.191 / 0.224 / 0.279 / 1.210 | 1.00 | h64 0.217 |
| 384 | 128 | attn_out | 0.168 / 0.205 / 0.211 / 0.348 | 0.237 / 0.263 / 0.306 / 1.602 | 0.78 | h32n32 0.219 |
| 384 | 128 | ffn_in | 0.208 / 0.232 / 0.244 / 0.688 | 0.342 / 0.370 / 0.416 / 1.204 | 0.63 | h32n32 0.237 |
| 384 | 128 | ffn_out | 0.207 / 0.232 / 0.278 / 0.564 | 0.280 / 0.534 / 0.578 / 1.692 | 0.43 | h32n32 0.231 |
| 384 | 512 | qkv | 0.328 / 0.356 / 0.389 / 1.320 | 0.317 / 0.761 / 0.791 / 1.762 | 0.47 | h32n32 0.349 |
| 384 | 512 | attn_out | 0.203 / 0.232 / 0.248 / 0.642 | 0.333 / 0.366 / 0.409 / 1.328 | 0.64 | h32n32 0.232 |
| 384 | 512 | ffn_in | 0.388 / 0.407 / 0.438 / 1.067 | 0.369 / 0.902 / 0.924 / 1.846 | 0.45 | h32n32 0.402 |
| 384 | 512 | ffn_out | 0.357 / 0.396 / 0.418 / 0.864 | 0.399 / 0.992 / 0.930 / 2.019 | 0.40 | h32n32 0.406 |
| 384 | 1024 | qkv | 0.499 / 0.534 / 0.569 / 1.135 | 0.429 / 1.162 / 1.270 / 24.411 | 0.46 | h32n32 0.537 |
| 384 | 1024 | attn_out | 0.268 / 0.287 / 0.300 / 0.852 | 0.483 / 0.524 / 0.574 / 1.806 | 0.55 | h32n32 0.286 |
| 384 | 1024 | ffn_in | 0.626 / 0.687 / 0.854 / 1.848 | 0.544 / 1.422 / 1.242 / 2.373 | 0.48 | h32n32 0.699 |
| 384 | 1024 | ffn_out | 0.597 / 0.663 / 0.723 / 1.272 | 0.586 / 1.178 / 1.162 / 3.243 | 0.56 | h32n32 0.678 |
| 384 | 4096 | qkv | 1.607 / 1.647 / 1.676 / 2.106 | 1.288 / 1.365 / 1.995 / 5.455 | 1.21 | h32n32 1.585 |
| 384 | 4096 | attn_out | 0.646 / 0.717 / 0.728 / 1.192 | 0.541 / 1.427 / 1.171 / 3.160 | 0.50 | h32n32 0.676 |
| 384 | 4096 | ffn_in | 2.008 / 2.109 / 2.137 / 2.358 | 1.655 / 1.741 / 2.432 / 6.123 | 1.21 | h32n32 2.036 |
| 384 | 4096 | ffn_out | 2.028 / 2.087 / 2.126 / 2.719 | 1.676 / 1.813 / 2.159 / 6.457 | 1.15 | h32n32 1.964 |
| 384 | 16384 | qkv | 5.777 / 5.936 / 6.179 / 27.287 | 4.528 / 4.635 / 4.843 / 8.814 | 1.28 | h32n32 5.656 |
| 384 | 16384 | attn_out | 2.079 / 2.151 / 2.501 / 28.449 | 1.636 / 1.751 / 2.126 / 6.462 | 1.23 | h32n32 2.050 |
| 384 | 16384 | ffn_in | 7.657 / 7.759 / 7.875 / 8.748 | 6.001 / 6.122 / 6.328 / 9.329 | 1.27 | h32n32 7.457 |
| 384 | 16384 | ffn_out | 7.306 / 7.392 / 7.511 / 8.327 | 5.942 / 6.053 / 6.220 / 7.663 | 1.22 | h32n32 7.031 |
| 768 | 128 | qkv | 0.325 / 0.342 / 0.356 / 0.586 | 0.318 / 0.383 / 0.433 / 1.464 | 0.89 | h32n32 0.349 |
| 768 | 128 | attn_out | 0.203 / 0.220 / 0.231 / 0.538 | 0.419 / 0.451 / 0.521 / 1.564 | 0.49 | h32n32 0.233 |
| 768 | 128 | ffn_in | 0.384 / 0.410 / 0.446 / 0.971 | 0.362 / 0.915 / 0.878 / 2.677 | 0.45 | h32n32 0.430 |
| 768 | 128 | ffn_out | 0.379 / 0.412 / 0.430 / 0.784 | 0.504 / 1.370 / 1.217 / 3.382 | 0.30 | h32n32 0.412 |
| 768 | 512 | qkv | 0.830 / 0.893 / 0.927 / 1.685 | 0.754 / 1.516 / 1.444 / 3.852 | 0.59 | h32n32 0.892 |
| 768 | 512 | attn_out | 0.368 / 0.396 / 0.413 / 1.229 | 0.378 / 0.910 / 0.922 / 1.746 | 0.44 | h32n32 0.395 |
| 768 | 512 | ffn_in | 1.051 / 1.114 / 1.145 / 1.764 | 0.904 / 1.223 / 1.560 / 4.082 | 0.91 | h32n32 1.133 |
| 768 | 512 | ffn_out | 1.050 / 1.104 / 1.131 / 1.815 | 0.998 / 1.134 / 1.654 / 3.808 | 0.97 | h32n32 1.110 |
| 768 | 1024 | qkv | 1.509 / 1.564 / 1.703 / 3.252 | 1.272 / 1.405 / 1.772 / 4.262 | 1.11 | h32n32 1.571 |
| 768 | 1024 | attn_out | 0.589 / 0.653 / 0.669 / 1.106 | 0.535 / 0.757 / 1.033 / 2.333 | 0.86 | h32n32 0.656 |
| 768 | 1024 | ffn_in | 1.904 / 2.009 / 2.034 / 2.624 | 1.655 / 1.738 / 2.152 / 5.483 | 1.16 | h32n32 2.024 |
| 768 | 1024 | ffn_out | 1.888 / 1.941 / 1.982 / 2.632 | 1.777 / 1.833 / 2.157 / 6.084 | 1.06 | h32n32 1.945 |
| 768 | 4096 | qkv | 5.584 / 5.722 / 6.460 / 34.252 | 4.514 / 4.635 / 4.829 / 8.143 | 1.23 | h32n32 5.427 |
| 768 | 4096 | attn_out | 2.036 / 2.078 / 2.112 / 2.818 | 1.666 / 1.736 / 2.236 / 5.412 | 1.20 | h32n32 1.984 |
| 768 | 4096 | ffn_in | 7.376 / 7.484 / 7.714 / 12.796 | 5.935 / 6.059 / 6.329 / 9.745 | 1.24 | h32n32 7.239 |
| 768 | 4096 | ffn_out | 7.302 / 7.379 / 7.431 / 7.946 | 5.996 / 6.119 / 6.271 / 8.132 | 1.21 | h32n32 7.005 |
| 768 | 16384 | qkv | 21.707 / 21.864 / 21.998 / 23.481 | 17.392 / 17.667 / 17.896 / 21.879 | 1.24 | h32n32 20.960 |
| 768 | 16384 | attn_out | 7.383 / 7.506 / 7.752 / 16.824 | 5.907 / 6.055 / 6.339 / 13.153 | 1.24 | h32n32 7.168 |
| 768 | 16384 | ffn_in | 28.854 / 29.100 / 29.673 / 32.933 | 23.158 / 23.490 / 23.759 / 26.343 | 1.24 | h32n32 27.887 |
| 768 | 16384 | ffn_out | 28.348 / 28.843 / 32.223 / 49.137 | 23.022 / 23.408 / 27.294 / 48.013 | 1.23 | h32n32 27.543 |

Only at M = 16384 does the product outweigh each side's launch and
synchronization: there ours is 1.22x to 1.28x MLX's median, 2.25 to 2.68
TFLOPS against 2.76 to 3.30. Below that, from 128 to 4096 tokens, the
ratios (0.30 to 1.24) mostly measure each side's per-call overhead, which
for MLX includes Python and a graph per call, not the tiles. Earlier
whole-sweep rounds, in which our sweep ran about twenty minutes and
MLX's three, put the h = 768 rows at 1.9x to 2.1x; interleaved, the same
shapes are 1.20x to 1.24x, so that gap was the GPU's state late in a
long sweep, not the tile. At 4096 and 16384 tokens the 32 x 32 tile is
the fastest by median in every row, 3% to 6% under `auto`'s 64 x 64,
which `auto` picks past 1024 tokens (#849's per-bin tuning).

### MLX as a recorded reference

`turbo-bench record` on metal runs MLX beside TEI (docs/benchmarks.md,
Reference programs), so a record holds the MLX comparison. The two dense
records in `benchmarks/records/` (`m2.metal.embed.*-dense.*.3484a9d48f34`)
were made from a release build of that commit:

```
cargo build --release -p turbo-bench --features metal
target/release/turbo-bench record --bundle <all-minilm-l6-v2> --device metal \
    --precision <fastest|model> --rows dense --batch 32 --seq 256 --warmup 5 --iterations 50 \
    --tei-bin <text-embeddings-router> --tei-model <upstream model> --mlx-model <upstream model>
```

The M2 is a desktop and was in use. Each record started once the
5-minute load average was under 3.5 and the GPU's utilization had read
under 35% six times over 30 s (16 to 19% just before), with the
desktop's apps running; MLX's `procedure` gives the load averages just
before it ran (2.95 3.35 4.87 and 2.20 3.02 4.60). The library, TEI and
MLX run one after another within each record, about a minute apart, so
the load can differ between them. min / mean / p50 / p99 in ms; MLX's
min and mean are in its `procedure`, and its time is host wall time
over the encoder and pooling, with the rows already on the device and
the vectors not read back, where the library's includes writing the
rows and reading the vectors:

| Precision | turbo | MLX (wall time) | TEI end to end p50 / p99 | MLX's lowest cosine |
|---|---|---|---|---|
| FASTEST (F16; MLX float16) | 97.59 / 98.96 / 98.74 / 101.57 | 102.95 / 104.90 / 104.84 / 107.65 | 1109.9 / 1202.7 | 0.9999991 |
| MODEL (F32; MLX float32) | 142.29 / 145.51 / 145.27 / 149.38 | 150.22 / 152.90 / 152.27 / 159.40 | 1151.0 / 1233.8 | 1 - 2.7e-13 |

Each record's `speed_ratio`, against MLX, its fastest reference, is
0.94 at FASTEST and 0.95 at MODEL. MLX is pinned by the SHA-256 of
`mlx/lib/libmlx.dylib` (`mlx@sha256:2ea12324...`), with `mlx.metallib`'s
and `mlx.core`'s in its `procedure`, and the runner refuses an MLX other
than 0.29.3.

### Wide heads and weights past F16's range

Six-layer BERTs written by the tests' `bert_weights` (weights scaled by
0.05, no real model of either kind is among the sealed bundles), dense
rows, 8 untimed then 32 timed runs from the host on an M2, p50 / least
in ms; main from before these changes and after, two rounds each in
alternating order (the machine's 5-minute load average 3 to 5):

| Model | Rows | FASTEST before | FASTEST | EXACT |
|---|---|---|---|---|
| hidden 768, heads of 128 | 32 x 256 | 1968 / 1352 | 518 / 507 | 2286 / 2048 |
| hidden 768, heads of 128 | 8 x 128 | 161 / 154 | 59.2 / 58.3 | 165 / 163 |
| hidden 768, heads of 128 | 1 x 128 | 35.7 / 32.7 | 8.05 / 7.67 | 36.5 / 36.2 |
| hidden 640, heads of 80 | 32 x 256 | 1436 / 1367 | 341 / 312 | 1602 / 1553 |
| hidden 640, heads of 80 | 8 x 128 | 125 / 121 | 42.0 / 40.8 | 137 / 136 |
| hidden 640, heads of 80 | 1 x 128 | 27.6 / 25.5 | 6.29 / 5.88 | 28.0 / 26.9 |
| hidden 384, layer 0 past F16 | 32 x 256 | refused | 156 / 130 | 218 / 212 |
| hidden 384, layer 0 past F16 | 8 x 128 | refused | 19.5 / 18.8 | 25.6 / 24.4 |
| hidden 384, layer 0 past F16 | 1 x 128 | refused | 3.49 / 3.19 | 4.80 / 4.37 |

FASTEST before ran these heads in the F32 narrow kernel; the F16 one is
2.6x to 4.4x faster on them (by p50 or by least). A model with one
layer past F16's range, refused at FASTEST before, now runs it about
1.4x faster than EXACT.
The kernels for heads a multiple of 8 up to 64 are unchanged:
all-minilm-l6-v2 and bge-base-en-v1.5 give main's vectors bit for bit at
MODEL and FASTEST.

### LayerNorm in registers, not kept

`embed_layer_norm` and `add_layer_norm` read each row from device memory
three times (the mean, the variance, the output). A version that held
rows of up to 1024 values in registers, 32 to a lane, read each once
and gave main's vectors bit for bit (all-minilm-l6-v2, bge-base-en-v1.5
and the tiny bundle, MODEL and FASTEST, 1 x 128 to 32 x 256). It was
slower. all-minilm-l6-v2 at FASTEST, dense rows, p50 in ms, two
`turbo-bench record` rounds in alternating order:

| Rows | main | registers |
|---|---|---|
| 32 x 256 | 104.6, 100.8 | 109.4, 105.0 |
| 8 x 128 | 12.28, 12.17 | 13.04, 12.93 |
| 2 x 128 | 3.67, 3.56 | 3.99, 4.10 |
| 1 x 128 | 2.28, 2.29 | 2.42, 2.41 |

Sized to the width (12 values to a lane for hidden 384) it was level
with main, within the rounds' spread, and at bge-base-en-v1.5's 768 the
32-value version was level as well. LayerNorm is a small part of a
layer, so the reads saved do not show, and the larger register file a
lane needs costs more than they save. `core/tests/metal_ranges.rs`
keeps a test at widths around 1024 for any later try.

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

# FASTEST at a real bundle's longest rows against the CPU, with the
# largest magnitude of each tensor it stores in F16:
TURBO_TEST_BUNDLE=<bundle-dir> cargo test --release -p turbo --features metal,internals \
    --test metal_f16 -- --include-ignored --nocapture

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
