# The levelzero backend

The `levelzero` backend runs embed sessions on Intel GPUs through the
oneAPI Level Zero driver. It is Rust in `core/src/levelzero.rs` and
`core/src/levelzero/`, with its kernels in OpenCL C in
`core/levelzero/encoder.cl`, which `core/build.rs` compiles to SPIR-V and
the library carries. The core reaches it only through its `turbo_backend`
table (`include/turbo/turbo_backend.h`). It is off by default: the
`levelzero` feature of the `turbo` crate links it, and `turbo_version()`
then says `0.1.0 levelzero cpu`. Its devices come before the CPU's.

## Requirements

- To build: clang 20 or newer, whose own SPIR-V backend compiles OpenCL
  C to SPIR-V (`--target=spirv64`) and takes `--spirv-ext`, for the
  Intel sub-group extension the kernels use. The build asks clang its
  version: an older one is refused with that reason, and to clang 20,
  which otherwise hands its output to `llvm-spirv`, it adds
  `-fintegrated-objemitter` (Ubuntu 24.04's `clang-20` builds it, and
  clang 21). Nothing of Level Zero is needed at build
  time without `levelzero-onednn`.
- To run: the Level Zero loader (`libze_loader.so.1`, 1.10 or newer) and
  Intel's GPU driver for it (`libze_intel_gpu.so.1`, the compute runtime,
  at Level Zero 1.9 or newer for in-order immediate lists), with the
  kernel's `xe` or `i915` driver bound to the GPU and
  the user able to open its render node (the `render` group).
- A GPU with 2D block reads and writes (`cl_intel_subgroup_2d_block_io`:
  Xe2 and later, such as the B70, and Xe-HPC), which the kernels are
  built with. An older part, such as the Arc A-series, is listed, and
  its capability cells say UNSUPPORTED with that reason, from the IP
  version the driver gives (`ZE_extension_device_ip_version`); a driver
  that gives none leaves the cell as it would be, and the module's build
  says why no session runs.
- A GPU whose driver computes in F64: the encoder sums the L2 norm in
  F64. A device without it is listed, and its capability cell
  says UNSUPPORTED with that reason.

The loader is opened when the first runtime lists its devices, not when
the library loads, so a machine without it runs everything else as
usual and the backend lists no device.

## Building

```
cargo build -p turbo --release --features levelzero
```

| Variable | Meaning |
|---|---|
| `TURBO_CLANG` | The clang that compiles the kernels. Unset: `clang` on the `PATH`. |
| `TURBO_LEVELZERO_CHOICES` | At run time: forces the session's kernel choice (Kernel choices, below). |
| `TURBO_LEVELZERO_PROFILE` | At run time: every context times each kernel and copy on the device, and logs the totals at debug level when it is released. A profiled run allocates on the host to name what it times. |

The driver builds the SPIR-V for the device the first time a context
needs a kernel, when a model or session is made; that build is not part
of any run.

### oneDNN for the linear layers

With the `levelzero-onednn` feature, the linear layers of a batch of
more than 8 tokens may run on oneDNN's kernels instead of the backend's
own, at every precision; which ones a session runs is its `linear` choice
(Kernel choices, below). At FASTEST oneDNN runs on the matrix engines in
F16, as OpenVINO's kernels do: the Q, K and V projection, the
feed-forward input with its GELU, and the two projections back to the
hidden width with their residual added, each followed by oneDNN's
LayerNorm; the last layer's hidden states are then widened to F32 for the
pooling. In F32 (MODEL and EXACT) it runs the same four matrix products
in F32, reading each layer's weights as they are stored, `[out, in]`; the
projections back to the hidden width add their bias and residual there,
and the backend's LayerNorm follows from that one array. For a hidden width that is a multiple of 64 up to 1024, the F32
LayerNorm after a projection holds each token's row in registers, so it
reads the row from memory once instead of once per pass. oneDNN runs on the backend's own device,
context and memory through a SYCL queue of its own, which the backend
orders against its command list by events. The F16 weights are
transposed once more at load, into the plain layout OpenVINO hands
oneDNN: left to choose, oneDNN pads and keeps the rows, and runs 2%
slower from that on a B70 on bge-base and bge-large. Everything else,
the attention above all, is the backend's.

The build needs the oneAPI compiler (`TURBO_ICPX`, else `icpx` on the
`PATH`, with its environment set), oneDNN's headers and library, and
Level Zero's headers (`ze_api.h`) and loader
(`libze_loader.so`, which `onednn.cpp` links), from the oneAPI
installation or the distribution's Level Zero development package; it
makes `libturbo_onednn.so` in the build directory, which the library then
needs at run time together with oneDNN and the SYCL runtime from the
oneAPI installation. The compiler's lib directory is written into both as
a run-time path; oneDNN's own directory is not, so a program run outside
the oneAPI environment (`source /opt/intel/oneapi/setvars.sh`) finds
`libdnnl.so` through `LD_LIBRARY_PATH`. A program built in another crate,
such as `turbo-bench`, also finds `libturbo_onednn.so` that way. A device
oneDNN cannot open runs the backend's own kernels, and the log says so.

oneDNN builds a kernel (a primitive) for one shape, its tokens included,
which takes host memory and, for a shape whose kernel oneDNN has not
compiled, tens of milliseconds. So oneDNN runs a batch at its tokens
rounded up to a bin: a multiple of 16 up to 512 tokens, then of a 32nd of
the next power of two (32 up to 1024, 64 up to 2048, and so on), at most
the session's `max_batch` times `max_seq`. The context keeps the
primitives it builds for every session on it, so their number is bounded
by the bins, and only the first run at a bin builds any: that run counts
them in its `host_allocs`, and later runs at the bin allocate nothing.
(Built for every bin when a session is made instead, the same primitives
ran 0.6% to 1% slower on a B70, so they are not.) The rows of a bin past
the batch's tokens are computed into scratch no other kernel reads, a row
at a time, so they change no vector; a batch computes at most 15 rows
more than it holds up to 512 tokens, and at most a 16th more past that.

What the feature changes in the contract: where a session runs oneDNN, a
row's bits depend on the rows around it, since oneDNN's kernel for a batch
sums in its own order and the few-token kernels in theirs. The backend's
own kernels, forced with `linear=own`, keep a row's bits the same alone
and among others.

### Kernel choices

A session reports one choice in `turbo_session_get_info`'s `choices`
(docs/autotune.md): which kernels run its linear layers past 8 tokens,

```
linear=onednn;forced=
```

`linear=own` is the backend's kernels, `linear=onednn` oneDNN's; `forced=`
names `linear` when the environment fixed it. A build without the
feature, or a device where oneDNN does not open, has one path and reports
no choices unless the environment names one. A session nothing forces or
tunes runs oneDNN where it runs, at every precision: on a B70 it is the
faster at each precision, row mix and model measured (all-MiniLM-L6-v2,
bge-small, bge-base, bge-large, at 32 rows of 256 tokens and in mixed
rows), but for all-MiniLM-L6-v2's mixed rows at FASTEST, where the two
are within 2% of each other.

`TURBO_LEVELZERO_CHOICES`, read when a session is made, forces the item
it names (`linear=own` or `linear=onednn`); a line a session reported,
`forced=` and all, forces the same kernels back. `linear=onednn` where
oneDNN does not run is `TURBO_E_UNSUPPORTED_OPTION`, and any other item
is `TURBO_E_INVALID_ARGUMENT`.

### Autotuning

A session made with `turbo_session_desc.tuning` ON or RETUNE (or
`TURBO_AUTOTUNE=on` or `retune`) where both kernels run times them when
it is made and runs the faster; docs/autotune.md has the switches, the
cache and what holds across backends. It is off by default, and a forced
choice is never timed.

How: on the session's own memory, under the context's lock, with rows of
`max_seq` tokens, ids 1, as many as fill 8192 tokens (all of the
session's rows when it holds fewer). Each variant runs the encoder's
first layer once untimed, which builds its kernels, then five times timed
from the host, each run waited for. The least time ranks, and the
challenger replaces the incumbent (the cached choice, else the default)
only when 5% faster. A variant whose median is more than 25% above its
least is timed again, up to three times in all; still apart, the device
is shared or throttling, and the session keeps the incumbent, reports
`default`, is not cached, and an INFO line says so. The budget
(docs/autotune.md) counts the timed runs alone; past it, what is left is
not timed. A measured session reports `measured` and logs at INFO
`levelzero device <n>: linear layers chosen in <ms> ms: <choice> (<bin>,
a layer: own <ms> ms, onednn <ms> ms)`.

## What it does

- **Devices.** One per GPU the loader's GPU drivers list, in their order,
  read once per process. `device_info` gives the driver's name for the
  device, `unified_memory` for an integrated GPU, the loader's version as
  `runtime_version`, and the driver's as `driver_version` (Intel's
  packing of it, `1.3.37020`, else the number the driver reports).
  `memory_total` is the device memory a context may allocate;
  `memory_free` is read on each call through sysman, and is 0 (unknown)
  where sysman is not available. `arch`, the label benchmark records are
  filed under, is from the PCI device id: `b70` for 0xe223, and
  `intel-<id>` for a device not named yet.
- **Capability.** Embed is EXPERIMENTAL at every precision, honoring
  every field of `turbo_embed_options`. MODEL and EXACT compute in F32;
  FASTEST in F16 on the matrix engines (XMX), for a model whose hidden and
  intermediate widths are multiples of 32, and in F32 otherwise, which
  `turbo_session_get_info` says. A
  model stored in F16 or BF16 computes from an F32 copy at EXACT and
  FASTEST; its session at MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`,
  field 3), as on the CPU.
- **Contexts.** A context is a Level Zero context on the device and one
  in-order immediate command list, used under the context's lock. The
  encoder's kernels are built into one module per context, on first
  need, and shared by its models and sessions.
- **Buffers.** `DEVICE` is `zeMemAllocDevice` memory, `PINNED` host memory
  the driver allocated (`zeMemAllocHost`), which the device reads
  directly, `HOST` pageable memory aligned to 64 bytes, `SHARED`
  `zeMemAllocShared` memory. Import takes a `TURBO_HANDLE_ZE_USM` (aux the
  context's `ze_context_handle_t`, which must be this context's: a USM
  pointer is valid only in its own) as any placement the memory is, and a
  `TURBO_HANDLE_HOST_PTR` as `HOST`, or as `PINNED` or `SHARED` when the
  driver allocated it so; the driver is asked what the memory is. Export
  gives the USM pointer and the context's handle for memory the driver
  allocated, and the host pointer of `HOST`, `PINNED` and `SHARED`
  buffers. Any other kind is `TURBO_E_UNSUPPORTED`, naming it.
- **Models.** Loading copies every tensor, in the dtype it is stored in,
  into one device allocation. The first session makes the rest, on the
  device, shared by every later one and freed with the model: an F16 or
  BF16 model's F32 copy; each layer's Q, K and V weights and biases side
  by side, for one projection; and the linear layers' weights transposed
  to [inputs, outputs], in F16 for the first session at FASTEST and in
  F32 for the first session in F32 (for a model whose hidden and
  intermediate widths are multiples of 32).
- **Sessions.** Every byte a run touches is allocated when the session is
  made, for its `max_batch` rows of `max_seq` tokens: device scratch, the
  output buffer, host staging the device reads, and the session's own
  kernel objects. `embed_write` packs the rows, wherever they are, on the host into the
  staging: each row's positions through its last live token, one after
  another, as ids, positions, types and mask, with a table of where each
  row starts and how long it is. The padding after a row's last live
  token is never computed; no output depends on it. Rows in device memory
  are refused (`TURBO_E_INVALID_ARGUMENT`): the core reads and checks
  every row on the host before the backend sees it. The run computes on
  the device over the packed tokens: the embedding lookup, which reads the
  packed rows from the staging over the bus, and its LayerNorm in one
  kernel; per layer one projection for Q, K and V with their biases, one
  attention kernel (scaled dot products over the keys whose mask is 1,
  with an online softmax), the attention output projection, a residual
  and LayerNorm kernel, the feed-forward input with GELU (erf) in its
  epilogue, the feed-forward output (its sums split four ways over its
  terms where the kernels below split them), and a kernel that adds the
  parts, the residual and the LayerNorm; then one kernel pools (mean over the mask, the first token,
  or the last live one) and cuts to `output_dim`, and when asked another
  normalizes. The LayerNorms run a sub-group per token, or a group per
  token below 256 tokens. In F32 the linear layers run on the vector
  engines from the transposed weights: a sub-group computes 8 tokens by
  32 outputs, a group 4 x 2 of them, a 2D block read handing each lane
  its outputs' weights and each token's terms loaded once for the whole
  sub-group, so every multiply-add takes its term as a scalar operand;
  each output adds its products in term order. With 8 tokens or fewer, a
  group of up to 8 sub-groups computes 32 outputs from the untransposed
  weights, each summing an equal share of the terms. A model whose widths
  are not multiples of 32 runs 8 tokens by 64 outputs a sub-group from
  the untransposed weights. Attention for head widths 32, 64 and 128
  keeps each query and its running context in registers and streams the
  row's keys and values through local memory.

  At FASTEST the linear layers run on the matrix engines (DPAS), both
  operands arriving by 2D block reads already in the instruction's
  layout: the activations as rows, the transposed weights through the
  read's VNNI transform. A sub-group computes 16 tokens by 32 outputs, 32
  terms a step, and a group of 8 by 2 sub-groups shares its rows of both
  in cache; with 8 tokens or fewer, 8 tokens by 32 outputs, 4 sub-groups a
  group. For a hidden width of 768 or more (a multiple of 64: bge-base,
  bge-large) a sub-group computes 32 tokens by 64 outputs and a group is
  4 by 1 sub-groups, twice the products for each byte read; the kernels
  are built with the driver's compiler choosing each kernel's register
  file size, and that tile takes the large one. A layer's sums are never
  split, so each output is summed in one order at every batch size. The LayerNorms write the hidden states in F16
  too, for the layers that read them, and the LayerNorm-fused projections
  keep the residual stream in F16 alone; the feed-forward input writes F16,
  and so does the Q, K and V projection for the head widths whose
  attention runs on the matrix engines. The attention output and the feed-forward output each
  take their residual and LayerNorm in their own epilogue: a sub-group
  computes 16 tokens by 32 outputs (32 by 64 for the wide models), and a
  group the whole hidden width for up to 4 blocks of those tokens (as
  many as 64 sub-groups and the kernel's largest group hold, which the
  driver is asked; one block below a full group's tokens), so each block
  after the first reads the weights from cache; the rows' sums meet in
  local memory. For a hidden width over 2048 the LayerNorm kernel follows
  them instead. For the narrow models, from 4096 tokens the
  whole feed-forward block runs as one kernel laid out the same way: each
  group works through the intermediate width a hidden width at a time,
  its sub-groups writing their slice of the GELU'd middle in F16 to local
  memory and then adding its products into their outputs, so the middle
  never goes through global memory (at 32 x 256 it would be 25 MB, more
  than a B70's 24 MB cache). It computes what the two kernels do, bit for
  bit; it needs the intermediate width to be a multiple of the hidden
  width and its blocks' middle to fit the group's local memory. The
  wide models run the block as two kernels at every batch: measured on
  a B70, the one kernel is the slower of the two ways for them, with
  their middle through memory and all. Attention for head widths 32
  and 64 runs on the matrix engines: a sub-group takes 16 queries, a lane
  each, and walks the row's keys 32 at a time (K and V by 2D block
  reads); a group's 4 sub-groups take consecutive blocks of queries, so
  they read the row's keys and values from cache between them. For head
  width 64, when the longest row has 128 tokens or more, a group of 8
  sub-groups stages each 32-key tile of K and V in local memory once and
  its sub-groups read it from there, which on a B70 takes 13 to 15% off
  the attention of bge-base and bge-large; on shorter rows and on head
  width 32 the plain kernel is the faster one and runs. Both store a
  query's context as whole 64-byte lines, and every attention kernel
  walks the batch's rows newest first, since the projection wrote them
  in order and a batch's projections can outgrow the cache: on a B70
  the two together take another 16% off the attention of bge-base at
  32 rows of 256, with the same bits. Other
  widths write an F16 context from the kernels above. The run waits for the queue before it returns, and
  leaves the vectors in the session's `DEVICE` buffer:
  `turbo_result_buffer` hands out that memory, and `turbo_result_read`
  copies it back.
- **Session limits.** Beyond the model's own, one refusal comes from the
  device, for a head width other than 32, 64 or 128: a `max_seq` whose
  attention scores do not fit the local memory the device gives one
  work-group, less what its driver keeps for the kernel's own reductions
  (4 bytes per token plus the head's width and the partial sums; 128 KiB
  less 4 KiB on a B70, so about 31500 tokens) is
  `TURBO_E_UNSUPPORTED_OPTION` naming field 2.
- **Failures.** The driver's immediate list cannot finish or be destroyed
  after an append to it fails. Every append signals an event of the
  context's, so when one fails the call waits for the event of the last
  append that succeeded, which the in-order list signals after all the
  work before it, and frees nothing before then. It then sets that list
  aside, gives the context a new one, logs a warning, and returns the
  failure.
- **Numerics.** In F32 the arithmetic follows the CPU encoder where order
  matters: LayerNorm takes the mean, then the variance about it (summed
  in F32 here, as on CUDA, and in F64 on the CPU), softmax is taken
  from the largest live score (online, rescaling as a larger one
  arrives), mean pooling sums in position order, the L2 norm is summed in
  F64 and floored at 1e-12. Products and sums round separately except in
  the linear layers' and attention's multiply-adds. Cosine against the
  fp32 reference must reach 0.9999. At FASTEST the linear layers take F16
  operands; where the projections take their LayerNorm in their epilogue
  the residual stream between layers is F16, and the last layer's output
  is written in F32 as well for the pooling; the
  feed-forward block's middle and the attention context are F16, the
  block's GELU taking erf from a polynomial within 4.5e-5 of it; for
  head widths 32 and 64 attention takes F16 operands, its softmax in base
  2 on the device's native exponential, and for other widths it runs as
  in F32 and rounds its context to F16. Every sum and the softmax are F32, and so are the LayerNorms in
  the projections' epilogues; cosine must reach 0.999.
- **What a result reports.** Stages: tokenize on the host for text,
  upload fused into the lookup kernel, which reads the packed rows over
  the bus, lookup, encode, pool and (when asked) normalize on the device,
  and no download: the vectors stay where
  they are. `h2d_bytes` is what the lookup reads over the bus: 4 bytes per
  packed token for each of ids, positions, mask and (when given) types,
  and 8 per row for the table. `d2h_bytes` is 0 after the run and grows by
  each read. `host_allocs` and `device_allocs` are what
  the backend allocated on the running thread during the run; a run
  allocates nothing, cold or warm, but for oneDNN's primitives the first
  time a context runs a bin of tokens (oneDNN for the linear layers,
  above).

## Testing on a machine with an Intel GPU

From the workspace root. `TURBO_TEST_REQUIRE_LEVELZERO=1` makes every test
in `core/tests/levelzero.rs` that needs a device fail when the backend
lists none, where without it the test passes with a line saying it was
skipped; set it on a machine with an Intel GPU, so a run that found none
cannot pass.

```
export TURBO_TEST_REQUIRE_LEVELZERO=1

# Everything, with the levelzero-only tests in core/tests/levelzero.rs:
cargo test -p turbo --features levelzero

# The levelzero-only tests, with what they print (the device, the largest
# difference from the f64 encoder and the CPU):
cargo test --release -p turbo --features levelzero --test levelzero -- --nocapture

# Conformance on the small sealed bundle (docs/conformance.md):
TURBO_TEST_DEVICE=levelzero cargo test --release -p turbo --features levelzero --test conformance -- --nocapture

# On a real bundle, made with turbo-bundle (bundle/README.md): conformance,
# and one run at the bundle's max_batch x max_seq against the CPU.
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=levelzero \
    cargo test --release -p turbo --features levelzero --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features levelzero --test levelzero -- --include-ignored --nocapture
```

`TURBO_TEST_DEVICE=levelzero` picks the first device the levelzero
backend lists, and the conformance test fails when it lists none. The
tests in `core/tests/levelzero.rs` also check the listing against the
Intel GPUs the kernel drives, from `/sys/class/drm`, and read device
memory from the caller's side through the loader, so they need no other
tool.
