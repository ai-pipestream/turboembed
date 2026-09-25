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
- **Capability.** Embed runs at every precision, computing in
  F32, honoring every field of `turbo_embed_options`. A model stored in
  F16 or BF16 computes in F32 from a converted copy at EXACT and FASTEST;
  its session at MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3),
  as on the CPU. A cell is SUPPORTED where a benchmark record in
  `benchmarks/records/` backs it (docs/benchmarks.md): on the M2, the
  records against TEI's router built natively with Metal. Elsewhere it is
  EXPERIMENTAL. On the M2, one 16-token request at a time runs in 1.69 ms
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
  across more threadgroups, the residual kernel summing the parts. Attention takes
  32 queries of a row and one head per threadgroup, and goes through the
  row's keys in chunks of 32 staged in threadgroup memory, with a running
  softmax; a head width that is not a multiple of 8, or is over 64, runs
  a narrower kernel, one query at a time. The run waits for the pass to
  complete before it returns, and leaves the vectors in the session's
  `SHARED` buffer, which `turbo_result_buffer` hands out and
  `turbo_result_read` copies from.
- **Session limits.** Beyond the model's own, a session whose scratch is
  more than one Metal buffer holds is `TURBO_E_UNSUPPORTED_OPTION` naming
  field 1. For a model whose heads run the narrow attention kernel, a
  `max_seq` whose scores do not fit the threadgroup memory the device
  gives (4 bytes per token; 32 KiB on Apple GPUs, so about 8000 tokens)
  names field 2.
- **Numerics.** F32 throughout. Apple GPUs have no F64, so where the CPU
  encoder sums in F64 this backend sums in F32: LayerNorm takes the mean,
  then the variance around it, in two passes, and the L2 norm is summed
  in F32 and floored at 1e-12. Softmax works from the largest live score,
  kept as it goes through the keys, and mean pooling sums in position
  order, as on the CPU. Metal has no `erf`; GELU's comes from erfc's
  Chebyshev fit in Numerical Recipes, with relative error under 1.2e-7.
  Cosine against the fp32 reference must reach 0.9999.
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

## Testing

From the workspace root. `TURBO_TEST_REQUIRE_METAL=1` makes every test in
`core/tests/metal.rs` that needs a device that runs embed fail when there
is none, where without it the test passes with a line saying it was
skipped; set it on Apple silicon, so a run that found no GPU cannot pass.

```
export TURBO_TEST_REQUIRE_METAL=1

# Everything, with the Metal-only tests in core/tests/metal.rs:
cargo test -p turbo --features metal

# The Metal-only tests, with what they print (the device, the largest
# difference from the f64 encoder and the CPU), in release for the
# largest shape:
cargo test --release -p turbo --features metal --test metal -- --nocapture

# Conformance on the small sealed bundle (docs/conformance.md):
TURBO_TEST_DEVICE=metal cargo test --release -p turbo --features metal --test conformance -- --nocapture

# On a real bundle, made with turbo-bundle (bundle/README.md): conformance,
# and one run at the bundle's max_batch x max_seq against the CPU.
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=metal \
    cargo test --release -p turbo --features metal --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features metal --test metal -- --include-ignored --nocapture
```

`TURBO_TEST_DEVICE=metal` picks the first device the metal backend lists,
and the conformance test fails when it lists none. The tests in
`core/tests/metal.rs` that need no device that runs embed (the table, the
device listing) run everywhere the feature builds.
