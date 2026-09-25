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

- To build: a clang whose own SPIR-V backend compiles OpenCL C to
  SPIR-V (`--target=spirv64`) and takes `--spirv-ext`, for the Intel
  sub-group extension the kernels use; the clang 21 Ubuntu ships does.
  Nothing of Level Zero is needed at build time.
- To run: the Level Zero loader (`libze_loader.so.1`, 1.10 or newer) and
  Intel's GPU driver for it (`libze_intel_gpu.so.1`, the compute runtime,
  at Level Zero 1.9 or newer for in-order immediate lists), with the
  kernel's `xe` or `i915` driver bound to the GPU and
  the user able to open its render node (the `render` group).
- A GPU with 2D block reads and writes (`cl_intel_subgroup_2d_block_io`:
  Xe2, such as the B70, and Xe-HPC), which the kernels are built with;
  on an older part, such as the Arc A-series, the module does not build
  and no session runs.
- A GPU whose driver computes in F64: the encoder sums LayerNorm and the
  L2 norm in F64. A device without it is listed, and its capability cell
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
| `TURBO_LEVELZERO_PROFILE` | At run time: every context times each kernel and copy on the device, and logs the totals at debug level when it is released. A profiled run allocates on the host to name what it times. |

The driver builds the SPIR-V for the device the first time a context
needs a kernel, when a model or session is made; that build is not part
of any run.

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
  by side, for one projection; and, for the first session at FASTEST, the
  linear layers' weights in F16, transposed to [inputs, outputs].
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
  epilogue, the feed-forward output with its sums split four ways over its
  terms, and a kernel that adds the parts, the residual and the
  LayerNorm; then one kernel pools (mean over the mask, the first token,
  or the last live one) and cuts to `output_dim`, and when asked another
  normalizes. The LayerNorms run a sub-group per token, or a group per
  token below 256 tokens. The linear layers run by sub-group, 8 tokens by
  64 outputs each, in F32 on the vector engines; with 8 tokens or fewer, a
  group of up to 8 sub-groups computes 32 outputs, each summing an equal
  share of the terms, in F32. Attention for head widths 32, 64 and 128
  keeps each query and its running context in registers and streams the
  row's keys and values through local memory.

  At FASTEST the linear layers run on the matrix engines (DPAS), both
  operands arriving by 2D block reads already in the instruction's
  layout: the activations as rows, the transposed weights through the
  read's VNNI transform. A sub-group computes 16 tokens by 32 outputs, 32
  terms a step, and a group of 8 by 2 sub-groups shares its rows of both
  in cache; with 8 tokens or fewer, 8 tokens by 32 outputs, 4 sub-groups a
  group. A layer's sums are never split, so each output is summed in one
  order at every batch size. The LayerNorms write the hidden states in F16
  too, for the layers that read them; the feed-forward input writes F16,
  and so does the Q, K and V projection for the head widths whose
  attention runs on the matrix engines. The attention output and the feed-forward output each
  take their residual and LayerNorm in their own epilogue: a group
  computes 16 tokens by the whole hidden width, a sub-group each 32
  outputs, and the rows' sums meet in local memory. For a hidden width over
  2048 the LayerNorm kernel follows them instead. Attention for head widths 32
  and 64 runs on the matrix engines: a group of 4 sub-groups takes 16
  queries, a lane each, the sub-groups walking the row's keys 32 at a time
  in turn (K and V by 2D block reads), and their running maxima, sums and
  contexts meet in local memory at the end; other widths write an F16
  context from the kernels above. The run waits for the queue before it returns, and
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
  matters: LayerNorm sums its mean and variance in F64, softmax is taken
  from the largest live score (online, rescaling as a larger one
  arrives), mean pooling sums in position order, the L2 norm is summed in
  F64 and floored at 1e-12. Products and sums round separately except in
  the linear layers' and attention's multiply-adds. Cosine against the
  fp32 reference must reach 0.9999. At FASTEST the linear layers take F16
  operands, the hidden states kept in F32 beside their F16 copy; the
  feed-forward block's middle and the attention context are F16; for
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
  allocates nothing, cold or warm.

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
