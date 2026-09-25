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

- To build: clang 15 or newer, which compiles OpenCL C to SPIR-V
  (`--target=spirv64`). Some clang builds do that through the
  `llvm-spirv` translator, which must then be on the `PATH`; the clang 21
  Ubuntu ships uses its own SPIR-V backend and needs none. Nothing of
  Level Zero is needed at build time.
- To run: the Level Zero loader (`libze_loader.so.1`, 1.10 or newer) and
  Intel's GPU driver for it (`libze_intel_gpu.so.1`, the compute runtime,
  at Level Zero 1.9 or newer for in-order immediate lists), with the kernel's `xe` or `i915` driver bound to the GPU and
  the user able to open its render node (the `render` group).
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
- **Capability.** Embed is EXPERIMENTAL at every precision, computing in
  F32, honoring every field of `turbo_embed_options`. A model stored in
  F16 or BF16 computes in F32 from a converted copy at EXACT and FASTEST;
  its session at MODEL is refused (`TURBO_E_UNSUPPORTED_OPTION`, field 3),
  as on the CPU.
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
  into one device allocation. An F16 or BF16 model's F32 copy is made on
  the device by the first session that computes in F32, shared by every
  later one, and freed with the model.
- **Sessions.** Every byte a run touches is allocated when the session is
  made, for its `max_batch` rows of `max_seq` tokens: device scratch, the
  output buffer, host staging the device reads, and the session's own
  kernel objects. `embed_write` sends the rows to the device, from the
  caller's memory when the driver allocated it (a `PINNED` or `SHARED`
  buffer's) and through the staging otherwise, and waits for them. Rows
  in device memory are refused (`TURBO_E_INVALID_ARGUMENT`): the core
  reads and checks every row on the host before the backend sees it. The
  run computes on the device over the written `[batch, seq]` grid: the
  embedding lookup and its LayerNorm in one kernel; per layer the Q, K
  and V projections, the attention output and the feed-forward layers as
  one tiled GEMM kernel, one attention kernel (scaled dot products over
  the keys whose mask is 1, softmax from the largest score), residual and
  LayerNorm kernels, and GELU with erf; then one kernel pools (mean over
  the mask, the first token, or the last live one), cuts to `output_dim`
  and normalizes. It waits for the queue before it returns, and leaves
  the vectors in the session's `DEVICE` buffer: `turbo_result_buffer`
  hands out that memory, and `turbo_result_read` copies it back.
- **Session limits.** Beyond the model's own, one refusal comes from the
  device: a `max_seq` whose attention scores do not fit the local memory
  the device gives one work-group, less what its driver keeps for the
  kernel's own reductions (4 bytes per token plus the head's width and the
  partial sums; 128 KiB less 4 KiB on a B70, so about 31500 tokens) is
  `TURBO_E_UNSUPPORTED_OPTION` naming field 2.
- **Failures.** The driver's immediate list cannot finish or be destroyed
  after an append to it fails. Every append signals an event of the
  context's, so when one fails the call waits for the event of the last
  append that succeeded, which the in-order list signals after all the
  work before it, and frees nothing before then. It then sets that list
  aside, gives the context a new one, logs a warning, and returns the
  failure.
- **Numerics.** F32 throughout. The arithmetic follows the CPU encoder
  where order matters: LayerNorm sums its mean and variance in F64,
  softmax subtracts the largest live score, mean pooling sums in position
  order, the L2 norm is summed in F64 and floored at 1e-12. Products and
  sums round separately except in the GEMM's multiply-add. Cosine against
  the fp32 reference must reach 0.9999.
- **What a result reports.** Stages: tokenize on the host for text,
  upload, lookup, encode and pool on the device, normalize fused into the
  pooling kernel when it runs, and no download: the vectors stay where
  they are. `h2d_bytes` is the rows sent by the write, 4 bytes per token
  for each of ids, mask and (when given) types. `d2h_bytes` is 0 after the
  run and grows by each read. `host_allocs` and `device_allocs` are what
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
