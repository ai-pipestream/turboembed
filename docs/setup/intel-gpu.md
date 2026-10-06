# Intel GPU (Level Zero)

The `levelzero` backend runs on Intel GPUs through the oneAPI Level Zero
driver, with kernels of its own compiled to SPIR-V when the library is
built. With the `levelzero-onednn` feature, the linear layers of batches
over 8 tokens may run on oneDNN's kernels instead. Backend reference:
[../levelzero.md](../levelzero.md).

## Prerequisites

The GPU: an Xe2 part such as the Arc Pro B70, or Xe-HPC. The kernels
use 2D block reads and writes, which the Arc A-series does not have, and
the encoder sums the L2 norm in F64, which the GPU's driver must offer.

To build `levelzero`:

- A clang whose own SPIR-V backend compiles OpenCL C
  (`--target=spirv64`) and takes `--spirv-ext`. Clang 21 and newer do as
  shipped. Clang 20 does with `-fintegrated-objemitter`, which
  `scripts/setup/clang-spirv.sh` adds; point `TURBO_CLANG` at it.
  Clang 18 and older do not.
- Nothing of Level Zero.

To build `levelzero-onednn`, also:

- The oneAPI DPC++/C++ compiler (`icpx`) and oneDNN, with the oneAPI
  environment set (`source /opt/intel/oneapi/setvars.sh`). The compile
  includes `level_zero/ze_api.h` and links `-lze_loader`, which the
  oneAPI installation or the distribution's Level Zero development
  package (`libze-dev` on Ubuntu) provides.

To run:

- The Level Zero loader (`libze_loader.so.1`, 1.10 or newer) and Intel's
  GPU compute runtime (`libze_intel_gpu.so.1`), recent enough for Xe2.
  The B70 records in `benchmarks/records/` name the versions they ran
  on in their device block (loader 1.28.2, driver 1.3.37020).
- The kernel's `xe` or `i915` driver bound to the GPU, and the user able
  to open its render node (`/dev/dri/renderD*`, the `render` group).
- With `levelzero-onednn`: oneDNN and the SYCL runtime. The library
  finds `libdnnl.so` through the oneAPI environment or
  `LD_LIBRARY_PATH`.

```
scripts/setup/intel-gpu.sh                    # check
scripts/setup/intel-gpu.sh --onednn           # check the oneDNN build too
scripts/setup/intel-gpu.sh --onednn --install # Ubuntu: clang, Level Zero, compute runtime, oneAPI
```

The check compiles a small kernel with the flags the build uses, so it
reports whether the clang on the machine will do. On Ubuntu `--install`
installs `clang-20`, the Level Zero loader and Intel's compute runtime
(from Intel's graphics PPA on Ubuntu 24.04 to 25.04, whose own packages
predate Xe2), and with `--onednn` the oneAPI compiler and oneDNN from
Intel's public apt repository. It never changes the kernel driver or
adds the user to `render`; the check prints the `usermod` line.

## Build

```
export TURBO_CLANG=$PWD/scripts/setup/clang-spirv.sh    # unless clang on the PATH compiles the kernels
cargo build --release -p turbo --features levelzero

# or with oneDNN for the linear layers
source /opt/intel/oneapi/setvars.sh
cargo build --release -p turbo --features levelzero-onednn
```

| Variable | Meaning |
|---|---|
| `TURBO_CLANG` | The clang that compiles the kernels. Unset: `clang` on the `PATH`. |
| `TURBO_SPIRV_CLANG` | For `clang-spirv.sh` only: the clang it runs. Unset: the first of `clang-22`, `clang-21`, `clang-20`, `clang`. |
| `TURBO_ICPX` | The oneAPI compiler, for `levelzero-onednn`. Unset: `icpx` on the `PATH`. |

The loader is opened at run time, so a machine without it runs everything
else and lists no Intel GPU. `turbo_version()` returns `0.1.0 levelzero
cpu`. The driver builds the SPIR-V for the device the first time a
context needs it; that build is not part of any run.

## A bundle

Any bundle with a `FORMAT_SAFETENSORS` artifact whose `backends` lists
`levelzero` (every recipe's `weights-f32`): see [bundles.md](bundles.md).

## Conformance

```
export TURBO_TEST_REQUIRE_LEVELZERO=1

TURBO_TEST_DEVICE=levelzero cargo test --release -p turbo --features levelzero --test conformance -- --nocapture
TURBO_TEST_DEVICE=levelzero TURBO_TEST_PRECISION=fastest \
    cargo test --release -p turbo --features levelzero --test conformance -- --nocapture

TURBO_TEST_BUNDLE=models/bundles/bge-small-en-v1.5 TURBO_TEST_DEVICE=levelzero \
    cargo test --release -p turbo --features levelzero --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=models/bundles/bge-small-en-v1.5 \
    cargo test --release -p turbo --features levelzero --test levelzero -- --include-ignored --nocapture
```

Use `--features levelzero-onednn` in place of `levelzero` to check the
oneDNN path. The tests also check the device listing against the Intel
GPUs in `/sys/class/drm`.

## Tiers

| Tier | Computes in | Notes |
|---|---|---|
| MODEL | F32 | A model stored in F16 or BF16 is refused. |
| FASTEST | F16 on the matrix engines (XMX) | When the hidden and intermediate widths are multiples of 32; otherwise F32, which `turbo_session_get_info` reports. |
| EXACT | F32 | |

With oneDNN, a row's bits depend on the rows batched with it, since
oneDNN sums a batch in its own order; `TURBO_LEVELZERO_CHOICES=linear=own`
keeps every row's bits the same alone and among others.

## Environment

| Variable | Meaning |
|---|---|
| `TURBO_LEVELZERO_CHOICES` | `linear=own` or `linear=onednn`: which kernels run the linear layers past 8 tokens. `onednn` where oneDNN does not run is refused; any other item is refused. |
| `TURBO_LEVELZERO_PROFILE` | Any value: each context times every kernel and copy and logs the totals at debug level when released. A profiled run allocates. |

A build with oneDNN runs it wherever it opens. Where it does not open,
the backend runs its own kernels and logs a warning.

## Limits

- Xe2 or Xe-HPC only: an Arc A-series GPU is listed but its kernels do
  not build, and the first model load fails.
- Head widths other than 32, 64 and 128 take a general attention kernel
  whose `max_seq` is bounded by the GPU's local memory.
- The steps and the check script on this page are for Linux.

## Serving

`cargo build --release -p turbo-kserve --features levelzero` (or
`levelzero-onednn`), then [../grpc.md](../grpc.md).
