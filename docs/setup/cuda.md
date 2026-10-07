# NVIDIA GPU (CUDA)

The `cuda` backend runs on NVIDIA GPUs through the CUDA runtime, with
kernels of its own. This page covers a desktop or server GPU on Linux
x86_64 and a Jetson Orin (aarch64). Backend reference:
[../cuda.md](../cuda.md).

## Prerequisites

To build:

- The CUDA toolkit, 12.x or 13.x: `nvcc`, the runtime's headers and
  `libcudart_static.a`. Release archives are built with 12.8.
- A host C++17 compiler nvcc accepts (gcc or clang; `NVCC_CCBIN` names
  another).
- With the `cuda-cublas` feature, which only measures and tests the
  backend's GEMMs against cuBLAS, the cuBLAS headers and `libcublas.so`.

To run:

- The NVIDIA driver alone: the runtime is linked in statically. A 12.x
  build needs driver 525 or newer, a 13.x build 580 or newer.
- `/dev/nvidia*` present (the driver's kernel module loaded).

```
scripts/setup/cuda.sh                 # check
scripts/setup/cuda.sh --install       # Ubuntu x86_64: NVIDIA's apt repository and the toolkit packages
scripts/setup/cuda.sh --cublas        # check for cuBLAS too
```

`--install` adds NVIDIA's public apt repository and installs
`cuda-nvcc-12-8`, `cuda-cudart-dev-12-8` and `cuda-cccl-12-8` (the same
packages the release workflow uses; `--toolkit 13.0` picks another
version). It never touches the driver: install the driver your
distribution packages for the GPU and reboot. The check reads the GPU's
compute capability from `nvidia-smi` and warns when the build would not
carry it.

### Jetson Orin

JetPack carries the driver and the toolkit; `sudo apt-get install
nvidia-jetpack` installs the toolkit if it is missing. The toolkit is at
`/usr/local/cuda`. Orin is sm_87.

## Build

```
export TURBO_CUDA_ROOT=/usr/local/cuda-12.8     # the toolkit's directory
export TURBO_CUDA_ARCH=89                        # your GPU's SM number
cargo build --release -p turbo --features cuda
```

| Variable | Meaning |
|---|---|
| `TURBO_CUDA_ROOT` | The toolkit's directory, with `bin/nvcc` and the libraries in `lib64/`, `lib/`, `targets/<arch>-linux/lib/` or `lib/<arch>-linux-gnu/`. A distribution's toolkit with nvcc in `/usr/bin` is `TURBO_CUDA_ROOT=/usr`. Unset: `CUDA_PATH`, then `CUDA_HOME`, then `/usr/local/cuda`. |
| `TURBO_CUDA_ARCH` | SM architectures to compile for, comma separated: `89`, or `80,86,89`. Default `89`. The highest also carries PTX, which a newer GPU compiles at load. |
| `NVCC_CCBIN` | The host compiler nvcc runs. |

`nvidia-smi --query-gpu=name,compute_cap --format=csv` prints the number:
8.9 is `89` (RTX 40 series), 8.6 `86` (RTX 30 series, A10), 8.0 `80`
(A100), 9.0 `90` (H100), 12.0 `120` (RTX 50 series), 8.7 `87` (Jetson
Orin). A GPU the build does not carry is still listed, and its
capability cell says UNSUPPORTED, naming `TURBO_CUDA_ARCH`.

On a Jetson Orin:

```
export TURBO_CUDA_ROOT=/usr/local/cuda
TURBO_CUDA_ARCH=87 cargo build --release -p turbo --features cuda
```

`turbo_version()` returns `0.1.0 cuda cpu`; the GPUs come before the CPU
in device order.

## A bundle

Any bundle with a `FORMAT_SAFETENSORS` artifact whose `backends` lists
`cuda` (every recipe's `weights-f32`): see [bundles.md](bundles.md).

## Conformance

```
export TURBO_TEST_REQUIRE_CUDA=1

# the sealed test bundle, at each tier
TURBO_TEST_DEVICE=cuda cargo test --release -p turbo --features cuda --test conformance -- --nocapture
TURBO_TEST_DEVICE=cuda TURBO_TEST_PRECISION=fastest \
    cargo test --release -p turbo --features cuda --test conformance -- --nocapture
TURBO_TEST_DEVICE=cuda TURBO_TEST_PRECISION=exact \
    cargo test --release -p turbo --features cuda --test conformance -- --nocapture

# a real bundle, and its largest shape against the CPU
TURBO_TEST_BUNDLE=models/bundles/bge-small-en-v1.5 TURBO_TEST_DEVICE=cuda \
    cargo test --release -p turbo --features cuda --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=models/bundles/bge-small-en-v1.5 \
    cargo test --release -p turbo --features cuda --test cuda -- --include-ignored --nocapture

# everything, with the cuBLAS checks
cargo test -p turbo --features cuda,cuda-cublas
```

## Tiers

| Tier | Computes in | Notes |
|---|---|---|
| MODEL | F32, F32 FMAs | `TURBO_CUDA_TF32=1` puts the GEMMs on the tensor cores as TF32 (sm_80 and newer), held to F32's cosine but not its largest absolute difference. |
| FASTEST | F16 | On sm_80 and newer, the GEMMs and attention run on the tensor cores with F16 operands and F16 sums within a chunk; below sm_80, F16 on the FMA units. A model with a weight beyond F16's range computes in F32, with a warning, and the session reports it. |
| EXACT | F32, F32 FMAs | Always. |

A model stored in F16 or BF16 is refused at MODEL.

The committed records (`benchmarks/records/rtx4080super.cuda.*` and
`orin.cuda.*`) cover an RTX 4080 SUPER and a Jetson Orin at each tier.

## Environment

Read when a session is made. The first two are for everyday use; the
rest force one kernel path, for measuring against the default.

| Variable | Meaning |
|---|---|
| `TURBO_CUDA_TF32` | `1`: MODEL's GEMMs as TF32 on the tensor cores. |
| `TURBO_CUDA_CHOICES` | A session's reported kernel line, forced back: the same kernels again ([../autotune.md](../autotune.md)). |
| `TURBO_CUDA_TILE` | The GEMMs' tile (`ctk`, `kt`, `8w`, `cs`, and the others [../cuda.md](../cuda.md) lists). |
| `TURBO_CUDA_ATTENTION` | `split`, `64`, `fa32`, `acc32` or `exact`. |
| `TURBO_CUDA_F16_ACCUMULATE` | `1`: FASTEST's GEMMs with F16 accumulators. |
| `TURBO_CUDA_LAYER_NORM` | `fused`: LayerNorm in the GEMM's epilogue. |
| `TURBO_CUDA_POOL` | `columns`: a thread per column for pooling. |
| `TURBO_CUDA_SK_STEPS` | 1 to 64, or `tiles`: how finely a GEMM's work is split. |
| `TURBO_CUDA_GELU` | FASTEST's GELU: `tanh` (the default), `erf` or `poly`. |
| `TURBO_CUDA_RESIDUAL`, `TURBO_CUDA_PRODUCT` | `f32`: FASTEST keeps an F32 residual stream, or F32 products, in place of F16. |
| `TURBO_CUDA_CUBLAS` | `qkv`, `out`, `ffn1`, `ffn2` or `all`: those GEMMs on cuBLAS (needs `cuda-cublas`). |

Of these, `TURBO_CUDA_CHOICES` refuses a value it does not know; the
others ignore one, so check `turbo_session_get_info`'s `choices` line
after setting them. With several GPUs, `CUDA_DEVICE_ORDER=PCI_BUS_ID`
keeps device indices in bus order (`turbo-bench` requires it).

## Limits

- Linux only.
- A head width up to 64 and a hidden width up to 2048; hidden and
  intermediate widths multiples of 8; `max_batch` up to 65535.
- Below sm_80 the GEMMs and attention run without the tensor cores.

## Serving

`cargo build --release -p turbo-kserve --features cuda`, then
[../grpc.md](../grpc.md).
