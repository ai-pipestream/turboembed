# Setting up TurboEmbed

One page per kind of machine. Each says what to install, how to build
with the right feature, where a bundle comes from, how to check the
vectors against the bundle's reference, which tiers run there, and what
that machine cannot do.

| Machine | Backend | Cargo feature | Page | Check script |
|---|---|---|---|---|
| Any CPU (Linux, macOS, Windows) | `cpu` | on by default | [cpu.md](cpu.md) | `scripts/setup/cpu.sh` |
| NVIDIA GPU (RTX, data centre) or Jetson Orin | `cuda` | `cuda` | [cuda.md](cuda.md) | `scripts/setup/cuda.sh` |
| Intel Arc GPU with Xe2 (Arc Pro B70) | `levelzero` | `levelzero`, `levelzero-onednn` | [intel-gpu.md](intel-gpu.md) | `scripts/setup/intel-gpu.sh` |
| Intel Core Ultra NPU (AI Boost) | `npu` | `npu` | [intel-npu.md](intel-npu.md) | `scripts/setup/intel-npu.sh` |
| Apple silicon Mac | `metal` | `metal` | [apple-metal.md](apple-metal.md) | `scripts/setup/metal.sh` |
| Raspberry Pi 5 with a Hailo-10H | `hailo` | `hailo` | [hailo-10h.md](hailo-10h.md) | `scripts/setup/hailo.sh --device hailo10h` |
| A Hailo-8 or Hailo-8L board | `hailo` | `hailo` | [hailo-8.md](hailo-8.md) | `scripts/setup/hailo.sh --device hailo8` |

Two more pages cover what every machine shares:

- [bundles.md](bundles.md): getting a prebuilt bundle, or making one
  with `turbo-bundle` and its pinned containers
  (`scripts/setup/bundle-tool.sh`).
- [release-archive.md](release-archive.md): using the prebuilt Linux
  library instead of building.

## The check scripts

Every script under `scripts/setup/` checks by default and changes
nothing:

```
scripts/setup/cuda.sh            # report what is present and what is missing
scripts/setup/cuda.sh --dry-run  # print what --install would run
scripts/setup/cuda.sh --install  # install what needs no login or licence
```

Each line says `ok`, `missing` or `warning`, a missing item carries the
command that fixes it, and the script exits 0 only when nothing is
missing, so it can gate a build. `--install` prints each command before
it runs it and uses `sudo` when not run as root. A second run installs
nothing. The scripts never install or change a GPU or NPU driver, a
kernel module or group membership. They also never download anything
that needs an account (Hailo's compiler and HailoRT outside Raspberry Pi
OS): for those they check for the file and say where it goes.

## Building

Every setup builds the same way, from the workspace root, with the
feature its backend needs:

```
cargo build --release -p turbo --features <feature>
```

The library is `target/release/libturbo.so` (`libturbo.dylib` on macOS,
`turbo.dll` on Windows), and its C interface is `include/turbo/turbo.h`.
`turbo_version()` lists the backends the build carries, `0.1.0 cuda cpu`
for example. Features combine: `--features cuda,levelzero` lists the
devices of both, then the CPU.

The workspace is Rust edition 2024. CI builds with current stable Rust;
release archives are built with Rust 1.98.

## Using the library

[`embed.c`](embed.c) is a whole program on the C interface: it lists the
devices, loads a bundle on one, makes a session at a tier, embeds two
texts and reads the vectors back.

```
cargo build --release -p turbo
cc -std=c11 -I include docs/setup/embed.c -L target/release -lturbo \
    -Wl,-rpath,$PWD/target/release -o embed
./embed testdata/tiny-bert-bundle 0 model
```

The calls, in order: `turbo_runtime_create`, `turbo_context_create` on a
device index, `turbo_model_load` on a bundle directory,
`turbo_session_create` with a precision, `turbo_embed_write_text`,
`turbo_session_run`, `turbo_result_read`, then the releases. Every call
returns a `TURBO_E_*` status and fills a `turbo_error` with a message
and, where one argument is to blame, its field number. A device that
cannot do what was asked refuses with a status; nothing falls back to
another device.

The gRPC server, `turbo-kserve`, serves the same bundles to clients that
do not link the library: [../grpc.md](../grpc.md) says how to build and
start it, and [../kserve.md](../kserve.md) how each call maps onto the C
interface. Build it with the same feature as the library.

## Tiers

A session's precision, `turbo_session_desc.precision`, picks how the
loaded artifact computes. It never picks another artifact.

| Tier | Constant | What it means |
|---|---|---|
| MODEL | `TURBO_PRECISION_MODEL` | The artifact's own compute dtype: its `compute_dtype` if the manifest fixes one, else the dtype its weights are stored in. |
| FASTEST | `TURBO_PRECISION_FASTEST` | The fastest dtype the backend has for the artifact, which may be below the weights'. |
| EXACT | `TURBO_PRECISION_EXACT` | F32 throughout. |

What each tier computes in, per setup:

| Setup | MODEL | FASTEST | EXACT |
|---|---|---|---|
| CPU | F32 | F32 | F32 |
| CUDA | F32 (TF32 with `TURBO_CUDA_TF32=1`) | F16 (on the tensor cores from sm_80) | F32 |
| Intel GPU | F32 | F16 on the matrix engines when the hidden and intermediate widths are multiples of 32, else F32 | F32 |
| Intel NPU | F16, the graph's | F16, the graph's | refused |
| Apple Metal | F32 | F32 | F32 |
| Hailo | I8 | I8 | refused |

`turbo_session_get_info` reports the dtype a session resolved to. On
CPU, CUDA, Intel GPU and Metal a model whose weights are stored in F16
or BF16 is refused at MODEL (`TURBO_E_UNSUPPORTED_OPTION`, field 3) and
computes from a widened copy at FASTEST and EXACT.

What each dtype costs in accuracy is the bound the conformance test
holds it to, against the bundle's reference vectors (fp32, upstream
pipeline, CPU):

| Compute dtype | Lowest cosine to the reference | Largest absolute difference |
|---|---|---|
| F32 | 0.9999 | 1e-4 |
| F16, BF16 | 0.999 | not bounded |
| I8 | 0.93 | not bounded |

A benchmark record's `cosine_floor` and `speed_ratio`, where one exists
for the device (`benchmarks/records/`), say what was measured;
`turbo_runtime_capability` returns them.

## Checking a machine: conformance

The same test runs on every backend, through the C interface alone
([../conformance.md](../conformance.md)):

```
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=<backend> TURBO_TEST_PRECISION=<tier> \
    cargo test --release -p turbo --features <feature> --test conformance -- --include-ignored --nocapture
```

Unset, `TURBO_TEST_BUNDLE` is the small sealed bundle in
`testdata/tiny-bert-bundle`, which every build carries; with
`--include-ignored` and no `TURBO_TEST_BUNDLE` the run fails on purpose,
so a run meant for a real bundle cannot pass on the small one. Each
backend's test file also takes `TURBO_TEST_REQUIRE_<BACKEND>=1`
(`CUDA`, `LEVELZERO`, `NPU`, `METAL`, `HAILO`), which turns "no device
found, skipped" into a failure: set it on the machine the run is for.

## Environment variables

Read when a session is made, on every backend:

| Variable | Meaning |
|---|---|
| `TURBO_AUTOTUNE` | `off` (unset), `on` or `retune`, for sessions whose `tuning` is `TURBO_AUTOTUNE_RUNTIME` (0). Any other value refuses the session ([../autotune.md](../autotune.md)). |
| `TURBO_AUTOTUNE_BUDGET_MS` | Milliseconds a session may spend measuring kernels. |
| `TURBO_AUTOTUNE_CACHE` | A directory for measured choices; unset, memory only. |

Each setup's page lists the variables of its own backend, including the
ones that force a kernel path.
