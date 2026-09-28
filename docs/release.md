# Releases

A tag `v<version>` on `main` makes one archive,
`turboembed-v<version>-linux-x86_64.tar.gz`, through
`.github/workflows/release.yml`, and publishes it as a GitHub release
with its SHA-256. Nothing else is published: no crate, no image.

## What the archive holds

```
lib/libturbo.so            the library: the cpu and cuda backends
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
licenses/CUTLASS-LICENSE.txt   the CUTLASS headers the CUDA backend carries, BSD-3-Clause
BUILD                      the commit, the compilers, the architectures
README.md                  this file
```

`libturbo.so` links the CUDA runtime statically and cuBLAS not at all
(docs/cuda.md, Requirements), so a machine needs the NVIDIA driver alone
to use the cuda backend, and nothing at all for the cpu backend: a
machine without a driver, or without an NVIDIA GPU, lists no CUDA device
and runs everything else as usual. The driver must support the CUDA
version the archive's `BUILD` line names; a 12.x build needs a 525
driver or newer.

The CUDA machine code the archive carries is what `BUILD` lists, one
architecture per NVIDIA generation from Turing (sm_75) up, and the PTX of
the last, which a newer GPU compiles when the library loads. A GPU
older than the first is listed, and its capability cell says UNSUPPORTED
with the reason. Below sm_80 the GEMMs and attention run without the
tensor cores.

## What a release claims

A release is built and packaged on a runner with no GPU. The workflow
checks that the library builds, links no CUDA library dynamically, and
that the header compiles as C11 and C++17; it runs nothing on a GPU.

What runs, and how fast, is what the records under `benchmarks/records/`
say, each from a named machine at a named commit, measured by
`turbo-bench` against the reference programs it names
(docs/benchmarks.md). The library carries those records and answers
`turbo_runtime_capability` and `turbo_runtime_select` from them: a
device for which no record exists reports its cell without a number and
never wins a selection on speed. Correctness on any machine is checked
the same way the records were: `docs/conformance.md` runs the sealed
test bundle, or a bundle of the user's, against the reference vectors it
carries, on the device named.

A machine not in the records is not measured. An architecture the
archive carries but no record names has been compiled for, and the
tests have not run on it.

## Other platforms

The archive is Linux x86_64. A Jetson (aarch64 with JetPack's CUDA) or
any other machine builds from the tree with the toolkit it has:

```
export TURBO_CUDA_ROOT=/usr/local/cuda
TURBO_CUDA_ARCH=87 cargo build -p turbo --release --features cuda
```

with `TURBO_CUDA_ARCH` the machine's own architecture (docs/cuda.md,
Building). The Metal, Level Zero and Hailo backends build the same way
on their machines (docs/metal.md, docs/levelzero.md, docs/hailo.md).

## Making one

```
git tag v0.1.0 <commit on main>
git push origin v0.1.0
```

The version is the `turbo` crate's (`core/Cargo.toml`). A tag on a
commit whose records are not on `main` publishes a library that carries
fewer records than it could; tag after the records land.
