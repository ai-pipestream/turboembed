# Releases

A tag `v<version>` on `main` makes two archives through
`.github/workflows/release.yml` and publishes them as one GitHub release,
each with its SHA-256:

- `turboembed-v<version>-linux-x86_64.tar.gz`: the cpu and cuda backends.
- `turboembed-v<version>-macos-arm64.tar.gz`: the metal and cpu backends,
  for Apple silicon.

The same tag pushes two images of the gRPC server to GHCR (Images,
below). Nothing else is published: no crate.

## What the Linux archive holds

```
lib/libturbo.so            the library: the cpu and cuda backends
bin/turbo-kserve           the gRPC server (docs/grpc.md), the same backends linked in
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
licenses/CUTLASS-LICENSE.txt   the CUTLASS headers the CUDA backend carries, BSD-3-Clause
BUILD                      the commit, the compilers, the architectures
README.md                  this file
```

`libturbo.so` and `turbo-kserve` link the CUDA runtime statically and
cuBLAS not at all (docs/cuda.md, Requirements), so a machine needs the
NVIDIA driver alone to use the cuda backend, and nothing at all for the
cpu backend: a
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

## What the macOS archive holds

```
lib/libturbo.dylib         the library: the metal and cpu backends, arm64
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
BUILD                      the commit, the compiler, the macOS SDK
README.md                  this file
```

`libturbo.dylib` links the Metal and Foundation frameworks and libc++,
which every macOS has, and nothing else; its install name is
`@rpath/libturbo.dylib`, so a program finds it through its own rpath.
The Metal kernels are in the library as source and Metal compiles them
for the device when the first context on it is made (docs/metal.md).
The metal backend runs on macOS 14 or later; on an earlier macOS its
devices are listed with their capability cell UNSUPPORTED and the
reason, and the cpu backend runs as usual. An Intel Mac builds from the
tree (below).

## Images

```
ghcr.io/ai-pipestream/turbo-kserve:<version>        the cpu backend; linux/amd64 and linux/arm64
ghcr.io/ai-pipestream/turbo-kserve:<version>-cuda   the cpu and cuda backends; linux/amd64
```

`<version>` is the tag without its `v`; the release notes give each
image's digest, and a deployment that wants the exact build names it by
digest. Each image holds `turbo-kserve` alone, built from
`server/Dockerfile` on a Debian bookworm base pinned by digest, and its
licences under `/opt/turbo/licenses/`. The cuda image is built with the
toolkit and architectures of the Linux archive and, like it, links the
CUDA runtime statically: the host needs the NVIDIA driver and the NVIDIA
container toolkit, nothing in the image; without a GPU it runs the cpu
backend.

The image holds no model. It runs as user 10001, listens on
`0.0.0.0:8081`, and serves every bundle directory under `/bundles`, a
volume, on device 0 with one session. Everything else is the server's
own configuration through its `TURBO_KSERVE_*` variables (docs/grpc.md,
Flags); `TURBO_KSERVE_MODELS` names the bundles, devices and sessions
when the defaults are not what is wanted. Arguments after the image name
are the server's flags and replace their variables.

```
docker run --rm -p 8081:8081 -v /srv/bundles:/bundles:ro \
    ghcr.io/ai-pipestream/turbo-kserve:<version>
docker run --rm --gpus all -p 8081:8081 -v /srv/bundles:/bundles:ro \
    -e TURBO_KSERVE_MODELS='bundle=/bundles/bge-small-en-v1.5,device=select,sessions=4' \
    ghcr.io/ai-pipestream/turbo-kserve:<version>-cuda
```

The image's health check runs `turbo-kserve health`, one
`grpc.health.v1.Health/Check` against the listen address: healthy once
every bundle is loaded, unhealthy while loading (after a 120 second
start period) and while draining. Kubernetes asks the same service
itself:

```yaml
startupProbe:
  grpc: { port: 8081 }
  periodSeconds: 5
  failureThreshold: 120   # ten minutes to load
livenessProbe:
  tcpSocket: { port: 8081 }
  periodSeconds: 10
readinessProbe:
  grpc: { port: 8081 }
  periodSeconds: 5
```

The health service answers readiness alone (every service name the
server serves maps to `ServerReady`), so liveness asks only that the
listener answers. The readiness probe turns false when the server
drains on SIGTERM, so
the endpoint leaves the service before the requests in flight finish;
`terminationGracePeriodSeconds` should cover the longest request.

Each build in `release.yml` runs its image with
`testdata/tiny-bert-bundle` mounted and waits for it to turn healthy;
the amd64 cpu image also embeds the bundle's reference texts over gRPC
and checks them against its reference vectors
(`server/tests/image_smoke.rs`). The cuda image is checked on its cpu
backend only: the runners have no GPU.

## What a release claims

A release is built and packaged on runners that run nothing on a GPU.
The workflow checks that each library builds, that the Linux one links
no CUDA library dynamically, that the macOS one links only system
libraries and lists the metal backend in `turbo_version()`, and that the
header compiles as C11 and C++17.

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
tests have not run on it. On Apple silicon the records are the M2's,
for all-MiniLM-L6-v2 at each precision, against TEI's router built with
its Metal path (docs/metal.md); other models and other chips run, and
their cells say EXPERIMENTAL until a record names them.

## Other platforms

The archives are Linux x86_64 and macOS arm64. A Jetson (aarch64 with JetPack's CUDA) or
any other machine builds from the tree with the toolkit it has:

```
export TURBO_CUDA_ROOT=/usr/local/cuda
TURBO_CUDA_ARCH=87 cargo build -p turbo --release --features cuda
```

with `TURBO_CUDA_ARCH` the machine's own architecture (docs/cuda.md,
Building). The Level Zero and Hailo backends build the same way on their
machines (docs/levelzero.md, docs/hailo.md), as does the Metal backend
on an Intel Mac (docs/metal.md).

## Making one

```
git tag v0.1.0 <commit on main>
git push origin v0.1.0
```

The version is the `turbo` crate's (`core/Cargo.toml`). A tag on a
commit whose records are not on `main` publishes a library that carries
fewer records than it could; tag after the records land.
