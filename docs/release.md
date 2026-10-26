# Releases

A tag `v<version>` on `main` makes four binary archives, a proto archive,
and generated Java and Python client stub artifacts through
`.github/workflows/release.yml` and publishes them as one GitHub release,
each with its SHA-256:

- `turboembed-v<version>-linux-x86_64.tar.gz`: the cpu and cuda backends.
- `turboembed-v<version>-macos-arm64.tar.gz`: the metal and cpu backends,
  for Apple silicon.
- `turboembed-v<version>-linux-aarch64.tar.gz`: the cpu backend.
- `turboembed-v<version>-windows-x86_64.zip`: the cpu backend, built with
  MSVC.
- `turboembed-proto-v<version>.tar.gz`: the versioned `server/proto/*.proto`
  definitions and `buf.yaml`.
- `turboembed-grpc-<version>.jar`, `turboembed-grpc-<version>-sources.jar`,
  and `turboembed-grpc-<version>.pom`: the generated Java Maven stub artifact
  (`ai.pipestream:turboembed-grpc:<version>`, built with pinned `protoc 4.36.2`
  and `protoc-gen-grpc-java 1.85.1`, depending on `protobuf-java 4.36.2` and
  `grpc-protobuf` and `grpc-stub` 1.85.1, Java 17 or newer).
- `turboembed_proto-<version>-py3-none-any.whl`: the generated Python wheel
  stub package (`turboembed-proto`, built with pinned `grpcio-tools 1.85.0`
  and `protobuf 7.36.2` on uv-managed Python). It requires Python 3.10 or
  newer, `grpcio>=1.85.0` and `protobuf>=7.35.1,<8`. `build_wheel.py` reads
  those floors from the generated code (`GRPC_GENERATED_VERSION` and the
  protobuf gencode version that grpcio-tools 1.85.0 bundles), and the
  smoke test installs the wheel on Python 3.10 with its dependencies
  resolved to exactly those floors before calling the server.

The generator pins live in `clients/java/pom.xml` (`protobuf.version`,
`grpc.version`) and `scripts/package-proto-stubs.sh` (the Python pins). The
Rust server builds its own stubs with tonic 0.14.6 and prost 0.14.4
(`Cargo.lock`).

Running the workflow by hand (`workflow_dispatch`) is the dry run: it
builds and checks the release archives and client stubs and keeps them,
with their sums, as the run's artifacts, and publishes nothing. What
changed in each release is in `CHANGELOG.md`; what a release may change
is under Versions and compatibility, below.

The same tag pushes two images of the gRPC server to GHCR (Images,
below), alongside the proto archive and generated Java and Python client
stub artifacts in the GitHub release (docs/grpc.md). The Rust crates are
not published.

## What the Linux archive holds

```
lib/libturbo.so            the library: the cpu and cuda backends
bin/turbo-kserve           the gRPC server (docs/grpc.md), the same backends linked in
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
licenses/NOTICE            notices for included third-party material
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
licenses/NOTICE            notices for included third-party material
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

## What the Linux aarch64 archive holds

```
lib/libturbo.so            the library: the cpu backend
bin/turbo-kserve           the gRPC server (docs/grpc.md), the same backend linked in
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
licenses/NOTICE            notices for included third-party material
BUILD                      the commit, the compilers, the glibc it was built against
README.md                  this file
```

It is built on an aarch64 runner (Ubuntu 24.04) and needs a glibc at
least as new as the one `BUILD` names. A Jetson or another aarch64
machine with an accelerator builds the backend it has from the tree
(Other platforms, below).

## What the Windows archive holds

```
bin/turbo.dll              the library: the cpu backend
bin/turbo-kserve.exe       the gRPC server (docs/grpc.md), the same backend linked in
lib/turbo.lib              turbo.dll's import library, for the MSVC linker
include/turbo/turbo.h      the C interface; the design
include/turbo/turbo_backend.h
licenses/LICENSE           Apache-2.0
licenses/NOTICE            notices for included third-party material
BUILD                      the commit and the compiler
README.md                  this file
```

`turbo.dll` links the C runtime dynamically: a machine needs the Visual
C++ redistributable (2015 or later), which most already have. The
workflow checks that the DLL loads and lists the cpu backend in
`turbo_version()`, and that the server starts and parses its arguments.

## Images

```
ghcr.io/ai-pipestream/turbo-kserve:<version>        the cpu backend; linux/amd64 and linux/arm64
ghcr.io/ai-pipestream/turbo-kserve:<version>-cuda   the cpu and cuda backends; linux/amd64
```

`<version>` is the tag without its `v`; the release notes give each
image's digest, and a deployment that wants the exact build names it by
digest. Each image holds `turbo-kserve` alone, built from
`server/Dockerfile` on a Debian bookworm base pinned by digest, and its
`LICENSE` and `NOTICE` under `/opt/turbo/licenses/`. The cuda image is built with the
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

For a different container port, set `TURBO_KSERVE_LISTEN` so both the
server and its separate health-check process use the same address:

```
docker run --rm -p 9090:9090 -v /srv/bundles:/bundles:ro \
    -e TURBO_KSERVE_LISTEN=0.0.0.0:9090 \
    ghcr.io/ai-pipestream/turbo-kserve:<version>
```

A server `--listen` flag does not change the health process's environment.
When using that flag, also replace Docker's health command:

```
docker run --rm -p 9090:9090 -v /srv/bundles:/bundles:ro \
    --health-cmd='turbo-kserve health --addr 127.0.0.1:9090' \
    ghcr.io/ai-pipestream/turbo-kserve:<version> --listen 0.0.0.0:9090
```

Changing only the host side of a port mapping, such as `-p 9090:8081`,
does not change the container listener or require a health override.

The image's health check runs `turbo-kserve health`, one
`grpc.health.v1.Health/Check` against `TURBO_KSERVE_LISTEN`: healthy once
every bundle is loaded, unhealthy while loading (after a 120 second
start period) and while draining. Kubernetes asks the same service
itself (adjust these ports when changing the listener):

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
both environment and command-line port overrides are checked too.
The amd64 cpu image also embeds the bundle's reference texts over gRPC
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

The archives are Linux x86_64 and aarch64, macOS arm64 and Windows
x86_64; only the first and third carry an accelerator backend. A Jetson
(aarch64 with JetPack's CUDA) or any other machine builds from the tree
with the toolkit it has:

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

Tag only after `CHANGELOG.md`'s top heading is the version being tagged
and holds every change since the last tag.

## Versions and compatibility

One version names everything a release ships: the tag, the `turbo`
crate's version, what `turbo_version()` begins with, the archives and
the server. It follows semantic versioning over three surfaces, the C
ABI, the gRPC services (with the server's flags and variables) and the
bundle format; nothing else is promised. The Rust crates are not
published and their APIs may change in any release.

While the version is `0.y.z`, a minor release (`0.y` to `0.y+1`) may
break any of the three, and says so under **Breaking** in
`CHANGELOG.md` with what to do about it; a patch release (`0.y.z` to
`0.y.z+1`) breaks none of them. From `1.0.0`, only a major release
breaks one.

### The C ABI (`include/turbo/`)

A program built against one release's headers runs, unchanged and
without a rebuild, against the library of any later release that does
not break the ABI. A release that does not break it may:

- add functions, constants, tasks (by the Tasks rule in `turbo.h`),
  status codes and enumeration values;
- add fields at the end of a struct: the `struct_size` a program passes
  tells the library which fields it knows, and a size an earlier release
  accepted stays accepted;
- fix behaviour that contradicts what the header says.

Breaking, so only in a release allowed to break: removing or renaming a
function, constant or field; changing a type, a field's offset or a
signature; reusing a task, status or enumeration number; changing what
an existing value or call means. A program must treat a status code it
does not know as a failure (`turbo_status_name` names it
`TURBO_E_UNKNOWN`) and must not assume a list of devices or backends.

### The gRPC services and the server's configuration

The packages are the vendored Open Inference Protocol (`inference`) and
`grpc.health.v1`, which follow their own specifications, and the
server's own `turbo.*` packages (docs/grpc.md). A release that does not
break them may add services, RPCs, messages, fields under new numbers,
enumeration values and server flags or `TURBO_KSERVE_*` variables; a
client must ignore fields it does not know and treat an enumeration
value it does not know as unspecified. Breaking: removing or renaming
an RPC, message, field, flag or variable; reusing or retyping a field
number; changing what a field, a status or a default means. A removed
field's number and name are reserved, never reused.

Once the `turbo.*` packages carry a version (`turbo.<name>.v1`, #1296),
a breaking change is a new package version, served beside the old one
for at least one minor release before the old one goes.

### The bundle format (docs/bundle.md)

A bundle names its schema in `manifest.json`'s `bundle_version`. A
release reads every `bundle_version` the releases before it read, and
refuses one it does not know, and reads a bundle an earlier release
made unchanged. A release that does not break the format may add
optional manifest fields and enumeration values; the manifest is parsed
strictly (docs/bundle.md), so an older library refuses a bundle that
uses one, by name, rather than misreading it. A required field, or a
change to what a field means, is a new `bundle_version`. Dropping a
`bundle_version` is breaking.

### What is not covered

Benchmark records and the speeds they report, the conformance floors'
numbers within the bounds docs/conformance.md states, log lines, error
message text (the status code and the field it names are covered; the
words are not), and anything under `analysis/` reached other than
through the C ABI or the gRPC services.
