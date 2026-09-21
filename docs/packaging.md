# Packaging

`scripts/package.sh` builds one archive per target triple for using Turbo
outside this repository: `dist/turbo-<version>-<target>.tar.gz`. `<version>`
is `workspace.package.version` from `Cargo.toml` (`2.0.0-alpha.0` today);
`<target>` is the `host:` line of `rustc -vV` (`x86_64-unknown-linux-gnu` on
`krick`). This is a working packaging script for local and machine-to-machine
distribution, not the containerized, notarized, `abidiff`-gated per-platform
release pipeline `PLAN.md` section 10 (P8) scopes; P8 gets its version
script, SONAME policy, and clean-consumer install test from what this script
already produces.

## Layout

```
lib/libturbo.so                 the C ABI library (cdylib), crate turbo-shared
lib/libturbo.a                  the same code as a static library, if produced
include/turbo/turbo.h           the generated C API header
include/turbo/turbo_types.h     the generated types/constants header
include/turbo/turbo_provider.h  the generated provider-plugin vtable header
bin/turbo-bundle                import/verify/inspect for model bundles
providers/libturbo_provider_mock.so       always present; contract-testing only
providers/libturbo_provider_static.so     present if built as a cdylib
providers/libturbo_provider_cuda.so       present if the CUDA provider built
providers/libonnxruntime_providers_cuda.so    ONNX Runtime CUDA EP, next to the CUDA provider
providers/libonnxruntime_providers_shared.so  ONNX Runtime EP loader, next to the CUDA provider
providers/libturbo_provider_openvino.so   present if the OpenVINO provider built
LICENSE                         Apache-2.0
NOTICE                          only if the repository has one (it does not, today)
README.md                       written by the script for this specific archive
SHA256SUMS                      sha256sum of every other file, relative paths
```

No wrapping version directory: extracting the archive into a directory
produces `lib/`, `include/`, `bin/`, and `providers/` directly under it.

Every path under `providers/` is a provider that this machine could build;
a provider that could not be built is not silently missing, it is named as
absent in the archive's own `README.md` with the reason (skipped by a flag,
no toolkit, or a build failure), so a consumer never has to guess whether a
missing provider is a build defect or an intentional omission. `providers/`
is otherwise flat, matching how `turbo_runtime_load_provider` takes one
library path at a time; there is no per-provider subdirectory.

## What is deliberately not bundled

- **The CUDA 13 user-space libraries** (cuBLAS, cuBLASLt, cuDNN 9, NVRTC, the
  CUDA 13 runtime) that the CUDA provider's ONNX Runtime execution provider
  links against. On `krick` these come from NVIDIA's PyPI wheels unpacked
  into `.libs/nvidia/lib`, which `scripts/package.sh` explicitly excludes.
  They are large (several hundred megabytes), tied to a specific CUDA/driver
  combination, and already managed by whatever CUDA install the target
  machine has; bundling a copy would either duplicate what is already there
  or ship the wrong build for the target GPU driver. The provider finds them
  through `TURBO_CUDA_LIB_DIR` (or the context option `cuda_lib_dir`) at
  context creation and fails loudly (`TURBO_E_DEVICE_UNAVAILABLE`) if one is
  missing there; see `providers/cuda/README.md`.
- **The OpenVINO runtime** (`libopenvino.so`, its device plugins, and the
  TBB library it ships). It is a multi-hundred-megabyte SDK with its own
  versioned install tree (`openvino_genai_ubuntu26_2026.3.1.0_x86_64` on
  `krick-1` and `krick`); `libturbo_provider_openvino.so` is built with a
  RUNPATH into that tree, so it finds the runtime automatically as long as
  the SDK stays where it was built, or through `LD_LIBRARY_PATH` if moved.
  Bundling it would mean shipping and updating a second copy of a large SDK
  independently of the one already installed for OpenVINO's other uses.

Both are named, with the same reasoning, in the `README.md` that
`scripts/package.sh` writes into the archive.

## Building it

```bash
scripts/package.sh                        # build everything this machine can
scripts/package.sh --no-cuda              # skip trying to build the CUDA provider
scripts/package.sh --no-openvino          # skip trying to build the OpenVINO provider
```

Run from the repository root. It writes only under `dist/` (the final
archive) and `target/` (build output and a staging tree at
`target/package/turbo-<version>-<target>/`); nothing else in the tree is
modified. It fails loudly if `libturbo.so`, the generated headers, or
`turbo-bundle` cannot be built; the CUDA and OpenVINO providers are optional
and their absence is reported, never silently skipped without a note.

The script checks every `.so` it packages with `ldd` and prints its
dependencies. `libturbo.so` and the mock provider must resolve to nothing
beyond the baseline C/C++ runtime (`libc`, `libm`, `libgcc_s`, `libstdc++`,
`libpthread`, `libdl`, the dynamic linker, and the vDSO); anything else, or
an entry `ldd` cannot resolve, fails the build, because those two libraries
have no legitimate reason to depend on anything else. The other providers
are reported only: `libonnxruntime_providers_cuda.so` legitimately reports
`libcublasLt.so.13`, `libcublas.so.13`, and `libcudart.so.13` as not found on
a machine without `TURBO_CUDA_LIB_DIR` on the default loader path, because
those come from the excluded CUDA user-space libraries described above.

The script then verifies the archive it just built: it extracts it into
`target/package/verify-turbo-<version>-<target>/`, compiles
`crates/turbo-conformance/c/smoke.c` against the extracted `include/` and
links it against the extracted `lib/libturbo.so` (the same compiler
invocation `scripts/c-smoke.sh` uses), and runs the resulting binary with
`LD_LIBRARY_PATH` pointing at the extracted `lib/` against the repository's
own `testdata/bundles/mock` fixtures (those fixtures are test data, not
packaged). It prints the archive's path and size at the end.

## Consuming it from C

```c
#include "turbo/turbo.h"
/* cc app.c -I<archive>/include -L<archive>/lib -lturbo -Wl,-rpath,<archive>/lib -o app */
```

Load a provider at runtime by path:

```c
turbo_runtime_load_provider(rt, T("<archive>/providers/libturbo_provider_mock.so"), &err);
```

or list every provider path the program needs in
`turbo_runtime_desc.provider_paths` at `turbo_runtime_create` (see the
README's own "Load a provider library" section). `bin/turbo-bundle` needs no
linking; it is a standalone binary for preparing and inspecting bundle
directories.

## Consuming it from Rust

Two different things, not to be confused:

- **Path dependency** (working inside or against a checkout of this
  repository): depend on `turbo = { path = "crates/turbo" }` as the
  workspace already does. This gets the safe Rust API, source, and doc
  comments directly; it is what every crate in this workspace does today,
  and it is unaffected by `scripts/package.sh`, which never touches
  `crates/`.
- **The archive**: there is no Rust crate in it. The archive is the C ABI
  plus headers and provider libraries; a Rust consumer outside this
  repository links against `lib/libturbo.so` (or `lib/libturbo.a`) the same
  way a C consumer does, through a build script (`build.rs`) that emits
  `cargo:rustc-link-lib` and `cargo:rustc-link-search` for the archive's
  `lib/` directory, and either hand-written `extern "C"` declarations or a
  `bindgen`-generated binding over `include/turbo/turbo.h`. This is the same
  shape `crates/turbo-capi` uses in reverse (Rust exporting a C ABI, not
  importing one); nothing in this repository builds that binding today.

## What a future Maven artifact needs from this layout

`PLAN.md` section 13 decision 5 names the Java package `ai.pipestream.turbo`
and section 10 (P7) scopes it as a JDK 25 FFM binding. A Maven build for it
needs from this archive:

- **`include/turbo/turbo.h`** (plus `turbo_types.h`) as the input to
  `jextract`, to generate the FFM `MethodHandle`/`MemorySegment` bindings the
  Java package wraps. This is the same header a C consumer compiles against;
  nothing Java-specific needs to be generated into the archive itself.
- **`lib/libturbo.so`** per platform, packaged as a Maven classifier
  artifact (`ai.pipestream:turbo:<version>:linux-x86_64`, `:linux-aarch64`,
  `:macos-aarch64`, matching this script's `<target>` naming, translated to
  Java's `os.name`/`os.arch` convention) so the JAR's `Arena`-scoped loader
  can extract and `System.load()` the right native library for the running
  JVM, the same pattern DJL and other JNI/FFM-native Java libraries use. An
  `-auto` aggregator POM depending on every classifier is what section 10
  (P8) calls the "Maven classifier artifacts plus an `-auto` aggregator"
  deliverable.
- **`providers/*.so`**, unmodified, shipped either inside the same
  classifier artifact or as a separate `ai.pipestream:turbo-provider-cuda`
  style artifact per provider, since a provider is loaded by path at
  runtime (`turbo_runtime_load_provider`) and needs no Java-specific
  wrapping; the Java binding only needs to know where the JAR extracted the
  `.so` to, to pass that path back into the native call.
- **A SONAME and an `abidiff` gate**, which this script does not produce
  yet: `libturbo.so` today has no `SONAME` distinct from its filename, so
  two versions cannot coexist on one loader path and there is no automated
  check that a new build did not silently break the ABI a shipped JAR
  depends on. `PLAN.md` section 10 (P8) scopes both; until they exist, a
  Maven consumer must pin the exact archive version behind each classifier
  artifact rather than allowing a loader-path lookup to pick up whatever
  `libturbo.so` happens to be installed.
- **`SHA256SUMS`**, to verify the native payload the Maven build downloads
  or vendors before it is unpacked into the JAR, the same way `turbo-bundle`
  verifies a model bundle before loading it.

None of this is implemented; P7 (the FFM binding itself) has not started in
this tree, so this section is a requirements list against the current
archive layout, not a description of existing code.
