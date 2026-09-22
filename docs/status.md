# Turbo: detailed status (turbo-v2, 2026-09-22)

The long-form status that used to be the top-level README: what has
landed, on which machines, with which receipts. The top-level `README.md`
is the short version.

Turbo is a native inference and embedding library: one C ABI (`turbo_`
prefix, `libturbo`), one Rust safe API on top, and hardware support added as
runtime-loaded provider libraries (`libturbo_provider_<name>`) rather than
Cargo feature flags. The governing design document is [`PLAN.md`](../PLAN.md);
this file summarizes it and points at the current tree. Where this README and
`PLAN.md` differ on a detail, `PLAN.md` is authoritative.

Turbo is a rewrite of the project's proof of concept (tagged
`poc-2026-09-21`). The PoC's per-arch gRPC servers, the `turboembed.h` /
`turboembed_prepared.h` headers, and Cargo-feature-selected engines are
retired; see `PLAN.md` section 1 for what was kept and why, and
`docs/history/README.md` for the retired documentation.

## What it does

The C ABI exposes one object model for every provider (`PLAN.md` section
4.2):

```
turbo_runtime -> turbo_device -> turbo_context -> turbo_buffer
                                              \-> turbo_model -> turbo_session -> turbo_result
                                                             \-> turbo_generation
                    turbo_tokenizer (from a bundle)
```

Tasks (`TURBO_TASK_*` in `include/turbo/turbo_types.h`): `EMBED` (dense
embeddings), `RERANK` (cross-encoder), `CLASSIFY` (sequence classification),
`TOKEN_CLASSIFY` (per-token labels with span aggregation), `GENERATE` (causal
LM), `TOKENIZE`, `RUN` (generic named-tensor execution), and `CHUNK` (host
text chunking). Every option a caller can pass maps to a `TURBO_CAP_*`
capability bit; a provider either honors the option exactly or the call fails
with `TURBO_E_UNSUPPORTED_OPTION` naming the field. Nothing is silently
ignored, capped, or substituted (`PLAN.md` section 2).

Hardware support is added as separate provider libraries loaded at runtime
(`libturbo_provider_<name>`), not as Cargo feature flags. A provider exports
one symbol, `turbo_provider_get`, returning a versioned vtable
(`include/turbo/turbo_provider.h`); the core loads it through
`turbo_runtime_load_provider` or `turbo_runtime_desc.provider_paths`. See
"Load a provider library" below and [`docs/providers.md`](providers.md).

Per-model truth (pooling, normalization, sequence limit, prefixes, labels,
dimension, dtype) lives in a bundle directory's `bundle.json` manifest,
verified by content hash before anything loads. See
[`docs/bundles.md`](bundles.md).

## Status

This tree (branch `turbo-v2`, 2026-09-21): milestones P0 and P1
of `PLAN.md` section 10 are done, P2 (the OpenVINO provider) has landed its
embed, rerank, classify, and token-classify tasks (including the review
fixes in `docs/reviews/2026-09-21-p0-p2.md`), P3 (the CUDA provider) has
landed the same four tasks on x86_64 and now also runs embedding on Jetson
`nano1`, P5 (the `hailo` provider) has landed `EMBED x TEXT` on two
Raspberry Pi boards with a Hailo-8, P6 (the `ggml` provider) has landed
GGUF generation and GGUF embeddings on CUDA and CPU (`krick`) and on Metal
(`krickert-mac`, Apple M2) together with the push-style `turbo_generate`,
and P7 (the Java and Swift bindings) has landed the JDK 25 FFM binding and
the Swift package, both with generation and tokenizer wrappers. A second
review (`docs/reviews/2026-09-21-p3-p7.md`) closed its highs and mediums
the same day. Every per-call option now either honors exactly what the
caller asked for, gated by a `TURBO_CAP_OPT_*` bit, or fails; see
[`docs/c-api.md`](c-api.md)'s capability-bit tables.
`crates/turbo-bench` benchmarks every landed provider (embed, rerank,
generate) and writes receipts under `testdata/receipts/turbo/bench/`; the
direct-native reference programs under `reference/` drive each runtime
alone on the same token rows, and `turbo-bench compare` writes the
verdict. As of 2026-09-22, SUPPORTED: cuda embeddings on the RTX 4080
SUPER, openvino embeddings on the Battlemage B70, ggml generation on the
RTX 4080 SUPER, hailo embeddings on the Hailo-8, metal embeddings on the
M2, ggml embeddings on the RTX 4080 SUPER, openvino embeddings on the
Ryzen 9 CPU. What exists today:

- The generated C header set (`include/turbo/turbo.h`,
  `include/turbo/turbo_types.h`, `include/turbo/turbo_provider.h`), produced
  from `crates/turbo-abi` and `crates/turbo-capi` by `scripts/gen-header.sh`.
  `turbo_provider.h` is the plugin ABI a provider library implements; see
  "Load a provider library" below.
- `crates/turbo-core`: the runtime, device registry, buffers, bundle v2
  loader and verifier, sessions with result leases, streaming generation,
  tokenizers, the chunk planner, provider plugin loading, and the error
  model.
- `crates/turbo-capi`: the `extern "C"` exports; `crates/turbo-shared` links
  them into `libturbo` (`cdylib` + `staticlib`, crate name `turbo`).
- `crates/turbo`: the safe Rust API (re-exports `turbo-core`); its
  `builtin_providers()` includes `mock` and `static`.
- Built-in providers: `mock` (`crates/turbo-core/src/mock.rs`, deterministic
  hash-derived outputs for contract testing, packaged as a loadable library
  under `providers/mock`) and `static` (`providers/static`, model2vec-style
  token-table embeddings, one capability cell `EMBED x TEXT x CPU`,
  `EXPERIMENTAL`).
- Four more loadable providers, each built separately: `openvino`
  (`providers/openvino`, a C++ library built with CMake, implementing
  `turbo_provider.h` directly rather than through the Rust
  `export_provider!` macro) with embed, rerank, classify, and token-classify
  `EXPERIMENTAL` on GPU and CPU; `cuda` (`providers/cuda`, a Rust library
  using `export_provider!`, built with `cargo build -p turbo-provider-cuda`)
  with the same four tasks `EXPERIMENTAL` on `krick` (RTX 4080 SUPER,
  x86_64) through the ONNX Runtime CUDA execution provider, and now also
  passing its live embedding tests on Jetson `nano1`; `ggml`
  (`providers/ggml`, a Rust library using `export_provider!` over the
  `llama-cpp-2` binding to llama.cpp, a workspace member built by the
  default workspace commands) with GGUF `GENERATE` and GGUF `EMBED`
  `EXPERIMENTAL` on the CUDA and CPU devices of `krick` and, through
  llama.cpp's own Metal backend, on `krickert-mac` (Apple M2); and `hailo`
  (`providers/hailo`, a C++ library built with CMake against HailoRT 4.23,
  also implementing `turbo_provider.h` directly) with `EMBED x TEXT`
  `EXPERIMENTAL` on two Raspberry Pi boards with a Hailo-8 (`pi5ai1`,
  `cm5ai1`). See [`providers/openvino/README.md`](../providers/openvino/README.md),
  [`providers/cuda/README.md`](../providers/cuda/README.md),
  [`providers/ggml/README.md`](../providers/ggml/README.md),
  [`providers/hailo/README.md`](../providers/hailo/README.md), and
  [`docs/providers.md`](providers.md).
- `bindings/java` (`ai.pipestream:turbo`): a JDK 25 foreign-function binding
  with a raw layer generated from `include/turbo/turbo.h` by jextract and a
  safe `AutoCloseable` API on top, including `Generation` (the pull
  iterator, `drain` with a stopping predicate, cross-thread `cancel`) and a
  `Tokenizer` wrapper; its conformance cases pass through the binding
  against the mock provider under `--illegal-native-access=deny`.
  See [`bindings/java/README.md`](../bindings/java/README.md) and
  [`docs/bindings.md`](bindings.md).
- `bindings/swift` (`PipestreamTurbo`): a SwiftPM package over `libturbo`'s C
  ABI; `CTurbo` exposes the generated header as a clang module and
  `PipestreamTurbo` is the Swift API on top, with `Generation` and
  `Tokenizer` wrappers mirroring the Java ones and the same conformance
  cases run as an executable (`swift run turbo-conformance`) because the
  Swift command line tools ship neither XCTest nor Swift Testing. All
  sixteen cases pass on `krickert-mac` (Apple M2, 2026-09-22). See
  [`bindings/swift/README.md`](../bindings/swift/README.md) and
  [`docs/bindings.md`](bindings.md).
- `tools/turbo-bundle`: `import` (derives a bundle's contract from a source
  model's own files), `verify`, and `inspect`. See
  [`docs/bundles.md`](bundles.md).
- `crates/turbo-conformance`: a C smoke test (`c/smoke.c`) and a
  provider-agnostic Rust suite (over 200 tests across 25 files under
  `tests/`) covering the P0 groups from `PLAN.md` section 10 plus live
  provider tests (`tests/live_embed.rs`, `tests/live_tasks.rs`,
  `tests/live_generate.rs`) that skip themselves without hardware. See
  [`docs/testing.md`](testing.md).
- `crates/turbo-bench`: `embed`/`rerank`/`generate` benchmark workloads run
  through the safe Rust API, a receipt writer, a `--budget` regression
  check against an earlier receipt, and `turbo-bench discover`, which
  surveys a machine's providers, devices, features, and capability matrix
  and says which named bundles it can run. Receipts are committed under
  `testdata/receipts/turbo/bench/` beside the native and comparison
  receipts of each matched pair (`reference/README.md`). See
  [`docs/testing.md`](testing.md).
- Mock bundle fixtures under `testdata/bundles/mock/` (embedding, reranker,
  classifier, token-classifier, generative, generic) and a tokenizer-only
  fixture at `testdata/bundles/minilm-tokenizer/`, all generated or imported,
  not hand-written.

Every function family declared in the header is implemented, including the
push-style `turbo_generate`: it creates a generation, applies the messages,
and drives the pull iterator (`turbo_generation_step`) internally, calling
back once per chunk until the generation finishes or the callback returns
`TURBO_STREAM_STOP` (which cancels it); the chunk's pointers are valid only
during the callback. See [`docs/c-api.md`](c-api.md) for the full
picture.

The dedicated `metal` provider (P4) landed on 2026-09-22 as a direct
Metal provider (Objective-C++, kernels compiled at load, no MLX) serving
embeddings and reranking on Apple M2; `ggml` reaches the same GPU for
generation through llama.cpp's own Metal backend. The `hailo` provider (P5) serves
embeddings on the Hailo-8 Pis; Hailo-10H is still open. The Android binding
(P10) is not available yet; the Java and Swift bindings (P7) have landed.
CUDA on Jetson (`nano1`, aarch64) now passes its live embedding
tests (cosine 1.000); the task suite (rerank, classify, token-classify) is
still being verified there (`PLAN.md` section 10, P3).

## Capability and hardware status

Every provider reports a capability cell per (device, task, modality) as
`UNSUPPORTED`, `PLANNED`, `EXPERIMENTAL`, or `SUPPORTED`
(`TURBO_CAP_*` in `turbo_provider.h`), plus measured precision figures once a
qualification receipt exists (`PLAN.md` section 4.4). Today:

| provider | status | notes |
|---|---|---|
| `mock` | supported for contract testing only | deterministic, hash-derived; serves only `mock`-artifact bundles; never a real model |
| `static` | EXPERIMENTAL | one capability cell, `EMBED x TEXT x CPU`; table lookup + mean + L2 on host, explicit device selection only; precision receipt against model2vec still pending |
| `openvino` GPU | EXPERIMENTAL | embed, rerank, classify, token-classify on `krick-1` (Battlemage B70); fused mean+L2 graph, device-resident results; receipts: [`testdata/receipts/turbo/openvino-minilm-2026-09-21.json`](../testdata/receipts/turbo/openvino-minilm-2026-09-21.json), [`openvino-tasks-2026-09-21.json`](../testdata/receipts/turbo/openvino-tasks-2026-09-21.json) |
| `openvino` CPU | SUPPORTED for embeddings on `krick` (`compare-openvino-krick-cpu-embed-2026-09-22b.json`, 1.03x to 1.30x); EXPERIMENTAL for the other tasks | same tasks, explicit selection only; same receipts |
| `cpu` | PLANNED (P3, folded into the CUDA/ORT provider work) | ORT CPU EP; ggml CPU |
| `cuda` | SUPPORTED for embeddings on `krick` (`compare-cuda-krick-embed-2026-09-22.json`); EXPERIMENTAL for the other tasks and on Jetson `nano1` (aarch64) | embed, rerank, classify, token-classify through the ONNX Runtime CUDA EP with device-side pooling/normalization/activation kernels; cosine 1.000 against the FP32 references on `krick`; receipt: [`testdata/receipts/turbo/cuda-2026-09-21.json`](../testdata/receipts/turbo/cuda-2026-09-21.json). On `nano1` (JetPack R39 rev 2.0, CUDA 13.2, ONNX Runtime 1.24.0 linked dynamically through `ORT_LIB_LOCATION` and `--no-default-features`) all 12 live embedding tests pass at cosine 1.000; no receipt file is committed for this run yet and the task suite (rerank/classify/token-classify) is still being verified there |
| `metal` | SUPPORTED for embeddings on `krickert-mac` (`compare-metal-mac-embed-2026-09-22.json`); EXPERIMENTAL for rerank | embed and rerank through Metal directly: MSL kernels compiled at load, shared `MTLBuffer`s end to end, results `SHARED` in unified memory at cosine 1.000 against the FP32 references; receipt: [`testdata/receipts/turbo/metal-2026-09-22.json`](../testdata/receipts/turbo/metal-2026-09-22.json) |
| `hailo` | SUPPORTED for embeddings on `pi5ai1` (`compare-hailo-pi5ai1-embed-2026-09-22.json`), EXPERIMENTAL on `cm5ai1` (no comparison yet); Hailo-8L untested; Hailo-10H open | embed through HailoRT 4.23 vstreams with the INT8 Model Zoo MiniLM HEF; host WordPiece, word-embedding gather, pooling, and L2; the capability cell reports a measured cosine floor of 0.30, and the live suite holds cosine to its own, higher floor for this provider and dtype ([`testdata/reference_embeddings/quantized_floors.json`](../testdata/reference_embeddings/quantized_floors.json), 0.45) plus a ranking gate on the STS corpus (Spearman 0.937 against 0.944 for FP32); throughput 76 rows/s at every batch and sequence length ([`testdata/receipts/turbo/bench/hailo-pi5ai1-embed-2026-09-21.json`](../testdata/receipts/turbo/bench/hailo-pi5ai1-embed-2026-09-21.json)); receipt: [`testdata/receipts/turbo/hailo-2026-09-21.json`](../testdata/receipts/turbo/hailo-2026-09-21.json) |
| `ggml` | SUPPORTED for generation (`compare-ggml-krick-gpu-generate-2026-09-22.json`) and for GGUF embeddings (`compare-ggml-krick-gpu-embed-2026-09-22b.json`, 0.98x to 1.90x) on the RTX 4080 SUPER of `krick`; EXPERIMENTAL on the CPU and on `krickert-mac` | GGUF generation through llama.cpp (`llama-cpp-2`): pull iterator, chat templates, stop strings/tokens, cancellation, logprobs, seeded sampling, GBNF grammars; GGUF embedding on the same devices at cosine 0.99999 or better against the FP32 references; receipt: [`testdata/receipts/turbo/ggml-2026-09-21.json`](../testdata/receipts/turbo/ggml-2026-09-21.json) |

A matched-native benchmark receipt is still required before OpenVINO,
`cuda`, `ggml`, `hailo`, or `static` can move from `EXPERIMENTAL` to
`SUPPORTED` (`PLAN.md` section 2, item 7); the receipts above record this
explicitly under `status_after`, and `crates/turbo-bench` already writes
the `libturbo`-side half of that comparison (`testdata/receipts/turbo/bench/`)
even though the direct-native reference programs are not written yet.

See [`docs/providers.md`](providers.md) for the full table (hardware,
runtime, lowest layer, machine) from `PLAN.md` section 7, and
[`docs/architecture.md`](architecture.md) for the capability matrix
model.

## C example

This mirrors `crates/turbo-conformance/c/smoke.c`. Error checking is
abbreviated for space; the smoke test shows the full pattern, including how
to read `turbo_error` and reject unsupported options by field index.

```c
#include "turbo/turbo.h"
#include <stdio.h>
#include <string.h>

static turbo_text T(const char *s) { turbo_text t = {s, (uint64_t)strlen(s)}; return t; }

int main(void) {
    turbo_error err = {0};
    err.struct_size = (uint32_t)sizeof err;

    turbo_runtime *rt = NULL;
    turbo_runtime_create(NULL, &rt, &err);          /* library instance */

    uint32_t dev = 0;                                /* AUTO: best accelerator, never CPU */
    turbo_runtime_select_device(rt, NULL, &dev, &err);

    turbo_context *ctx = NULL;
    turbo_context_create(rt, dev, NULL, &ctx, &err); /* device + memory domain */
    turbo_runtime_release(rt);                       /* ctx retains it */

    turbo_model *model = NULL;                       /* load and verify a bundle */
    turbo_model_load(ctx, T("testdata/bundles/mock/embedding"), NULL, &model, &err);

    turbo_session *session = NULL;
    turbo_session_create(model, NULL, &session, &err);
    turbo_model_release(model);                      /* session retains it */
    turbo_context_release(ctx);

    turbo_text texts[1] = {T("hello world")};
    turbo_session_write_text(session, texts, 1, NULL, &err);

    turbo_result *result = NULL;
    turbo_session_run(session, NULL, &result, &err);

    float out[8];
    uint64_t written = 0;
    turbo_result_read(result, 0, out, sizeof out, &written, &err);
    printf("embedding[0] = %f (%llu bytes)\n", (double)out[0], (unsigned long long)written);

    turbo_result_release(result);
    turbo_session_release(session);
    return 0;
}
```

## Load a provider library

A hardware provider such as OpenVINO is a separate shared library that
exports `turbo_provider_get` (`include/turbo/turbo_provider.h`). Load it
either through `turbo_runtime_desc.provider_paths` at creation or later
through `turbo_runtime_load_provider`:

```c
#include "turbo/turbo.h"
#include <string.h>

static turbo_text T(const char *s) { turbo_text t = {s, (uint64_t)strlen(s)}; return t; }

turbo_error err = {0};
err.struct_size = (uint32_t)sizeof err;

turbo_runtime *rt = NULL;
turbo_runtime_create(NULL, &rt, &err);          /* built-in mock and static providers */

turbo_runtime_load_provider(rt, T("build/openvino/libturbo_provider_openvino.so"), &err);

/* Or skip the built-ins entirely and load only explicit providers: */
turbo_text path = T("build/openvino/libturbo_provider_openvino.so");
turbo_runtime_desc desc = {0};
desc.struct_size = (uint32_t)sizeof desc;
desc.flags = TURBO_RUNTIME_NO_DEFAULT_PROVIDERS;
desc.n_provider_paths = 1;
desc.provider_paths = &path;
turbo_runtime *rt2 = NULL;
turbo_runtime_create(&desc, &rt2, &err);
```

See [`docs/providers.md`](providers.md) for the ownership rules a
provider library must follow and [`providers/openvino/README.md`](../providers/openvino/README.md)
for building the OpenVINO library.

## Rust example

The safe `turbo` crate uses `Arc`-retained handles and drops instead of
explicit release calls.

```rust
use turbo::{ContextDesc, DeviceSelector, EmbedOptions, ModelDesc, RunOptions, RuntimeDesc, SessionDesc};
use std::path::Path;

fn main() -> turbo::Result<()> {
    let rt = turbo::create_runtime(RuntimeDesc::default())?;
    let device = rt.select(&DeviceSelector::default())?; // AUTO: best accelerator, never CPU
    let ctx = turbo::Context::create(rt, device, &ContextDesc::default())?;
    let model = ctx.load_model(Path::new("testdata/bundles/mock/embedding"), &ModelDesc::default())?;
    let session = model.create_session(&SessionDesc::default())?;

    session.write_text(&["hello world"], &EmbedOptions::default())?;
    let result = session.run(&RunOptions::default())?;

    let mut out = vec![0u8; result.output(0)?.logical_bytes()? as usize];
    result.read(0, &mut out)?;
    println!("read {} bytes", out.len());
    Ok(())
}
```

## Java

`ai.pipestream:turbo` (`bindings/java`) is a JDK 25 binding built on the
foreign function and memory API: no JNI, no generated C. It has two layers,
a raw `ai.pipestream.turbo.ffi` surface generated from `include/turbo/turbo.h`
by jextract (`scripts/gen-java-ffi.sh`, committed, not hand-edited) and a
safe `ai.pipestream.turbo` API (`Turbo`, `Context`, `Model`, `Session`,
`Result` as `AutoCloseable` handles over the C handles).

```java
try (Turbo rt = Turbo.create(List.of("/opt/turbo/providers/libturbo_provider_cuda.so"));
     Context ctx = rt.createContext(rt.selectDevice());
     Model model = ctx.loadModel("/opt/bundles/minilm-onnx");
     Session s = model.createSession(8, 256)) {
    s.writeText(List.of("hello world"), EmbedOptions.defaults());
    try (Result r = s.run()) {
        float[] v = r.readFloats(0);
        System.out.println(r.placement() + " " + v.length);
    }
}
```

The conformance cases run through the binding against the mock provider
under `--illegal-native-access=deny`; on `krick` (JDK 25.0.3, Temurin) and
`krick-1` (JDK 25.0.4, Temurin) sixteen tests pass in under a second, and
the same job runs in CI (`.github/workflows/ci.yml`, `java`) against the
mock provider only. `Model.createGeneration` and `Turbo.createTokenizer`
add a `Generation` (pull iterator, `drain` with a stopping predicate,
cross-thread `cancel`) and a `Tokenizer` wrapper; see
[`bindings/java/README.md`](../bindings/java/README.md) and
[`docs/bindings.md`](bindings.md) for locating the library, building,
regenerating the raw layer, and a generation/tokenizer example.

## Build and test

```bash
cargo test --locked --workspace --exclude turbo-provider-cuda   # unit tests plus the Rust conformance suite, against mock
scripts/gen-header.sh --check          # fail if include/turbo/*.h is stale
scripts/gen-versioned.py --check       # fail if the struct_size table is stale
scripts/c-smoke.sh                     # build libturbo, compile and run smoke.c
```

`scripts/gen-header.sh` (no `--check`) regenerates the three committed
headers (`turbo_types.h`, `turbo_provider.h`, `turbo.h`) from
`crates/turbo-abi` and `crates/turbo-capi` with cbindgen; never hand-edit
`include/turbo/*.h`. `scripts/gen-versioned.py` (no `--check`) regenerates
the accepted `struct_size` table, `crates/turbo-abi/src/versioned.rs`.
`cargo run -p turbo-core --example write_mock_bundles` regenerates the
fixtures under `testdata/bundles/mock/`. CI (`.github/workflows/ci.yml`)
runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --locked
--workspace` (which runs `crates/turbo-conformance`'s suite against mock as
part of the workspace), the header parity check, the `struct_size` table
check, the mock bundle fixture parity check, the C smoke test, and a header
compile check under both C11 and C++17. `turbo-provider-ggml` is a workspace
member with no `--exclude`, so these commands build and unit-test it too
(CPU backend only; the `cuda`/`metal`/`vulkan` ggml backends are opt-in
Cargo features, not exercised by CI). CI excludes `turbo-provider-cuda`
(hosted runners have no CUDA toolkit) and does not build the OpenVINO or
Hailo providers (they need the OpenVINO SDK and HailoRT respectively);
these providers are built and their live tests run manually, see
[`providers/openvino/README.md`](../providers/openvino/README.md),
[`providers/cuda/README.md`](../providers/cuda/README.md),
[`providers/hailo/README.md`](../providers/hailo/README.md), and
[`docs/testing.md`](testing.md). Two further CI jobs: the Java binding's
conformance cases run under JDK 25 against the mock provider
(`bindings/java`, `docs/bindings.md`), and `scripts/package.sh --no-cuda
--no-openvino` builds and verifies the distribution archive, uploaded as a
build artifact (`docs/packaging.md`).

## Repository layout

From `PLAN.md` section 9. "Here" means the directory exists in this tree
today (branch `turbo-v2`, 2026-09-21); "planned" means it is scoped for a later milestone.

| path | purpose | status |
|---|---|---|
| `include/turbo/` | generated, committed headers (`turbo.h`, `turbo_types.h`, `turbo_provider.h`) | here |
| `crates/turbo-abi/` | `#[repr(C)]` types, cbindgen config, the provider vtable types | here |
| `crates/turbo-core/` | registry, device discovery, buffers, bundles, tokenizers, chunker, sessions, provider plugin loading | here |
| `crates/turbo-shared/` | links `turbo-capi` into `libturbo` (`cdylib` + `staticlib`) | here |
| `crates/turbo/` | safe Rust API; `builtin_providers()` (mock, static) | here |
| `crates/turbo-conformance/` | provider-agnostic contract suite: C smoke plus a Rust suite (25 files under `tests/`) | here |
| `crates/turbo-bench/` | benchmark harness (batch x seq grid, rerank, generation), receipt writer with budget check, and `turbo-bench discover`, which surveys the providers, devices, features, and runnable bundles on a machine | here |
| `providers/mock/` | mock provider, loadable and statically linked | here |
| `providers/static/` | model2vec-style static embedding provider | here |
| `providers/openvino/` | OpenVINO provider (C++, built separately with CMake) | here |
| `providers/cuda/` | CUDA provider (Rust, ONNX Runtime CUDA EP); EXPERIMENTAL on `krick` (x86_64); embedding landed on Jetson `nano1` | here |
| `providers/ggml/` | ggml/llama.cpp provider (Rust, GGUF generation and embeddings); SUPPORTED on the RTX 4080 SUPER of `krick`, EXPERIMENTAL on the CPU and on `krickert-mac` (Apple M2, Metal) | here |
| `providers/hailo/` | Hailo provider (C++, HailoRT 4.x vstreams); EXPERIMENTAL on the Hailo-8 Pis | here |
| `providers/cpu/` | folded into the CUDA (ONNX Runtime) and ggml providers' CPU devices | not a separate provider |
| `providers/metal/` | Metal provider (Objective-C++, `make` + `clang++`, kernels compiled at load); EXPERIMENTAL on Apple M2 | here |
| `server/` | Inferstream (`turbo-inferstream`): OIP v2 over gRPC and REST with reflection, the extension service (streamed generation, model repository load and unload at run time), OpenAI-shaped routes, session buckets per model; container image and KServe manifests under `packaging/`; verified on the mock bundles and on `krick` with cuda and ggml | here |
| `native/wordpiece/` | shared C++ WordPiece tokenizer, used by the OpenVINO, Hailo and Metal providers | here |
| `native/provider_common/` | shared C++ provider helpers (error boundary, descriptor size checks, bundle reader) | here |
| `native/turbo_buffer/` | shared C++ arenas salvaged from the PoC | present, not yet wired into a provider |
| `bindings/java/` | JDK 25 FFM binding (`ai.pipestream:turbo`) | here |
| `bindings/swift/` | SwiftPM package over the C ABI (`PipestreamTurbo`) | here |
| `bindings/android/` | remaining language binding | planned (P10) |
| `tools/turbo-bundle/` | bundle import, verify, inspect | here (`fetch` from `crates/fetch` is not ported yet) |
| `docs/` | documentation (this tree) | here |
| `testdata/` | fixtures, goldens, receipts | here |
| `scripts/` | `gen-header.sh`, `gen-versioned.py`, `c-smoke.sh`, `package.sh`, `gen-java-ffi.sh` | here |

## Documentation

- [`PLAN.md`](../PLAN.md) — the governing plan: principles, architecture,
  milestones, and acceptance gates.
- [`docs/architecture.md`](architecture.md) — layers, object model,
  provider contract, capability matrix, memory/threading/error models.
- [`docs/c-api.md`](c-api.md) — a walkthrough of every C function
  family, ownership rules, the status-code table, and the option/capability
  mapping.
- [`docs/bundles.md`](bundles.md) — the bundle v2 manifest format and
  verification rules.
- [`docs/providers.md`](providers.md) — the provider table and what a
  provider must implement.
- [`docs/testing.md`](testing.md) — the conformance suite, C smoke,
  header parity, and fixtures.
- [`docs/bindings.md`](bindings.md) — the Java FFM binding: its two
  layers, how the library is located, and the conformance tests.
- [`docs/packaging.md`](packaging.md) — the per-target distribution
  archive `scripts/package.sh` builds, its layout, and what it deliberately
  excludes.
- [`AGENTS.md`](../AGENTS.md) — working rules for agents and contributors.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — toolchain, commands, coding
  standards, review checklist.
- [`docs/history/README.md`](history/README.md) — index of the retired
  proof-of-concept documentation kept for its hardware runbooks and receipts.

## License

[Apache-2.0](../LICENSE).
