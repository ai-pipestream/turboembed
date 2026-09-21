# Turbo

Turbo is a native inference and embedding library: one C ABI (`turbo_`
prefix, `libturbo`), one Rust safe API on top, and hardware support added as
runtime-loaded provider libraries (`libturbo_provider_<name>`) rather than
Cargo feature flags. The governing design document is [`PLAN.md`](PLAN.md);
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
"Load a provider library" below and [`docs/providers.md`](docs/providers.md).

Per-model truth (pooling, normalization, sequence limit, prefixes, labels,
dimension, dtype) lives in a bundle directory's `bundle.json` manifest,
verified by content hash before anything loads. See
[`docs/bundles.md`](docs/bundles.md).

## Status

This tree is at commit `13b58ff` on branch `turbo-v2`: milestones P0 and P1
of `PLAN.md` section 10 are done, P2 (the OpenVINO provider) has landed its
embed, rerank, classify, and token-classify tasks, and P3 (the CUDA
provider) has landed the same four tasks on x86_64. Every per-call option
now either honors exactly what the caller asked for, gated by a
`TURBO_CAP_OPT_*` bit, or fails; see [`docs/c-api.md`](docs/c-api.md)'s
capability-bit tables. What exists today:

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
- Two more loadable providers, each built separately: `openvino`
  (`providers/openvino`, a C++ library built with CMake, implementing
  `turbo_provider.h` directly rather than through the Rust
  `export_provider!` macro) with embed, rerank, classify, and token-classify
  `EXPERIMENTAL` on GPU and CPU; and `cuda` (`providers/cuda`, a Rust
  library using `export_provider!`, built with `cargo build -p
  turbo-provider-cuda`) with the same four tasks `EXPERIMENTAL` on `krick`
  (RTX 4080 SUPER, x86_64) through the ONNX Runtime CUDA execution provider.
  See [`providers/openvino/README.md`](providers/openvino/README.md),
  [`providers/cuda/README.md`](providers/cuda/README.md), and
  [`docs/providers.md`](docs/providers.md).
- `tools/turbo-bundle`: `import` (derives a bundle's contract from a source
  model's own files), `verify`, and `inspect`. See
  [`docs/bundles.md`](docs/bundles.md).
- `crates/turbo-conformance`: a C smoke test (`c/smoke.c`) and a
  provider-agnostic Rust suite (over 200 tests across 24 files under
  `tests/`) covering the P0 groups from `PLAN.md` section 10 plus live
  provider tests (`tests/live_embed.rs`, `tests/live_tasks.rs`) that skip
  themselves without hardware. See [`docs/testing.md`](docs/testing.md).
- Mock bundle fixtures under `testdata/bundles/mock/` (embedding, reranker,
  classifier, token-classifier, generative, generic) and a tokenizer-only
  fixture at `testdata/bundles/minilm-tokenizer/`, all generated or imported,
  not hand-written.

What is declared in the header but **not implemented** in this build: the
push-style `turbo_generate` (P6, stubs the pull iterator). It returns
`TURBO_E_NOT_IMPLEMENTED` rather than doing something silently different.
Every other function family declared in the header — including
`turbo_runtime_load_provider`, `turbo_tokenizer_*`, and `turbo_chunk_plan_*`
— is implemented; see [`docs/c-api.md`](docs/c-api.md) for the full picture.

Hardware providers for Metal, Hailo, and ggml generation, and the
Java/Swift/Android bindings, are milestones P4 through P10 and are **not
available yet**. CUDA on Jetson (`nano1`, aarch64) is also not available yet
(`PLAN.md` section 10, P3); CUDA on x86_64 (`krick`) has landed.

## Capability and hardware status

Every provider reports a capability cell per (device, task, modality) as
`UNSUPPORTED`, `PLANNED`, `EXPERIMENTAL`, or `SUPPORTED`
(`TURBO_CAP_*` in `turbo_provider.h`), plus measured precision figures once a
qualification receipt exists (`PLAN.md` section 4.4). Today:

| provider | status | notes |
|---|---|---|
| `mock` | supported for contract testing only | deterministic, hash-derived; serves only `mock`-artifact bundles; never a real model |
| `static` | EXPERIMENTAL | one capability cell, `EMBED x TEXT x CPU`; table lookup + mean + L2 on host, explicit device selection only; precision receipt against model2vec still pending |
| `openvino` GPU | EXPERIMENTAL | embed, rerank, classify, token-classify on `krick-1` (Battlemage B70); fused mean+L2 graph, device-resident results; receipts: [`testdata/receipts/turbo/openvino-minilm-2026-09-21.json`](testdata/receipts/turbo/openvino-minilm-2026-09-21.json), [`openvino-tasks-2026-09-21.json`](testdata/receipts/turbo/openvino-tasks-2026-09-21.json) |
| `openvino` CPU | EXPERIMENTAL | same tasks, explicit selection only; same receipts |
| `cpu` | PLANNED (P3, folded into the CUDA/ORT provider work) | ORT CPU EP; ggml CPU |
| `cuda` | EXPERIMENTAL on `krick` (x86_64); Jetson (`nano1`, aarch64) not started | embed, rerank, classify, token-classify through the ONNX Runtime CUDA EP with device-side pooling/normalization/activation kernels; cosine 1.000 against the FP32 references; receipt: [`testdata/receipts/turbo/cuda-2026-09-21.json`](testdata/receipts/turbo/cuda-2026-09-21.json) |
| `metal` | PLANNED (P4) | MLX over the shared Metal arena |
| `hailo` | PLANNED (P5) | Hailo-8/8L, Hailo-10H |
| `ggml` | PLANNED (P6) | GGUF generation, CUDA/SYCL/Metal/CPU |

A matched-native benchmark receipt is still required before OpenVINO,
`cuda`, or `static` can move from `EXPERIMENTAL` to `SUPPORTED` (`PLAN.md`
section 2, item 7); all three receipts above record this explicitly under
`status_after`.

See [`docs/providers.md`](docs/providers.md) for the full table (hardware,
runtime, lowest layer, machine) from `PLAN.md` section 7, and
[`docs/architecture.md`](docs/architecture.md) for the capability matrix
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

See [`docs/providers.md`](docs/providers.md) for the ownership rules a
provider library must follow and [`providers/openvino/README.md`](providers/openvino/README.md)
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
compile check under both C11 and C++17. CI excludes `turbo-provider-cuda`
(hosted runners have no CUDA toolkit) and does not build the OpenVINO
provider (it needs the OpenVINO SDK); both providers are built and their
live tests run manually, see
[`providers/openvino/README.md`](providers/openvino/README.md),
[`providers/cuda/README.md`](providers/cuda/README.md), and
[`docs/testing.md`](docs/testing.md).

## Repository layout

From `PLAN.md` section 9. "Here" means the directory exists in this tree
today (commit `13b58ff`); "planned" means it is scoped for a later milestone.

| path | purpose | status |
|---|---|---|
| `include/turbo/` | generated, committed headers (`turbo.h`, `turbo_types.h`, `turbo_provider.h`) | here |
| `crates/turbo-abi/` | `#[repr(C)]` types, cbindgen config, the provider vtable types | here |
| `crates/turbo-core/` | registry, device discovery, buffers, bundles, tokenizers, chunker, sessions, provider plugin loading | here |
| `crates/turbo-shared/` | links `turbo-capi` into `libturbo` (`cdylib` + `staticlib`) | here |
| `crates/turbo/` | safe Rust API; `builtin_providers()` (mock, static) | here |
| `crates/turbo-conformance/` | provider-agnostic contract suite: C smoke plus a Rust suite (24 files under `tests/`) | here |
| `crates/turbo-bench/` | matched-native benchmark harness | planned (used from P2 on; no matched-native benchmark receipt exists yet) |
| `providers/mock/` | mock provider, loadable and statically linked | here |
| `providers/static/` | model2vec-style static embedding provider | here |
| `providers/openvino/` | OpenVINO provider (C++, built separately with CMake) | here |
| `providers/cuda/` | CUDA provider (Rust, ONNX Runtime CUDA EP); EXPERIMENTAL on `krick` (x86_64), Jetson (`nano1`) not started | here |
| `providers/cpu/`, `providers/hailo/`, `providers/ggml/` | remaining hardware providers | planned (P3, P5, P6) |
| `providers/metal/` | Swift package producing `libturbo_provider_metal.dylib` | planned (P4) |
| `native/wordpiece/` | shared C++ WordPiece tokenizer, used by the OpenVINO provider | here |
| `native/turbo_buffer/` | shared C++ arenas salvaged from the PoC | present, not yet wired into a provider |
| `bindings/java/`, `bindings/swift/`, `bindings/android/` | language bindings | planned (P7, P10) |
| `tools/turbo-bundle/` | bundle import, verify, inspect | here (`fetch` from `crates/fetch` is not ported yet) |
| `server/` | Inferstream on the new ABI | planned (P9) |
| `docs/` | documentation (this tree) | here |
| `testdata/` | fixtures, goldens, receipts | here |
| `scripts/` | `gen-header.sh`, `gen-versioned.py`, `c-smoke.sh` | here |

## Documentation

- [`PLAN.md`](PLAN.md) — the governing plan: principles, architecture,
  milestones, and acceptance gates.
- [`docs/architecture.md`](docs/architecture.md) — layers, object model,
  provider contract, capability matrix, memory/threading/error models.
- [`docs/c-api.md`](docs/c-api.md) — a walkthrough of every C function
  family, ownership rules, the status-code table, and the option/capability
  mapping.
- [`docs/bundles.md`](docs/bundles.md) — the bundle v2 manifest format and
  verification rules.
- [`docs/providers.md`](docs/providers.md) — the provider table and what a
  provider must implement.
- [`docs/testing.md`](docs/testing.md) — the conformance suite, C smoke,
  header parity, and fixtures.
- [`AGENTS.md`](AGENTS.md) — working rules for agents and contributors.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — toolchain, commands, coding
  standards, review checklist.
- [`docs/history/README.md`](docs/history/README.md) — index of the retired
  proof-of-concept documentation kept for its hardware runbooks and receipts.

## License

[Apache-2.0](LICENSE).
