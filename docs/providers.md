# Providers

A provider is a set of devices plus the ability to run tasks on them,
implementing the traits in `crates/turbo-core/src/provider.rs`. Today the
only provider in this tree is `mock`. Every other row below is planned; see
`PLAN.md` sections 7 and 10 for the full hardware/runtime detail and
milestone gates this table summarizes.

## Provider table

| provider | status | hardware / machine | runtime | lowest layer used |
|---|---|---|---|---|
| `mock` | supported for contract testing only | any host; two synthetic devices (CPU ordinal 0, Accel ordinal 1) | none (pure Rust) | deterministic hash-derived math on host; serves only bundles with a `mock` artifact |
| `static` | PLANNED (P1) | any CPU | none (pure Rust/C++, model2vec-style static token embeddings) | table lookup + mean + L2 on host; one capability cell `EMBED x TEXT x CPU` |
| `cpu` | PLANNED (P1) | any; explicit selection only, never `AUTO` | ORT 1.30 CPU EP; ggml CPU for GGUF | host arena, write-through tokens |
| `openvino` | PLANNED (P2) | `krick-1` Battlemage B70; Intel NPU when available | OpenVINO 2026.4.0 | `ov::Core` compiled model with mean+L2 fused into the graph; `ClContext`/`ZeroContext` remote tensors |
| `cuda` | PLANNED (P3) | `krick` RTX 4080 (x86_64); `nano1` Orin Nano Super (aarch64) | ORT 1.30 CUDA EP; TensorRT EP; llama.cpp CUDA | IoBinding on pinned/device arena, external allocator pool, device mean+L2 kernel |
| `metal` | PLANNED (P4) | Apple M2 | MLX 0.32 via mlx-swift 0.31 | MLX arrays over Metal shared buffers, no-copy construction; pooling and L2 as MLX ops |
| `hailo` | PLANNED (P5) | two Pis with Hailo-8; one Pi with Hailo-10H; x86_64 hosts with a PCIe Hailo-8 card | HailoRT 4.24.0 / 5.4.0 | `VDevice` + `InferModel` with `dma_map` zero-copy on page-aligned arena rows |
| `ggml` | PLANNED (P6) | every machine | llama.cpp v0.4.1 / ggml 0.24 (CUDA, SYCL, Metal, CPU backends) | `ggml_backend_dev` registry; `llama_batch` decode; embeddings copied once from `llama_get_embeddings_seq` |
| `hailo` GenAI | PLANNED (P6) | Hailo-10H Pi | `hailort::genai::LLM` (HailoRT 5.4) | native LLM on the NPU with its own sampler |

The `mock` row is the only one with code behind it (`crates/turbo-core/src/mock.rs`, packaged as the `turbo-provider-mock` crate under `providers/mock/`). It exists to exercise the contract without hardware and is documented in the capability matrix itself as "mock: deterministic hash-derived outputs for contract testing; never a real model" — see the `notes` field `MockProvider::capability` returns.

## What a provider must implement

From `crates/turbo-core/src/provider.rs` (see also `AGENTS.md`'s "Adding a
provider" section):

- `Provider`: `id()`, `version()`, `devices()` (enumerate hardware; a probe
  failure is recorded and logged, not a panic), `capability(ordinal, task,
  modality)`, `can_run(ordinal, bundle, task, modality)`, `create_context`.
- `ProviderContext`: `alloc`, `import`, `load_model`.
- `ProviderModel`: `info()`, `create_session`, and `create_generation` for
  generative models (default: `TURBO_E_UNSUPPORTED_TASK`).
- `ProviderSession`: `write_text`, `write_tokens`, `write_pairs`,
  `write_text_classify`, `bind`, `run`, `stats`. A provider only overrides
  the write methods for the tasks its capability matrix actually offers;
  every other method's default already fails loudly.
- `ProviderGeneration`: `prompt`, `prompt_tokens`, `step`, `cancel`.

The provider vtable is defined at task granularity, not graph granularity
(`PLAN.md` section 4.7): a provider is free to run `embed`, `rerank`,
`classify`, `token_classify`, `generate`, and `run` as one fused device
pipeline. `turbo-core`'s own tokenize-then-run-then-pool sequence is a
fallback for stages a provider declines to implement, never a hidden
default a provider's own fused path is measured against.

## Honesty rules

These are `PLAN.md` section 2's principles, applied specifically to what a
provider reports:

- **Stage placement.** `ModelInfo::stages` (`stage_placement[TOKENIZE|
  ENCODE|POOL|NORMALIZE|POSTPROCESS]`) must reflect where each stage
  actually ran (`HOST`, `DEVICE`, or `FUSED`), not where the provider
  intends it to run eventually. `fully_accelerated` is `1` only if every
  applicable stage is off the host.
- **Capability status.** `Provider::capability` returns one of
  `UNSUPPORTED`, `PLANNED`, `EXPERIMENTAL`, `SUPPORTED`. A cell is marked
  `SUPPORTED` only once it carries a conformance receipt, a precision
  receipt (cosine floor and max absolute error against a reference dtype),
  and a matched-native benchmark from a named machine (`PLAN.md` section 2,
  item 7, and section 11). `PLANNED` and `EXPERIMENTAL` are honest,
  non-blocking states to use instead of overclaiming; the `mock` provider's
  own capability cells return `SUPPORTED` because its outputs really are a
  deterministic, fully-tested function of the input, not because
  `SUPPORTED` is a convenient default.
- **Capability bits gate options, not silently ignore them.** Every
  `TURBO_CAP_OPT_*` bit either matches what the provider actually honors, or
  the corresponding option field must fail with
  `TURBO_E_UNSUPPORTED_OPTION` (`turbo-core` enforces the rejection side of
  this centrally in `Model::validate_*`; the provider only needs to
  advertise its bits honestly and implement what it advertises).
- **Device probing failures are visible, not swallowed.** `Runtime::register`
  records a provider's device-probe failure as a `ProviderFailure` and logs
  it; the provider stays registered with zero devices rather than the
  runtime silently pretending the provider does not exist.

See `docs/architecture.md` for how the capability matrix, memory model, and
provider contract fit together, and `docs/testing.md` for how a provider is
expected to be run against the conformance suite once it exists.
