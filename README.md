# inferstream

Lean, arch-specific **gRPC streaming inference / embedding servers** in Rust, sharing one wire contract: [KServe Open Inference Protocol (OIP) V2](https://github.com/kserve/open-inference-protocol) with raw byte tensors, bidirectional `ModelStreamInfer`, and bearer auth. Optimized for **raw latency and $/token** — a native, thread-safe alternative to DJL-style serving with no Java or Python hop between the socket and the engine.

One shared core (protocol, auth, routing), **three arch binaries**:

| binary | host class | engines | status |
|---|---|---|---|
| `inferstream-nvidia` | NVIDIA Linux (e.g. **krick**) | **ONNX Runtime CUDA EP** (primary today — encoder embeddings), llama.cpp-CUDA (GGUF, later), TensorRT-LLM Executor (optional later feature for generative) | **ORT engine real** (`ort-runtime` / `ort-cuda`), validated on krick against TEI; TRT-LLM + llama.cpp links stubbed |
| `inferstream-intel` | Intel Linux (e.g. **krick-1**, Arc/Battlemage) | **OpenVINO** (primary): `ovms` gRPC client to the host's Model Server (**live — real GPU embeddings today**) + in-process runtime (stubbed); llama.cpp-SYCL (secondary, Docker-proven) | ovms client working; in-process links stubbed |
| `inferstream-apple` | **native macOS host** (Mac worker) | **MLX**, llama.cpp-Metal for GGUF | engine links stubbed; macOS-only by design; needs a Mac "My Machines" worker for real builds |
| `inferstream` | anywhere | mock only | fully working — dev/client-validation binary |

## Per-arch gap status (honest)

Every arch binary serves **both** gRPC services on one port behind one bearer interceptor: the vendored OIP V2 `inference.GRPCInferenceService` and the `inferstream.v1.InferstreamService` extension (Tokenize / Detokenize / Embed / ListModels / Rerank). Registration lives in the shared `serve()` in `crates/server/src/lib.rs`, and all three binaries reach it through the same `run_cli` path — audited: none of them can start without exposing the extension service.

| capability | `inferstream-nvidia` | `inferstream-intel` | `inferstream-apple` |
|---|---|---|---|
| Embed (real accelerator) | **LIVE** — ORT CUDA EP (CPU EP anywhere), TEI-parity validated on krick | **LIVE** — `ovms` gRPC client to the host Model Server, Battlemage GPU verified on krick-1 | not in-repo — MLX embed CLI exists only on an unpushed Mac branch (see below); `MlxBackend` here is a stub that reports `Unavailable` |
| Tokenize / Detokenize | **LIVE** — ORT engine's own HF tokenizer, or `tokenizer_dir` server-side | **LIVE when `tokenizer_dir` is set** (OVMS pipelines tokenize upstream, so the façade loads its own HF tokenizer); without it: `UNAVAILABLE` with the config fix named | via `tokenizer_dir` only (works today — tokenization is engine-independent); backend delegation is stub |
| Generative infer / stream | mock only — TRT-LLM (`trtllm-sys`) and llama.cpp-CUDA are stubs | mock only — OVMS upstream has no `ModelStreamInfer` (streamed requests adapt to unary, one chunk per request); llama.cpp-SYCL and in-process OpenVINO are stubs | mock only — MLX generation and llama.cpp-Metal are stubs |
| `InferstreamService` registered in `serve()` | yes (shared) | yes (shared) | yes (shared) — binary compiles on Linux for CI, functions only on macOS |
| Rerank | mock scorer only | mock scorer only | mock scorer only |

### Apple MLX branch: not on the remote

The `ai-pipestream/apple-mlx-3ea5` branch (MLX embed CLI work) was never pushed — the Mac worker's push was blocked — so it is **not** integrated here. Integration plan once it can be pushed from the Mac:

1. Push the branch, then rebase it onto current `main` (expected touch points: `crates/backend-apple/src/lib.rs` replacing the stub `MlxBackend`, `crates/arch-apple/src/main.rs` factory wiring, `config/apple.toml`, and this README's apple rows — no shared-crate changes should be needed, the `BackendFactory` seam is designed for exactly this).
2. Gate the real `mlx-rs` link behind a cargo feature (mirroring `ort-runtime` / `ovms`) so Linux CI keeps type-checking the stub.
3. Wire `Embed` through the existing extension service (nothing to add server-side — the typed `Embed` RPC already adapts to any backend's `infer`), and add MLX rows to the GPU-golden harness (`#[ignore]`d, feature-gated, same pattern as `backend-ort/tests/gpu_goldens.rs`).
4. Merge to `main` after `cargo test --workspace` on Linux and a real `cargo test` on the Mac worker.

**Current honest status:** the façade is real — gRPC service, streaming, auth, routing, raw-tensor wire helpers, mock backend, and all three arch binaries build and run today (`cargo test --workspace` passes with zero GPU libraries). **Two real engine paths are live.** NVIDIA: `backend-ort` loads ONNX embedding models (BGE/MiniLM class) through the `ort` crate with server-side tokenization, mean/CLS pooling, and L2 normalization — CPU EP anywhere, CUDA EP on the GPU host — and its output matches TEI on the same model to fp32 tolerance. Intel: `inferstream-intel` with `backend = "ovms"` forwards typed OIP requests to the OpenVINO Model Server already running on krick-1 and returns real GPU embeddings (verified end-to-end: `minilm_pipeline` 384-dim / `mpnet_pipeline` 768-dim through the façade with bearer auth). TRT-LLM, llama.cpp, in-process OpenVINO, and MLX remain stubs with full config surface; routing to them still fails at startup with the exact feature named. Beyond OIP, every binary now also serves the **`inferstream.v1` extension service** — Tokenize/Detokenize (server-side HF tokenizer), a typed `Embed` wrapper, `ListModels`, and a `Rerank` stub — documented below.

## Architecture

```mermaid
flowchart TB
    subgraph clients [Clients - one OIP V2 contract for all hosts]
        C[gRPC: raw tensors + ModelStreamInfer + bearer auth]
    end

    subgraph nvidia [inferstream-nvidia - krick, Linux container OK]
        N1[ONNX Runtime CUDA EP - primary, embeddings live]
        N2[llama.cpp CUDA - GGUF fallback]
        N3[TensorRT-LLM Executor - optional later, generative]
    end

    subgraph intel [inferstream-intel - krick-1, Linux]
        I0[OVMS gRPC client - LIVE, forwards OIP to host Model Server]
        I1[OpenVINO in-process - CPU/GPU/NPU, needs oneAPI env]
        I2[llama.cpp SYCL - Level Zero, secondary]
    end

    subgraph apple [inferstream-apple - native macOS host, NOT containerized]
        A1[MLX]
        A2[llama.cpp Metal]
    end

    C --> nvidia
    C --> intel
    C --> apple
```

Shared plumbing lives in `crates/server` (service, auth interceptor, config, registry) behind a `BackendFactory` seam; each arch binary is a ~100-line factory that wires only its own engines. Nothing engine-specific leaks into the shared layer.

| Crate | Purpose |
|---|---|
| `crates/protocol` | Vendored OIP V2 proto + generated tonic types + raw tensor wire helpers |
| `crates/backend` | `Backend` trait, error types, model → backend `Registry` |
| `crates/backend-mock` | Deterministic mock (embeddings + chunked token streaming) — in every binary |
| `crates/backend-trtllm` | TensorRT-LLM **Executor** skeleton: config + OIP mapping (runtime link behind `trtllm-sys`) |
| `crates/backend-llamacpp` | llama.cpp for all flavors — device = `cuda` / `sycl` / `metal` / `vulkan` / `cpu` |
| `crates/backend-ort` | **ONNX Runtime embedding engine** (feature `runtime`; EPs: CPU / `cuda` / `tensorrt`) — tokenizes server-side, mean/CLS pooling + L2 norm; stub without the feature |
| `crates/backend-openvino` | OpenVINO in-process stub (Intel CPU/GPU/NPU) |
| `crates/backend-ovms` | **Working** OVMS/KServe gRPC client — forwards OIP requests to a running OpenVINO Model Server (typed protobuf, no JSON hop) |
| `crates/backend-apple` | MLX stub (`MlxBackend`) — functional only on macOS |
| `crates/server` | Shared: tonic service, auth, config, registry, CLI runner + mock-only `inferstream` bin |
| `crates/arch-nvidia` | `inferstream-nvidia` binary |
| `crates/arch-intel` | `inferstream-intel` binary |
| `crates/arch-apple` | `inferstream-apple` binary |

## Building each arch binary

Everything below builds on a plain Linux box today (engines are stubs); the extra host requirements kick in when the real engine links land.

```bash
# Dev / client validation (mock only) — anywhere
cargo run -p inferstream-server -- --config config/example.toml

# NVIDIA (krick): stub surface builds anywhere. The real embedding engine is
# ONNX Runtime — `ort`'s download-binaries fetches a matching libonnxruntime
# at build time, so no system ORT install is needed:
cargo build -p inferstream-arch-nvidia --release --features ort-runtime  # CPU EP, anywhere
cargo build -p inferstream-arch-nvidia --release --features ort-cuda     # CUDA EP, GPU host
cargo build -p inferstream-arch-nvidia --release --features trtllm-sys   # optional later: TRT-LLM
./target/release/inferstream-nvidia --config config/nvidia.toml
# See "NVIDIA GPU host requirements" below for the CUDA 13 runtime libs the
# ort-cuda build loads at startup.

# Intel (krick-1): source oneAPI first (build shell AND service unit)
source /opt/intel/oneapi/setvars.sh
cargo build -p inferstream-arch-intel --release
./target/release/inferstream-intel --config config/intel.toml

# Apple: build and run ON THE MAC, never in a container
cargo build -p inferstream-arch-apple --release
./target/release/inferstream-apple --config config/apple.toml
```

Routing a model to an engine a binary doesn't ship fails **at startup** with the exact feature flag or the right binary named — never at request time.

### NVIDIA GPU host requirements (ort-cuda)

The `ort` crate's prebuilt CUDA bundle (ONNX Runtime 1.28) is built against **CUDA 13**, so the CUDA EP needs at runtime:

- NVIDIA driver new enough for CUDA 13 (krick's 595.84 → CUDA 13.2: OK).
- CUDA 13 user-space runtime libs: `libcublasLt.so.13`, `libcublas.so.13`, `libcudart.so.13`, `libnvrtc.so.13`, and a cuDNN 9 built for CUDA 13. A CUDA **12** toolkit (krick's 12.4) does **not** satisfy this.

**Bundled route (recommended, no sudo)** — the repo scripts fetch pinned
NVIDIA pip wheels (~1.6 GB) into `.libs/nvidia/lib` and set the loader path
for you:

```bash
scripts/fetch-runtime-libs.sh nvidia          # once per host; pinned wheel versions
cargo build -p inferstream-arch-nvidia --release --features ort-cuda
scripts/run-nvidia.sh --config config/nvidia.toml   # sets LD_LIBRARY_PATH, execs the binary
```

Alternatives, if you prefer to manage the libs yourself:

```bash
# Manual pip wheels (what the script automates):
python3 -m venv .venv-cuda-libs
.venv-cuda-libs/bin/pip install --only-binary :all: \
    nvidia-cublas nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cudnn-cu13
NV=$PWD/.venv-cuda-libs/lib/python3*/site-packages/nvidia
LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cudnn/lib \
    ./target/release/inferstream-nvidia --config config/nvidia.toml

# With sudo — system install (NVIDIA apt repo):
sudo apt install cuda-runtime-13-2 libcudnn9-cuda-13
```

libonnxruntime itself is fetched at **build time** by `ort`'s
`download-binaries` feature, so no system ORT install is ever needed.

**Intel bundling:** nothing to fetch — `backend = "ovms"` is a pure
tonic/prost gRPC client to the OpenVINO Model Server already running on the
host; it links zero OpenVINO libraries (`scripts/fetch-runtime-libs.sh intel`
prints exactly this). Runtime libs for the in-process OpenVINO backend will
be added to the script when its FFI link lands.

EP registration uses `error_on_failure`: if these libs are missing the binary **fails at startup** with the loader's actual error instead of silently serving on CPU. `device = "tensorrt"` (build feature `ort-tensorrt`) additionally requires TensorRT 10 (`sudo apt install tensorrt-libs` from the NVIDIA repo) — not installed on krick today, so stay on `device = "cuda"`.

## Why per-arch binaries (and why a façade at all)

- **Latency and $/token, not portability theater.** Each accelerator's peak path is a different runtime (TRT-LLM Executor vs Level Zero vs Metal/MLX). One fat binary linking all of them means compromise flags, giant images, and driver conflicts. Three lean binaries mean each host runs exactly its optimum and nothing else.
- **No Java/Python hop.** Unlike DJL (JVM) or Python servers, the socket-to-engine path is a single Rust process; streaming tokens don't cross an interpreter.
- **Not NIM.** NVIDIA NIM wraps engines in an OpenAI-style HTTP service. inferstream keeps engines in-process under its own gRPC (ORT EPs now; TRT-LLM Executor when generative LLMs are mandated). **NIM is used as a benchmark oracle only**: we run NIM beside `inferstream-nvidia` on the same GPU and model to sanity-check our tokens/sec and TTFT — if we're slower than the HTTP wrapper, that's a bug to fix, not a product to adopt.
- **Not OVMS/Triton/TEI.** Those own the process and the protocol; adding an engine or changing streaming/auth policy means forking C++ serving infrastructure. Here the protocol layer is ours, engines are leaf dependencies behind one trait — and clients speak the same OIP V2 they'd speak to Triton anyway. (krick-1 runs OVMS today; because it speaks the same OIP V2 family, `backend = "ovms"` forwards to it as a leaf engine while inferstream keeps auth/routing/streaming — and it doubles as the Intel-side benchmark oracle for the future in-process OpenVINO link.)

## Bake-off methodology (upcoming, per arch)

Metrics collected per engine/model/host, same client, same prompts:

| metric | definition |
|---|---|
| TTFT | request sent → first `ModelStreamInfer` chunk (p50/p95/p99) |
| ITL | inter-token latency between stream chunks (p50/p95) |
| tokens/sec | steady-state decode throughput, per stream and aggregate |
| embed p95 | unary `ModelInfer` latency for the embedding model |
| max concurrent streams | before p95 TTFT doubles |
| watts + $/1M tokens | wall power (or cloud $/hr) ÷ aggregate throughput |

Planned matchups:

- **krick (NVIDIA):** embeddings first — ORT CUDA EP vs ORT TensorRT EP vs llama.cpp-CUDA, with **NIM as oracle** where a comparable NIM exists. TRT-LLM Executor enters the generation matchup only once generative LLMs are mandated.
- **krick-1 (Intel Battlemage):** OpenVINO-GPU (primary; compare against the OVMS instance already on the host as oracle) vs llama.cpp-SYCL (Docker-proven), both via `inferstream-intel` — winner becomes the default `backend` in `config/intel.toml`; both stay compiled in, so switching is a config edit.
- **Mac:** MLX vs llama.cpp-Metal on the same GGUF/MLX model pair (needs the Mac "My Machines" worker for real builds).

Results land in `docs/bakeoff/` as they happen; no numbers are published until they come from these builds on this contract (no vendor-quoted numbers).

## Protocol notes

The proto is vendored at `crates/protocol/proto/open_inference_grpc.proto` from [kserve/open-inference-protocol](https://github.com/kserve/open-inference-protocol), pinned at commit `d49cc23f89d709d87b210ef9449e273ae243984e`.

Upstream OIP defines only unary `ModelInfer`. inferstream adds a clearly marked extension:

- `rpc ModelStreamInfer(stream ModelInferRequest) returns (stream ModelStreamInferResponse)` — the de-facto streaming shape established by NVIDIA Triton's `grpc_service.proto`, so existing streaming clients interoperate.
- Multiple requests can be multiplexed on one stream; every response chunk **echoes the request `id`** for correlation, and a backend may emit many chunks per request (one per generated token). Per-request failures are reported in `error_message` without tearing down the stream.
- `ServerMetadata.extensions` advertises `model_stream_infer` and `inferstream.v1`.

## INFERSTREAM EXTENSION: `inferstream.v1.InferstreamService`

A second, clearly separated gRPC service (`crates/protocol/proto/inferstream_extension.proto`) runs on the same endpoint with the same bearer auth. The vendored OIP service is untouched — clients that only speak OIP lose nothing (every `Embed` is expressible as `ModelInfer` with a BYTES `text` tensor plus the `pooling` / `normalize` / `truncate` InferParameters).

| RPC | what it does |
|---|---|
| `Tokenize` | batch tokenization with the model's server-side tokenizer: `input_ids`, `attention_mask`, token strings, byte offsets on request, optional truncation and pad-to-longest |
| `Detokenize` | inverse of Tokenize; `skip_special_tokens` drops CLS/SEP/PAD frames |
| `Embed` | typed convenience wrapper over ModelInfer — send strings, get `float` vectors back; pooling/normalize/truncate forwarded as InferParameters |
| `ListModels` | one call for the whole repository: name, backend id, readiness, platform, embedding dim, tokenizer availability |
| `Rerank` | query/document relevance scores, sorted, with `top_n`; engines without a reranker answer `UNAVAILABLE` (the mock implements a deterministic scorer so the wire path tests everywhere) |

Tokenizer resolution: when a model's config sets `tokenizer_dir` (a `tokenizer.json` file or a directory containing one), the server loads a local HuggingFace fast tokenizer at startup and answers Tokenize/Detokenize itself — including for `backend = "ovms"` models whose upstream pipelines tokenize server-side on the Model Server. Without a configured tokenizer the request is delegated to the backend (the ORT engine reuses its own HF tokenizer; the mock ships a lossless byte-level tokenizer; everything else reports `UNAVAILABLE` with the config fix named).

grpcurl examples (mock server from `config/example.toml`):

```bash
PROTO="-proto crates/protocol/proto/inferstream_extension.proto"

# Repository listing: names, backends, readiness, dims
grpcurl -plaintext $PROTO 127.0.0.1:8461 inferstream.v1.InferstreamService/ListModels

# Batch tokenize with offsets and padding
grpcurl -plaintext $PROTO \
  -d '{"model_name":"mock-embed","texts":["hello world","hi"],"with_offsets":true,"pad_to_longest":true}' \
  127.0.0.1:8461 inferstream.v1.InferstreamService/Tokenize

# Detokenize (skip special tokens)
grpcurl -plaintext $PROTO \
  -d '{"model_name":"mock-embed","sequences":[{"ids":[1,107,108,2]}],"skip_special_tokens":true}' \
  127.0.0.1:8461 inferstream.v1.InferstreamService/Detokenize

# Embed without touching raw tensors
grpcurl -plaintext $PROTO \
  -d '{"model_name":"mock-embed","texts":["hello world"],"pooling":"mean","normalize":true}' \
  127.0.0.1:8461 inferstream.v1.InferstreamService/Embed

# Rerank with top_n
grpcurl -plaintext $PROTO \
  -d '{"model_name":"mock-embed","query":"rust inference","documents":["cooking pasta","rust inference server"],"top_n":1}' \
  127.0.0.1:8461 inferstream.v1.InferstreamService/Rerank
```

With bearer auth enabled add `-H 'authorization: Bearer <key>'` — the extension service sits behind the same interceptor.

Raw tensor rules (helpers in `inferstream_protocol::tensor`):

- Fixed-size dtypes (`FP32`, `INT64`, …) are flat, row-major, **little-endian** byte blobs in `raw_input_contents` / `raw_output_contents`; blob length must equal `element_count(shape) * element_size`.
- `BYTES` elements are length-prefixed with a little-endian `u32`.
- `FP16` / `BF16` exist only in raw form — one more reason raw is the primary path.

Generation contract (all engines, identical to what the mock emits today): input `text` (BYTES) or `input_ids` (INT32); each stream chunk carries `token` (BYTES) and sets the bool parameter `final` on the terminal chunk. Embeddings: unary, output `embedding` FP32 `[d]`. Clients don't change between the mock and a GPU engine.

## Testing & reference embeddings (goldens)

```bash
cargo test --workspace          # 90+ tests, passes with zero GPU libraries
```

Fixed prompts (short / medium / empty / unicode / long-truncation) live as JSON goldens in `testdata/reference_embeddings/`. The deterministic-mock goldens run on every `cargo test` (cosine ≥ 0.999 plus exact-value and L2 checks, both directly and end-to-end through the `Embed` RPC); regenerate them with `cargo run -p inferstream-server --example gen_reference_embeddings`. GPU goldens for the real engines are `#[ignore]`d and feature-gated (`cargo test -p inferstream-backend-ort --features cuda -- --ignored gpu_golden` on krick) — see [`testdata/reference_embeddings/README.md`](testdata/reference_embeddings/README.md) for the schema and the krick/krick-1 regeneration walkthrough.

## Trying it now

```bash
cargo test --workspace          # passes with no GPU libs
cargo run -p inferstream-server -- --config config/example.toml
# in another shell — unary embed + two multiplexed bidi streams:
cargo run -p inferstream-server --example client -- http://127.0.0.1:8461
```

grpcurl works against any binary with the vendored proto:

```bash
grpcurl -plaintext -proto crates/protocol/proto/open_inference_grpc.proto \
  127.0.0.1:8461 inference.GRPCInferenceService/ServerLive

# Unary infer: "hi" as a length-prefixed BYTES tensor (base64 of 02 00 00 00 68 69)
grpcurl -plaintext -proto crates/protocol/proto/open_inference_grpc.proto \
  -d '{"model_name":"mock-embed","id":"r1","inputs":[{"name":"text","datatype":"BYTES","shape":[1]}],"raw_input_contents":["AgAAAGhp"]}' \
  127.0.0.1:8461 inference.GRPCInferenceService/ModelInfer
```

## Auth

Static **bearer tokens (API keys)** enforced by a tonic interceptor on every RPC, on every binary:

```toml
[auth]
mode = "bearer"
bearer_tokens = ["change-me"]     # and/or the env var below
```

```bash
INFERSTREAM_API_KEYS="key-a,key-b" ./target/release/inferstream-nvidia --config config/nvidia.toml
```

Clients send gRPC metadata `authorization: Bearer <key>` (`grpcurl -H 'authorization: Bearer key-a' …`). Comparison is constant-time per candidate. Bearer keys need an encrypted transport: TLS/mTLS is the next auth step — tonic's `ServerTlsConfig` (with `client_ca_root` for mTLS) slots into `serve()` in `crates/server/src/lib.rs` without touching the service layer. Until then terminate TLS in front or stay on trusted networks. No quotas by design.

## Deployment: Apple hosts vs Linux containers

NVIDIA CUDA passes into Linux containers via the NVIDIA container toolkit. **Apple GPU (Metal) and the Neural Engine do not** — there is no macOS container GPU passthrough, and Linux containers on a Mac run inside a VM without Metal. Therefore:

- `inferstream-nvidia` and `inferstream-intel` deploy as Linux containers (or bare processes) on their GPU hosts.
- `inferstream-apple` deploys **natively on the Mac** (launchd service or plain process). It compiles on Linux as a stub so CI type-checks the wiring, but it only functions on macOS.

Same client, same contract, heterogeneous fleet: krick (NVIDIA) + krick-1 (Intel) + Mac workers behind one OIP endpoint shape.

## Roadmap

1. ~~ONNX Runtime session wiring (`backend-ort`)~~ — **done**: CPU + CUDA EPs, embeddings live on krick; TensorRT EP wired but blocked on host TensorRT libs.
2. ~~**OVMS client backend**~~ — **done**: `backend = "ovms"` serves real Battlemage-GPU embeddings on krick-1 through the façade today (see `config/intel.toml` and `crates/arch-intel/examples/ovms_embed.rs`).
3. **OpenVINO runtime link** (`backend-openvino`) — krick-1's in-process path (the live OVMS route as oracle); llama.cpp-SYCL as the Docker-proven secondary.
4. **llama.cpp FFI** (`backend-llamacpp`) — one binding, all devices (CUDA secondary on krick, SYCL secondary on krick-1, Metal on Mac).
5. **MLX** (`backend-apple`) via `mlx-rs` — needs the Mac "My Machines" worker for real builds.
6. **TRT-LLM Executor FFI** (`backend-trtllm`, feature `trtllm-sys`) — optional later feature for generative models; cxx/bindgen layer over `tensorrt_llm::executor`.
7. **TLS / mTLS** in `serve()`; per-key model ACLs after.
8. Optional adapters: TEI-compatible proto (lowest priority), shared-memory tensor hints, richer stream metadata.
9. ORT session pooling (today one session per model behind a mutex; ONNX Runtime's intra-op threads still parallelize each request).

Out of scope: dual independent pub/sub subscribe streams ("Surface 1") — request-scoped bidi only. No NIM HTTP wrapping, ever.

## License

[Apache-2.0](LICENSE). Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
