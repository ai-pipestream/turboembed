# TurboEmbed architecture

This page maps the existing ABI and server wiring. The
[library definition](library-design.md) and [roadmap](../ROADMAP.md) govern new
work on prepared inputs, device buffers, Java bindings, and later integration.
See the [review](code-review-2026-09-13.md) for known contract defects; historical
status labels below are not certification of the current checkout.

TurboEmbed exposes in-process embedding execution through a C ABI, with Rust
and Swift wrappers. Desktop Java bindings are planned. Inferstream's NVIDIA,
Intel, and Swift Apple servers consume the library and expose OIP V2 and
`inferstream.v1`. Native callers pass pointer-and-length views directly.

## Layer cake

```mermaid
flowchart TB
    subgraph clients [Clients]
        Rust["Rust crate turboembed"]
        C["C / C++"]
        Swift["Swift (Apple)"]
        Java["Java planned: FFM / JNI"]
    end

    subgraph grpc [gRPC — thin bonus]
        Ext["inferstream.v1.InferstreamService"]
        Embed["Embed unary + batch"]
        Stream["EmbedStream"]
        Packed["typed float[] or packed LE FP32 bytes"]
    end

    subgraph abi [Frozen C ABI — include/turboembed.h]
        H["create / destroy / list / load / embed / embed_stream / free"]
    end

    subgraph impl [Arch implementations — same symbols]
        Cpp["C++ mock-smoke default + ORT CUDA/TensorRT + OpenVINO GenAI"]
        Mlx["Swift @_cdecl → MLXEmbedders mean+L2 (Metal)"]
    end

    subgraph engines [Engines behind the ABI]
        ORT["ORT CUDA / CPU EP"]
        GenAI["OpenVINO compiled model + native tokenizer"]
        Apple["swift/ MlxEngine mean+L2"]
        Mock["mock-embed — ABI smoke only"]
    end

    Rust --> abi
    C --> abi
    Swift --> abi
    Java -. planned .-> abi
    grpc --> abi
    Ext --> Embed
    Ext --> Stream
    Embed --> Packed
    abi --> Cpp
    abi --> Mlx
    Cpp --> ORT
    Cpp --> GenAI
    Cpp --> Mock
    Mlx --> Apple
```

| Layer | Who owns it | Status |
|---|---|---|
| C header `include/turboembed.h` | Frozen ABI v1 | landed |
| C++ `native/turboembed` | nvidia / intel / Linux CI | default no-feature link is mock smoke (`mock-embed` on explicit `MOCK`/`CPU` only). `--features ort-cuda` wires CUDA + CPU + TensorRT MiniLM; `--features genai` wires GPU + CPU. GPU request never silently becomes CPU or mock |
| Swift `@_cdecl` shim | Apple (same header) | **LIVE** — `libTurboEmbed.dylib` → `MlxEngine` mean+L2 on Metal (Machine C). Rust `inferstream-apple` is a Linux CI compile stub only |
| Rust `crates/turboembed` | safe view wrapper over the C ABI | ABI smoke + `mlx-live` / `ort-cuda` / `genai` receipts |
| gRPC `Embed` / `EmbedStream` | thin façade over the C ABI | **LIVE** — catalog aliases call TurboEmbed on all three arches |
| inferstream arch servers | LLM + mock unchanged | catalog embeds (`minilm`, …) go through `inferstream-backend-turboembed` / Swift `TurboEmbedBackend` |

## ABI ownership rules

These are part of the contract, not style notes. Full text lives in the
header comment.

1. **Inputs are views.** `turboembed_str { ptr, len }` and `const char *` +
   `len` are borrowed. The caller keeps the allocation for the duration of
   the synchronous call. Strings are UTF-8. `config_path` and name helpers
   use C strings; the load-model contract also treats `alias_len == 0` as
   a NUL-terminated alias. Safe wrappers must uphold that exception.
2. **Outputs are engine-owned.** `turboembed_embed_result` and
   `turboembed_model_info` lists are released with the matching
   `*_free`. Do not `free` inner pointers. Destroying the engine
   invalidates anything it allocated.
3. **Packed bytes alias typed floats.** The result's `values` (FP32
   row-major) and `packed` (little-endian byte view of the same buffer)
   share one allocation. Releasing the result releases both.
4. **Stream callback pointers die at return.** Copy the row inside
   `turboembed_stream_cb` if you need it later.
5. **One engine, one thread.** Distinct engines may run concurrently.
6. **No silent CPU.** GPU/Metal/AUTO/NPU requests fail if that
   accelerator is missing. `CPU` / `OPENVINO_CPU` only when selected.

The Rust `Embeddings` guard retains the native engine and bounds returned
`&[f32]` views by the guard's lifetime. Native calls and result release are
serialized. Stream callbacks cannot reenter the same engine; with unwinding
enabled, callback panics resume in Rust after the native call returns.
The Swift wrapper also retains its engine through result release and serializes
wrapper operations. Its native macOS regression tests remain to be run.

## Provider plugin sketch

Real engines stay behind a vtable so we can add **model2vec** (or any other
encoder) without growing the public ABI.

```c
typedef struct turboembed_provider_vtbl {
    const char *id;          /* "ort", "openvino-genai", "mlx", "model2vec" */
    turboembed_status (*load)(void *ctx, const char *alias, size_t alias_len);
    turboembed_status (*embed)(void *ctx, const turboembed_str *texts,
                               size_t n, const turboembed_embed_options *opts,
                               turboembed_embed_result **out);
    void *ctx;
} turboembed_provider_vtbl;

turboembed_status turboembed_register_provider(const turboembed_provider_vtbl *);
```

Today `turboembed_register_provider` returns `NOT_IMPLEMENTED`. The
default no-feature link answers `mock-embed` itself. inferstream catalog
Embed does not call this vtable; it calls the frozen create/load/embed
symbols.

Wiring map (do not invent a fourth runtime):

| Provider id | Host | Existing code to call later |
|---|---|---|
| `ort` | nvidia | `crates/backend-ort` (CUDA EP / CPU EP / TensorRT EP) |
| `openvino-genai` | intel | `native/turboembed/src/genai.cpp`: OpenVINO compiled model and native WordPiece tokenizer |
| `mlx` | apple | `swift/Sources/MlxEngine` (`MLXEmbedders`) |
| `mock` | any | default no-feature C++ link / `backend-mock` — ABI smoke only |
| `model2vec` | later | plugin only — not shipped |

Aliases stay the catalog names (`minilm`, `bge-small`, …). Clients never
learn which provider served the vector.

## gRPC surface

TurboEmbed does **not** add a second gRPC service. It maps onto the
existing [`inferstream.v1.InferstreamService`](../proto/inferstream_extension.proto)
`Embed` RPC — already a typed convenience over OIP `ModelInfer`.

| C ABI | gRPC | Notes |
|---|---|---|
| `turboembed_list_models` | `ListModels` | same catalog aliases |
| `turboembed_load_model` | *(startup `serve` / implicit)* | gRPC servers load at process start |
| `turboembed_embed` / `_one` | `Embed` | `texts[]` is already a batch |
| `turboembed_embed_stream` | `EmbedStream` | unary request, streamed rows |
| `values` (typed) | `EmbedResponse.embeddings[].values` | default `TYPED` |
| `packed` | `EmbedResponse.packed_embeddings` | `PACKED_BYTES` — LE FP32 `[n*d]` |

`EmbedRequest.output_format` selects typed **or** packed. Packed is the
cheap path for Java / Rust clients that want one `bytes` copy matching
OIP `raw_output_contents`. The façade writes that blob into a rented
output-scratch slab (`docs/grpc-output-scratch.md`, SOLIDIFY 6) — after
warmup the dest `Vec` is reused; prost still copies onto the wire.
Default stays typed so today's grpcurl and `inferstream-e2e` clients do
not change.

OIP `ModelInfer` remains the interoperable raw-tensor path (BYTES `text`
in, FP32 `embedding` out). Every TurboEmbed embed is expressible as
`ModelInfer`.

## Java note

The planned desktop Java API targets JDK 25+ and runs in process through our
own Panama FFM adapter over the native ABI. It follows native ownership, device selection,
and error semantics, with prepared buffers as well as text convenience calls.
The FFM adapter follows native contract and packaging work in
[roadmap M3](../ROADMAP.md#m3-deliver-desktop-java-on-jdk-25-through-ffm).

An optional remote client can use grpc-java stubs from the existing proto.
Remote requests retain typed or packed LE FP32 responses. The gRPC adapter
is a separate transport choice and is not required for local Java inference.
Android will receive its JNI adapter and Android-capable backend in M5; desktop FFM support
does not establish Android runtime support.

## Inferstream servers are façades

Catalog embed aliases (`minilm` first; every `serve` embed that the
arch actually lists) are **not** a second embed stack. The arch gRPC
servers construct `TurboEmbedBackend` (Rust nvidia/intel) or Swift
`TurboEmbedBackend` (Apple) and call `turboembed_engine_create` /
`turboembed_load_model` / `turboembed_embed`.

| arch binary | catalog `backend` | TurboEmbed device | provider feature |
|---|---|---|---|
| `inferstream-nvidia` | `ort` | `AUTO` / `CUDA` (catalog default `cuda`); explicit `cpu` is CPU EP; `tensorrt` is ORT TensorRT EP | `--features ort-cuda` |
| `inferstream-intel` | `openvino` | `AUTO` / `OPENVINO_GPU` (catalog default `GPU`); explicit `CPU`; `NPU` fails loud until a host lists the plugin (`intel-npu.json`) | `--features openvino-genai` |
| `inferstream-apple` (Swift) | `mlx` + `pooling` set | `AUTO` = Metal | `libTurboEmbed.dylib` on a Mac |

Without the provider feature (or without the accelerator) **startup
fails** with the feature / host named. Catalog aliases never sit on
`backend-mock` and never get dim-8 FNV.

`backend-ort` / `backend-openvino` remain as libraries the ABI
providers reuse (pooling math, ORT session). The servers do not
construct those backends for catalog embeds.

LLM paths are untouched: `backend = "llama-cpp"` (nvidia/intel) and
Apple `mlx` aliases **without** `pooling` (default-llm / qwen-*) still
use llama.cpp / mlx-swift-lm for Tokenize and `ModelStreamInfer`.

## Remaining gaps

These are the things TurboEmbed still does **not** do. The C ABI itself
is live (ORT CUDA / CPU / TensorRT, Intel GenAI GPU / CPU, Apple MLX
Metal) — see receipts under `testdata/receipts/turboembed/`.

- Does not replace Tokenize / StreamInfer for generative GGUF / MLX LLMs.
- Does not add Python bindings.
- Does not invent a second gRPC service.
- **model2vec** is not shipped. `turboembed_register_provider` returns
  `NOT_IMPLEMENTED`.
- **Zero-copy buffer pool** is **TurboBuffer** (`include/turbo_buffer.h`).
  Machine A ORT CUDA + explicit CPU rent tokens / hidden / results from
  the engine arena (PINNED+DEVICE or HOST). ORT CUDA EP allocations go
  through `gpu_external_alloc` → DEVICE rent. Steady-state embed after
  warmup is 0 arena allocs. CUDA mean+L2 runs on DEVICE into mapped
  PINNED; `d2h_hidden_bytes` is 0. The 384-d row the caller reads is
  `result_host_bytes` (≪ hidden volume). Intel GenAI GPU rents ZE SHARED USM
  (`docs/turboembed-genai-ze-machine-b.md`). Apple `libTurboEmbed.dylib`
  rents Metal SHARED tokens, last-hidden, and results
  (`docs/apple-turboembed-metal-arena-machine-c.md`). See
  [`docs/turbo-buffer.md`](turbo-buffer.md).
- **Intel NPU** create is fail-loud until a Core Ultra client NPU host
  lists the plugin (`intel-npu.json` on Machine B, `pass=false`).
- Inferstream **Rerank RPC** is a thin façade over the TurboRerank
  C ABI when `--features turborerank` is on (catalog
  `ms-marco-minilm-l6`). The TurboRerank
  **library** (Phase 1 CPU MiniLM CE) lives beside this ABI; see
  [`docs/turborerank-architecture.md`](turborerank-architecture.md).
  **TRT-LLM generation**
  is a separate stub (`trtllm-sys`) — not the live ORT TensorRT MiniLM
  embed path (`nvidia-minilm-tensorrt.json`).

## Mock is smoke-only

`TURBOEMBED_DEVICE_MOCK` / `Device::Mock` (and the `mock-embed` / `mock`
aliases) exist for **ABI smoke**: create / list / load / embed / free of
an 8-d FNV vector. That path is never a silent substitute for a missing
Metal/GPU/ORT/GenAI provider.

Catalog aliases (`minilm`, `bge-*`, `e5-*`, …) on `AUTO` / `METAL` /
`CUDA` / `TENSORRT` / `OPENVINO_*` must come from the real engine or
**fail loud**.
A live/integration test that accepts `dim == 8` for those aliases is
wrong — it is asserting the mock.

Build the default no-feature link and the Rust smoke test: see the
[root README](../README.md#turboembed) and `native/turboembed/README.md`.
