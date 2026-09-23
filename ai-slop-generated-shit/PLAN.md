# Turbo: one native inference and embedding library

Plan date: 2026-09-21. This document supersedes `ROADMAP.md` (2026-09-13) as the
governing plan. It is written to be executed in one pass, milestone by
milestone, by an agent or a contributor who has not seen the prior code.
Research inputs (runtime versions, API designs, binding technology) were
verified against primary sources on the plan date and are cited inline.

## 0. The point, the entities, and what is left (2026-09-23)

This section was written after two days of building and is the statement
the rest of the plan is held against. Where a later section disagrees
with it, this section wins.

### The point

One native library with one interface. For each kind of hardware it goes
to the lowest, fastest layer that hardware has, with nothing in between:
Metal directly on Apple, HailoRT on the Pi, OpenVINO on Intel GPUs and
CPUs and, through its NPU plugin, on the Intel NPU (static shapes, so the
session's buckets are the shapes it compiles), CUDA with its own kernels
on NVIDIA, llama.cpp for GGUF. On NVIDIA that means CUDA
itself: the encoder runs on the provider's own kernels over cuBLASLt from
the bundle's weights, resident from upload to result. ONNX Runtime is the
last resort, taken only for a model the direct path does not implement,
and a result that came through it says so. Speed is the reason it
exists, and it is proven by matched benchmarks against the vendor's own
loop, never claimed. A program written against the interface, from Java,
Swift, Rust, C or over gRPC, embeds, ranks, classifies, chunks and
generates the same way on any of those machines. The model's contract
travels in a hash-verified bundle so the answer is the same everywhere.
Serving and bindings are how people reach it; they are not the product.

Three deliverables, in this order: the core library, which is what people
build their own applications on; Inferstream, the gRPC and REST service
over it for deployments that want a server (the Open Inference Protocol,
so it runs under KServe); and the front end for people to try models
and hardware. The second and third are projections of the first and add
no semantics. The core carries no OpenNLP dependency: it takes sentence
boundaries as an input, or computes them with its own rules, and the
OpenNLP service (a separate gRPC server, a separate effort) is composed
in front of the core at the Inferstream layer, where a network hop for a
few kilobytes of text is fine and where both services are exposed
together. The Java layer is JDK 25 with the foreign function API over
the C ABI; under the ABI the core is Rust and the providers are Rust, C++
and Objective-C++, and Java never sees which. ONNX is a format a user
brings, never the endpoint: a provider compiles it (OpenVINO, TensorRT,
the Hailo compiler), takes the weights out of it for its own kernels
(the direct CUDA and Metal paths), or, for an architecture with no
kernels yet, runs the graph through the fallback engine and says so. DJL
is not an integration target; it is the feature list to beat. Its
capabilities (a model zoo with lookup by criteria, per-task pre and post
processing, batching, image and audio tasks, serving) are covered one by
one as layers of our own over the core, in our shape, each a cell in the
matrix, after R0 to R6. If DJL wants an Engine over our Java binding,
that is their work to do against a stable ABI, not ours.

Slow hardware is fine. A path slower than the hardware's own best is not.
A feature that exists on one configuration and not another is fine, and
the compliance matrix says so. Performance over elegance: ONNX Runtime is
built for elegance, one graph format and one session API over every
backend, and pays for it in copies and host round trips. This library is
a better ONNX, the same one interface as close to the metal as each
hardware allows. When a cleaner abstraction and a faster path disagree,
the faster path wins and the abstraction is made to fit it. Existing unifying libraries fail on one of
two sides: they lower every task to generic operations and pay for the
copies between them, or they keep the vendor pipeline and lose the common
interface. This library keeps the vendor pipeline and puts the common
interface above it.

Tasks, not operations. The interface speaks in the units an application
needs, and a provider receives the whole task and runs it on the
hardware's own pipeline, fused the way that vendor's stack fuses it. The
vendor pipelines on CUDA and on Intel take text in and hand vectors out,
with tokenizing and chunking inside them; that is why the tokenizer was
rewritten natively in the core: one implementation that feeds any
provider with no copies, which a provider replaces with its own fused
stage when the hardware does that stage faster. A stage runs where its
data already is: if the tokens are on the GPU and the GPU can gather,
pool or chunk, it does, and the host fallback, with the copy it costs, is
taken only when the provider cannot run that stage. Where a stage runs is
the provider's decision and is reported like any other placement, and so
is the copy.

### The entities

- Hardware (device). What is physically in the machine: kind (GPU, NPU,
  CPU), vendor, architecture label, memory, the vendor runtime and driver
  version. Discovered at runtime, never configured. The list today: the
  RTX 4080 SUPER and the Jetson Orin Nano (NVIDIA GPU), the Arc B70
  (Intel GPU), the Intel NPU (through OpenVINO, pending the vendor's
  approval of the device), the Hailo-8 and Hailo-10H (NPU), the M2 (Apple
  GPU), and the x86 and Arm CPUs of those machines. Each is a column of
  the matrix, and a provider that serves it has the rows.
- Provider. The implementation that drives one hardware family through
  its lowest layer: `cuda`, `openvino`, `metal`, `hailo`, `ggml`, and the
  explicit CPU providers. Loaded as a library. It offers tasks per device;
  that offer, with its status, is the compliance matrix (section 4.4).
- Task. The unit of work: chunk, tokenize, embed, rerank, classify, tag
  tokens, generate; audio and image later. Defined once on the interface
  with its inputs, options and outputs (sections 4 and 5).
- Model. A checkpoint with a contract per task (pooling, normalization,
  dimension, labels, prompts). An application meets it as a bundle.
- Bundle. The hash-verified directory: contract, tokenizer, one or more
  artifacts (section 6).
- Artifact. The model made runnable for one target: a format (weights
  for the cuda and metal providers' own kernels, `openvino_ir`, `hef`,
  `gguf`, and `onnx` for the fallback engine) for an architecture. The
  thing a provider loads. A recipe produced it.
- Stage. One step of a task: normalize text, tokenize, encode, pool,
  normalize vectors, pool segments, group, centroid, score, decode. A
  provider states per stage where it runs (device or host) and whether a
  copy was taken to get there. Boundaries for chunking are a host stage
  everywhere (they are computed over text before anything reaches a
  device; OpenNLP's native image is the provider of that stage, with
  sentence detection and its analysis, and the core's rules are the
  other). Pooling the segments they define, grouping sentence embeddings
  by neighbour similarity or a running centroid, and the centroids
  themselves are vector math over data that is already resident, so they
  are device stages wherever the stack allows it (CUDA, OpenVINO, Metal)
  and host stages on Hailo and, until its public API keeps an output on
  the device, on ggml. Generative chunking (a generator turns a span of
  text into statements of fact, which are then embedded) is the generate
  task followed by one small host hop, detokenize and retokenize, because
  the generator's vocabulary is not the embedder's; the hop is reported
  as the copy it is. Tokenization is a host stage on every stack today;
  the core tokenizer is the reference and a fused provider tokenizer
  replaces it only after an equivalence check.
- Receipt. Proof binding provider, device architecture, runtime, bundle
  and artifact hashes, and task: conformance, precision, matched-native
  (section 11). It fills the matrix and it is the data path selection
  ranks on. The matched benchmark's other side is the fastest known
  program for that hardware, at the commit `docs/reference-code.md` pins.
- Provenance. What one result carries back: provider, device
  architecture, runtime version, artifact hash, tokenizer hash, and the
  placement of every stage with any copy taken. Receipts are per
  configuration and committed; provenance is per result and returned.
- Interface. The common layer: the C ABI and the Rust API over it. Every
  surface (Java FFM, Swift, Android JNI, C and C++, gRPC and REST, the
  web routes) is a thin projection of the same calls and adds no
  semantics of its own.
- Runtime objects. A context on a device; a model loaded on it through a
  provider; a session with its buckets; buffers, host and device, with
  zero-copy import; results resident on the device with an explicit read
  (section 4.5).
- Selection. Two calls the interface does not have yet. Resolve: a task
  plus constraints (dimension, languages, size, licence, or a name) to
  the bundles available on this machine that have an artifact for its
  hardware. Select: a task plus a bundle to the device and provider that
  will run it fastest here, ranked by cell status (SUPPORTED before
  EXPERIMENTAL) and then by the receipts' measured throughput for that
  device class and task, with the choice and its reason reported. Today
  `AUTO` returns the first accelerator in load order and takes no task;
  that is the gap between the README and the code, and the README is
  corrected until the calls exist.
- Catalog. The typed data selection reads: model, artifact, target,
  receipts. Built locally from the bundles on disk and the committed
  receipts; the protobuf schema in P11 is this and nothing more.
- Fallbacks. Explicit, never silent: the CPU providers, and for chunking
  an OpenNLP native-image build (in progress in a separate effort),
  plugged in as a provider of that task.

Overlaps to keep straight: one device can be served by two providers
(an RTX 4080 by `cuda` and by `ggml`), and selection arbitrates; one
bundle can hold artifacts for several targets; the server is an
interface and a deployment at once; a receipt binds five entities, which
is why it is the ranking data and why it must name hashes, not claims.

### What exists and what is left: the roadmap

The common layer exists: the ABI, the core, the conformance suite, the
five providers with receipts for embeddings on five machines, the bundle
contract, the bindings and the server (P0 to P9). What is disjointed is
the spine: tasks as the unit, selection on the interface, placement and
provenance on every result, and one bar for "fast". The roadmap is that
spine, in order. Each item names what ends it; an item without its
receipt is not done. What the pinned reference source says about each
stack, with file and line, is in `docs/reference-code.md`, and the items
below follow from it.

R0. The mission where every reader starts. `AGENTS.md` opens with the
    mission and the reject rules; the README states the mission, one
    example, the matrix and the benchmark table, and nothing else.
    Ends: both files reviewed and merged.

R1. Stages, placement and provenance on the interface. The stage list
    above becomes an ABI enumeration; `stage_placement` names every stage
    and whether a copy was taken; a result carries its provenance. Chunk
    becomes one task with a strategy and a plan of stages, asked for in
    one call so the provider can keep it resident: fixed (the core's
    rules), boundaries (the OpenNLP native image as the provider of that
    stage), semantic (boundaries, embed the sentences, group them by
    neighbour similarity or running centroid, centroids as segment
    means, optionally re-embed the merged chunks, all on the device where
    the stack allows it), and generative (the generate task producing
    statements of fact, the text hop, then embed). The output is the
    spans, and the chunk embeddings when asked, with a placement per
    stage. The embed task takes an optional segment plan. Ends: headers regenerated, the mock
    provider and every real provider report placement for every stage,
    the conformance suite checks provenance, bindings compile.

R2. Selection on the interface. Resolve: task plus constraints or a name
    to the bundles on this machine with an artifact for its hardware.
    Select: task plus bundle to the device and provider ranked by cell
    status then by receipts for that device class and task, with the
    reason reported. The local catalog behind them is built from the
    bundle directories and the committed receipts (P11 cut to that).
    Ends: the calls specified, implemented, conformance cases for both,
    the reason string checked, every binding projecting them.

R3. The bar per machine. Build the fastest known loop from the pinned
    checkouts and measure it with `turbo-bench`'s token dumps: on the RTX
    4080, a TensorRT FP16 engine with pooling in the graph and TEI's
    unpadded FlashBert on candle's kernels (onnxruntime with IO binding
    is measured too, as the bar for the fallback engine only, never as
    the target); on the B70, openvino.genai's pipeline against a
    hand loop with USM tensors; on the M2, MLX (whose fused attention
    does not cover MiniLM's head size, so the composed path); on the Pi,
    `hailortcli run2` in full async mode; for GGUF, `llama-embedding`.
    Ends: a native receipt per loop naming the reference commit, the
    compare verdicts recomputed against the fastest of them, cells
    regraded.

R4. Providers to the bar. First, and not conditional on R3's numbers:
    the cuda provider's direct path. Today its encoder runs through ONNX
    Runtime's CUDA execution provider with the provider's own kernels
    only for pooling, normalization, sigmoid and softmax (P3). The direct
    path runs the encoder on the provider's own kernels: cuBLASLt for the
    projections, a fused attention kernel for head sizes 32 and 64, fused
    add and layernorm, GELU, then the existing pooling and segment
    pooling, all resident, with weights read from the bundle by the
    safetensors reader `tools/turbo-bundle` already has. ONNX Runtime
    stays as the fallback engine for an architecture the direct path does
    not implement, named in the placement and in the cell. Then, wherever
    R3 puts a cell under 0.95: openvino
    fuses segment pooling with the pooling it already has and moves
    inputs and outputs to USM device tensors; metal takes the head size
    32 attention path that MLX lacks; ggml avoids the host readback the
    public API forces where it can and reports it where it cannot; hailo
    measures the host transform against raw async streams. Ends: the
    verdicts at or above 0.95, or the cell EXPERIMENTAL with the number.

R5. The matrix filled where the hardware allows. Rerank, classify and
    token classification on the B70 (the bench workloads and OpenVINO
    reference are parked on `wip/b70-task-benchmarks`); the Jetson cells;
    the Hailo-10H artifact (needs DFC 5.1); the Intel NPU as static-shape
    buckets, batch one, when the approval arrives. Ends: receipts per
    cell.

R6. Bindings as projections. Java, Swift, gRPC and REST expose provenance
    and the two selection calls and add nothing else. Ends: the binding
    conformance cases cover both.

Deferred until R0 to R6 are done, and not on the front of the plan:
remote repositories, the DJL index view, signing, mirrors, GPU-side
tokenization (cudf's WordPiece is the only one and needs its own
normalize and pack steps; a PLANNED cell on cuda, nothing else), and any
tooling not needed by the items above.

## 1. Verdict on the current code and what we keep

The repo is a proof of concept whose layers overclaimed. Two independent
reviews on 2026-09-21 agreed on the shape of the problem:

- The "common interface" exists at the symbol level (`include/turboembed.h`)
  but not at the semantic level. Per-call options mean four different things
  across ORT, OpenVINO, MLX, and Hailo; default pooling for one alias comes
  from three unrelated sources; lifetime rules differ by language.
- The "lowest, fastest layer" holds on NVIDIA only. Intel and Apple pull the
  full `[batch, seq, hidden]` activation to the CPU to pool it.
- The fast path that does exist on Intel lives in a second, vendor-specific
  ABI (`include/turboembed_prepared.h`), and that is the only ABI Java can
  reach.
- Generation is not behind any native contract; it exists only inside the
  Rust gRPC server.

We refactor by extraction, not by restart. The contract is new. The
internals that were measured to work are moved behind it from the
`poc-2026-09-21` tag.

| Keep (move behind the new contract) | Why |
|---|---|
| `native/wordpiece` | Gated loader (exact WordPiece + BertNormalizer match), HF-parity normalizer, write-through encode into caller rows |
| `native/turbo_buffer` arenas (CPU, CUDA pinned/device, Level Zero USM, Metal shared) | Working placements with allocation counters |
| `native/turboembed/src/pool_cuda.cu` and the ORT CUDA IoBinding path in `crates/turboembed/src/ort_cuda.rs` | Device-side mean+L2, `d2h_hidden_bytes == 0`, refcounted external allocator |
| `native/turboembed/sdk/prepared.cpp` | Fused mean+L2 OpenVINO graph, bundle verifier, OpenCL result lease, context/model/slot/result handle model |
| `native/turboembed/src/hailo.cpp` | Host gather + NPU encoder body + host pooling split, proven on Hailo-8 |
| `swift/Sources/TurboEmbed`, `MlxEngine`, `MetalArena` | MLX provider and Metal shared arena (pooling must move onto MLX ops) |
| `crates/turboembed/src/chunker.rs` | Byte-offset chunk planner |
| `crates/fetch`, `models/manifests/*.json` | Hash-pinned provisioning |
| `proto/`, `crates/protocol`, `crates/server` | Wire contract and server plumbing, rebuilt on the new ABI in P9 |
| `crates/backend-llamacpp` | Becomes the ggml generation provider |
| Test fixtures, goldens, receipts under `testdata/` | Reference vectors and hardware history |

| Discard | Why |
|---|---|
| `include/turboembed.h` and `include/turboembed_prepared.h` as public surfaces | Replaced by one versioned header set; a compat shim is optional and last |
| `native/turboembed/src/stub.cpp` dispatcher | `#ifdef` dispatch per feature is the source of the drift |
| `TURBOEMBED_WORKSPACE_ROOT` baked into binaries | Models come from explicit bundle paths |
| Thread-local last-error, three pooling defaults, per-provider option branches | Replaced by caller-owned errors and a single bundle contract |
| `crates/backend-apple`, `crates/arch-apple`, `native/mlx-engine`, `crates/backend-trtllm` | Stubs and legacy |
| Cargo feature flags as the way to pick a provider | Providers become runtime-loaded plugins; one core binary |

## 2. Principles (non-negotiable)

1. One contract. Every provider implements the same header set. Vendor
   differences are expressed through capability reporting, never through a
   second API.
2. Honest capabilities. Every option and every placement has a capability
   bit. A provider honors it exactly or reports the bit clear and the call
   fails with `TURBO_E_UNSUPPORTED` naming the field. Nothing is silently
   ignored, capped, or substituted.
3. Lowest layer per device. Tokenization writes into the buffers the model
   reads. Pooling and normalization run on the device that produced the
   hidden state. Results stay resident until the caller asks for a host copy.
   Where a device cannot do this (Hailo-8 has no pooling on the NPU), the
   provider reports `fully_accelerated = 0` and says which stages run on the
   host.
4. Device policy. `AUTO` means the host's best accelerator. CPU runs only when
   selected. An absent device is an error, never a fallback. The mock
   provider serves only the mock model.
5. Ownership is executable. Child handles retain parents. Releasing a parent
   never invalidates a child. Bindings enforce this with types, not comments.
6. Per-model truth lives in the bundle. Pooling, normalization, sequence
   limit, prefixes, dimension, dtype, and tokenizer identity are derived once
   at import from the model's own files and frozen with hashes.
7. Measured, not claimed. Every capability cell a provider marks
   `SUPPORTED` ships a conformance receipt, a precision receipt (variance
   against the reference), and a matched-native benchmark from a named
   machine. `EXPERIMENTAL` and `PLANNED` are honest states, not failures.
8. No one-offs. A consumer need (OpenNLP token classification, KServe) is met
   by the common surface. If the surface cannot express it, the surface is
   extended for everyone.

## 3. Product definition

- Library name: `turbo`. C prefix `turbo_`, shared library `libturbo` plus
  one `libturbo_provider_<name>` per provider. TurboEmbed stays the project
  and repository name. Inferstream stays the server.
- Deliverables, in order of delivery: C ABI + Rust crate; providers for
  OpenVINO, CUDA (x86_64 and Jetson aarch64), Metal, Hailo-8/8L, Hailo-10H,
  and ggml generation; Java (JDK 25 FFM) and Swift packages; SDK archives per
  platform; Inferstream server on the new ABI; Android JNI; GraalVM
  native-image validation for OpenNLP.
- Tasks: `EMBED` (dense; sparse and multi-vector reserved), `RERANK`
  (cross-encoder), `CLASSIFY` (sequence classification with labels),
  `TOKEN_CLASSIFY` (per-token labels with span aggregation, the OpenNLP
  NER/POS shape), `GENERATE` (causal LM, streaming), `TOKENIZE` and
  `DETOKENIZE`, `RUN` (generic named-tensor execution for any ONNX/IR/GGUF
  model, the KServe `ModelInfer` shape), and `CHUNK` (host utility).
- Scope of v1 is text. The modality axis (`TEXT, AUDIO, IMAGE, VIDEO`) is
  in the ABI from P0 so audio and video providers (Whisper on Hailo-10H,
  audio or image encoders on CUDA and OpenVINO) are added as capability
  cells right after v1 without an ABI change. `RUN` accepts image or audio
  tensors a caller has already prepared; decoders and resizers come with
  those modalities.
- Consumers this plan must not break: an OpenNLP provider on GraalVM
  native-image, and a KServe-compatible server. Both are reached through the
  same header.

## 4. Architecture

### 4.1 Layers

```
 bindings:   Rust crate | Java (FFM, later JNI) | Swift package | C/C++
 ------------------------------------------------------------------------
 libturbo:   runtime + provider registry | device discovery | buffers
             bundles + tokenizers + chunker | sessions | results | errors
 ------------------------------------------------------------------------
 providers:  cpu | cuda | openvino | metal | hailo | ggml   (dlopen plugins)
 ------------------------------------------------------------------------
 runtimes:   ORT 1.30 | OpenVINO 2026.4 | MLX 0.32 | HailoRT 4.24 / 5.4 | llama.cpp 0.4
```

The core is a Rust `cdylib` that also compiles the shared C++ (`wordpiece`,
`turbo_buffer`, pooling kernels). Providers are separate shared libraries
that export one symbol, `turbo_provider_get`, returning a versioned vtable.
The core loads providers from a search path and from explicit
`turbo_runtime_load_provider(path)` calls. A binary without a provider
library for a device reports that device absent. This replaces Cargo
features as the way to choose hardware and follows the ggml
`ggml_backend_load_all` and Windows ML `RegisterExecutionProviderLibrary`
distribution shape (ggml-backend.h; ORT 1.30 plugin EP docs).

### 4.2 Object model

```
turbo_runtime      library instance; owns the provider registry; thread-safe
  turbo_device     enumerated hardware; immutable info + capabilities
    turbo_context  device + memory domain; owns allocators and queues
      turbo_buffer typed memory in a placement; import/export native handles
      turbo_model  loaded bundle on a context; immutable contract + placement
        turbo_session  execution workspace; fixed max shape; one in-flight op
          turbo_result   leased output of the last op; explicit host read
        turbo_generation streaming generation state; pull-style iterator
  turbo_tokenizer  from a bundle; encode into caller buffers, decode, chat template
```

Every handle is opaque and reference counted internally. `turbo_*_release`
on a parent does not invalidate a child. A session accepts one operation at
a time and returns `TURBO_E_BUSY` otherwise. Distinct sessions on one model
run concurrently. Weight sharing across sessions is a provider capability.

### 4.3 Error and versioning model

- One `int32_t` status space, graded so callers can branch without parsing
  text: `0` ok; `0x1xx` argument/contract; `0x2xx` unsupported on this
  device/model/option; `0x3xx` resource (`OUT_OF_MEMORY`, `BUSY`,
  `OVERLOADED`); `0x4xx` device/runtime failure; `0x5xx` bundle integrity;
  `0x6xx` internal. Modeled on ExecuTorch's graded `Error` and TEI's
  424/429/422 split.
- Caller-owned error struct, no allocation on the error path, no shared
  state: `turbo_error { uint32_t struct_size; int32_t code; uint32_t field;
  char message[496]; }`. `field` names the offending option for `0x2xx`.
- Every public struct begins with `uint32_t struct_size`. The library reads
  only fields below the size the caller declared, and accepts a size only
  when it is the end of a field the struct has ever had (a per-struct table,
  `crates/turbo-abi/src/versioned.rs`, generated from the field lists by
  `scripts/gen-versioned.py` and checked by `scripts/gen-versioned.py
  --check`); any other size, including one that ends inside a field, is
  `TURBO_E_INVALID_STRUCT_SIZE`. Appended fields must have a
  zero value that means "old behavior". Enumerations in ABI position are
  `uint32_t` with named constants, never C `enum`. An optional `const void
  *next` chain is reserved for rare vendor imports (CUDA stream, `cl_mem`,
  `MTLBuffer`, Level Zero handle, DMA-BUF fd). This is the Linux-kernel
  extensible-struct rule; `pNext` is not used on per-call structs.
- `TURBO_ABI_VERSION` is a single integer. The header set is generated by
  cbindgen 0.29 from `#[repr(C)]` Rust declarations, committed, and checked
  by a parity test; the committed header is the artifact consumers see.

### 4.4 Capability model

Capabilities are a matrix, not a bitset alone. The axes are task
(`EMBED, RERANK, CLASSIFY, TOKEN_CLASSIFY, GENERATE, TOKENIZE, RUN`),
modality (`TEXT, AUDIO, IMAGE, VIDEO`; v1 fills in `TEXT` only, the others
are declared so audio and video are new cells later, not a new ABI), and
device. Each cell is a `turbo_capability { struct_size; status; dtype;
precision; determinism; notes[] }` where `status` is `SUPPORTED |
EXPERIMENTAL | PLANNED | UNSUPPORTED` and `precision` is the measured
variance against the reference from the qualification receipt (cosine
floor, max absolute error, reference dtype). A provider that offers one
cell (a static-embedding provider offering `EMBED x TEXT x CPU`) is a
complete, valid provider. `turbo_device_capability(device, task, modality,
&cell)` reads a cell; the same matrix is what `turbo_model_info` reports
after load for the loaded model.

Two further levels, both required:

1. Static: `turbo_device_info` carries `kind` (`CPU`, `GPU`, `IGPU`, `NPU`,
   `ACCEL`), vendor, name, provider id, ordinal, memory totals, runtime and
   driver versions, and a `uint64_t caps` bitset: `ASYNC`, `HOST_PTR_IMPORT`,
   `DEVICE_RESULT`, `EXTERNAL_QUEUE`, `DMABUF`, `UNIFIED_MEMORY`,
   `DYNAMIC_SHAPE`, `WEIGHT_SHARING`, plus one `OPT_*` bit per option field
   (`OPT_TRUNCATE`, `OPT_MAX_TOKENS`, `OPT_PROMPT_ROLE`, `OPT_NORMALIZE`,
   `OPT_POOLING_OVERRIDE`, `OPT_OUTPUT_DIM`, `OPT_OUTPUT_DTYPE`, and for
   generation `OPT_GEN_STRUCTURED`, `OPT_GEN_TOOLS`, `OPT_GEN_N`,
   `OPT_GEN_LOGIT_BIAS`, `OPT_GEN_PENALTIES`), plus `DEVICE_TOKENIZE` and
   `DEVICE_POSTPROCESS`. Modeled on
   `ggml_backend_dev_props.caps` and LiteRT's accelerator bitmask.
2. Per request: `turbo_can_run(device, bundle_or_model, task, options,
   &reason)` answers "this model, these options, on this device" before any
   allocation. Modeled on `ggml_backend_dev_supports_op`.

After load, `turbo_model_info` reports what actually happened:
`placement` (which stages run on device vs host), `fully_accelerated`,
`dtype_used`, `max_batch`, `max_seq`, and the frozen contract (below).

### 4.5 Memory model

`turbo_buffer_desc { struct_size; context; placement (HOST, PINNED, DEVICE,
SHARED); dtype; ndim; shape[8]; strides[8]; bytes; next }`. The dtype set
is the KServe Open Inference Protocol set plus BF16: `BOOL, U8, U16, U32,
U64, I8, I16, I32, I64, F16, BF16, F32, F64, BYTES` (BYTES is a
length-prefixed string tensor for text-input graphs). A provider reports
which dtypes it can hold in each placement.

- `turbo_buffer_alloc` from a context; `turbo_buffer_import` wraps caller
  memory (host pointer, CUDA device pointer, `cl_mem`, USM, `MTLBuffer`,
  Level Zero handle, DMA-BUF fd, via `next`); `turbo_buffer_export` returns
  the native handle for a downstream GPU stage; `turbo_buffer_map/unmap` for
  host access where the placement allows it.
- Sessions preallocate inputs, workspace, and outputs at creation for the
  declared max shape. `turbo_session_write_tokens` copies caller token rows
  into the bound input buffers; `turbo_session_write_text` tokenizes directly
  into them. `turbo_session_run` never allocates after warmup; the
  conformance suite asserts `allocs_per_run == 0` on the provider's own
  counter and reports provider-internal allocations separately.
- Results lease the session's output buffer. `turbo_result_read` is an
  explicit blocking copy to host memory. `turbo_result_buffer` returns the
  device buffer for zero-copy consumption. Releasing the result returns the
  lease; the session then accepts the next operation.
- Synchronous in v1. `CAP_ASYNC` providers additionally expose
  `turbo_session_submit` + `turbo_fence_wait`; external queue import is
  `CAP_EXTERNAL_QUEUE`.

### 4.6 Threading model

Runtime, device, and model handles are immutable after creation and may be
used from any thread. Context is thread-safe for allocation. Session is
single-owner: the caller serializes; a concurrent call returns `BUSY`
without corrupting state. Callbacks are invoked on the calling thread, never
from provider worker threads, and must not call back into the same session.
Bindings encode this (Rust sessions are `Send + Sync` handles whose calls are single-owner at run time; Java
`Arena.ofConfined` per call with `synchronized` handle methods; Swift
`final class` with an internal lock).

### 4.7 Provider contract: task-level, fused pipelines preferred

The provider vtable is defined at task granularity, not graph granularity.
A provider implements `embed`, `rerank`, `classify`, `token_classify`,
`generate`, and `run` against the model and session handles, and is free to
execute each as one fused device pipeline. The core's own sequence
(tokenize on host into arena rows, run the graph, pool, normalize) is the
fallback path a provider uses only for stages it declines to implement,
and a provider says which stages it took by reporting per-stage placement
in `turbo_model_info` (`stage_placement[TOKENIZE|ENCODE|POOL|NORMALIZE|
POSTPROCESS]` = `DEVICE | HOST | FUSED`). This is the gRPC rule: the header
is the contract; how a provider honors it is the provider's business, and a
faster, less uniform implementation wins over a uniform, slower one.

Consequences:

- `turbo_session_write_text` may upload raw UTF-8 into a device buffer when
  the provider reports `CAP_DEVICE_TOKENIZE`; otherwise the core tokenizes
  into the bound input rows. GPU tokenization is a measured experiment in
  the CUDA provider (P3 stretch), not a v1 requirement; the win is CPU
  offload at high throughput, not latency at batch one.
- OpenVINO compiles tokenizer (CPU device), encoder, pooling, and
  normalization as one unit; the tokenizer stage is reported `HOST`
  because string ops have no GPU kernels.
- Post-processing for `CLASSIFY` and `TOKEN_CLASSIFY` (softmax, argmax,
  span aggregation with the bundle's label set) runs on the device where the
  provider can express it and on the host otherwise, and is reported.
- The core never branches on hardware. Vendor differences live only inside
  provider libraries; the `#ifdef` dispatch of the PoC does not return.

## 5. The C ABI (representative declarations)

The full header is produced in P0. These are the shapes that decide the
design; names are final unless P0 finds a conflict.

```c
/* turbo/turbo.h  (generated; TURBO_ABI_VERSION 2) */
typedef struct turbo_error { uint32_t struct_size; int32_t code; uint32_t field; char message[496]; } turbo_error;

int32_t turbo_runtime_create(const turbo_runtime_desc *, turbo_runtime **, turbo_error *);
int32_t turbo_runtime_load_provider(turbo_runtime *, const char *path, turbo_error *);
int32_t turbo_runtime_device_count(turbo_runtime *, uint32_t *);
int32_t turbo_runtime_device_info(turbo_runtime *, uint32_t index, turbo_device_info *, turbo_error *);
int32_t turbo_runtime_select_device(turbo_runtime *, const turbo_device_selector *, uint32_t *index, turbo_error *);
/* selector: { struct_size; policy AUTO|EXPLICIT; kind mask; vendor; provider_id; ordinal } */

int32_t turbo_context_create(turbo_runtime *, uint32_t device_index, const turbo_context_desc *, turbo_context **, turbo_error *);
int32_t turbo_buffer_alloc(turbo_context *, const turbo_buffer_desc *, turbo_buffer **, turbo_error *);
int32_t turbo_buffer_import(turbo_context *, const turbo_buffer_desc *, turbo_buffer **, turbo_error *);
int32_t turbo_buffer_export(turbo_buffer *, uint32_t handle_kind, turbo_native_handle *, turbo_error *);

int32_t turbo_can_run(turbo_runtime *, uint32_t device_index, const char *bundle_path, uint64_t len, uint32_t task, const void *options, turbo_error *);
int32_t turbo_model_load(turbo_context *, const char *bundle_path, uint64_t len, const turbo_model_desc *, turbo_model **, turbo_error *);
int32_t turbo_model_info(turbo_model *, turbo_model_info *, turbo_error *);
/* model_info: { struct_size; task; kind EMBEDDING|RERANKER|GENERATIVE|GENERIC; dim; n_labels; pooling; normalize;
     max_seq; max_batch; dtype_used; stage_placement[5]; fully_accelerated; prefix_query[]; prefix_document[];
     n_labels; labels (via turbo_model_label(i));
     model_id[]; revision[]; tokenizer_sha256[]; n_inputs; n_outputs; provider_id[] } */
int32_t turbo_model_io_info(turbo_model *, uint32_t index, uint32_t direction, turbo_tensor_info *, turbo_error *);

int32_t turbo_session_create(turbo_model *, const turbo_session_desc *, turbo_session **, turbo_error *);
/* session_desc: { struct_size; max_batch; max_seq; n_options; const turbo_kv *options; next }  -- options are
   provider knobs as key/value strings (tensorrt=1, cuda_graph=1, npu_tiles=2), validated, unknown keys rejected */
int32_t turbo_session_write_text(turbo_session *, const turbo_text *texts, uint32_t count, const turbo_embed_options *, turbo_error *);
int32_t turbo_session_write_tokens(turbo_session *, const turbo_token_batch *, turbo_error *);
int32_t turbo_session_bind(turbo_session *, const char *io_name, turbo_buffer *, turbo_error *);   /* RUN task */
int32_t turbo_session_run(turbo_session *, const turbo_run_options *, turbo_result **, turbo_error *);
/* run_options: { struct_size; n_params; const turbo_kv *params }  -- the KServe per-request parameters map */
/* classification: same session; result carries scores[n][n_labels] (CLASSIFY) or per-token labels + aggregated spans
   {start_byte, end_byte, label, score} (TOKEN_CLASSIFY); labels come from the bundle contract */
int32_t turbo_session_write_text_classify(turbo_session *, const turbo_text *, uint32_t n, const turbo_classify_options *, turbo_error *);
int32_t turbo_session_stats(turbo_session *, turbo_session_stats *, turbo_error *);

int32_t turbo_result_info(turbo_result *, turbo_result_info *, turbo_error *);
int32_t turbo_result_buffer(turbo_result *, uint32_t index, turbo_buffer **, turbo_error *);
int32_t turbo_result_read(turbo_result *, uint32_t index, void *dst, uint64_t capacity_bytes, turbo_error *);
void    turbo_result_release(turbo_result *);

/* embed options (per call). Each field maps to a capability bit; MODEL means "use the bundle contract". */
typedef struct turbo_embed_options {
    uint32_t struct_size;
    uint32_t truncate;     /* MODEL | NONE | RIGHT | LEFT */
    uint32_t max_tokens;   /* 0 = MODEL */
    uint32_t prompt_role;  /* NONE | QUERY | DOCUMENT */
    uint32_t normalize;    /* MODEL | NONE | L2 */
    uint32_t pooling;      /* MODEL | MEAN | CLS | LAST  (requires OPT_POOLING_OVERRIDE) */
    uint32_t output_dim;   /* 0 = MODEL; Matryoshka truncation requires OPT_OUTPUT_DIM */
    uint32_t output_dtype; /* F32 | F16 | I8 (requires OPT_OUTPUT_DTYPE) */
} turbo_embed_options;

/* rerank: query + documents through the same session; result is scores[n] in input order plus optional sorted index */
int32_t turbo_session_write_pairs(turbo_session *, const turbo_text *query, const turbo_text *docs, uint32_t n, const turbo_rerank_options *, turbo_error *);

/* generation: pull-style iterator is primary; works without upcall stubs in every binding; cancel is a call, not a return value.
   A push form, turbo_generate(model, desc, messages, n, callback, user_data, error), is declared from P0 and returns
   TURBO_E_NOT_IMPLEMENTED until P6 lands it as a loop over turbo_generation_step; the callback returns CONTINUE | STOP. */
int32_t turbo_generation_create(turbo_model *, const turbo_generate_desc *, turbo_generation **, turbo_error *);
int32_t turbo_generation_prompt(turbo_generation *, const turbo_message *messages, uint32_t n, turbo_error *); /* applies chat template */
int32_t turbo_generation_prompt_tokens(turbo_generation *, const int32_t *ids, uint32_t n, turbo_error *);
int32_t turbo_generation_step(turbo_generation *, turbo_generation_chunk *, turbo_error *);  /* token ids, text piece, logprobs, done, finish_reason */
int32_t turbo_generation_cancel(turbo_generation *);
void    turbo_generation_release(turbo_generation *);
/* generate_desc: { struct_size; max_new_tokens; min_new_tokens; n_sequences; temperature; top_k; top_p; min_p;
   repeat_penalty; presence_penalty; frequency_penalty; seed; n_stop; stop[]; n_logit_bias; logit_bias[]; logprobs; echo;
   structured: { kind NONE | JSON_SCHEMA | GRAMMAR; text }; n_tools; tools[] (JSON); n_options; options[] }
   Each of structured output, tools, n_sequences > 1, logit_bias, and penalties is capability-gated (OPT_GEN_*). */

int32_t turbo_tokenizer_create(turbo_runtime *, const char *bundle_path, uint64_t len, turbo_tokenizer **, turbo_error *);
int32_t turbo_tokenizer_encode(turbo_tokenizer *, const turbo_text *, uint32_t n, const turbo_encode_options *, turbo_token_batch *out, turbo_error *);
int32_t turbo_tokenizer_decode(turbo_tokenizer *, const int32_t *ids, uint32_t n, char *dst, uint64_t cap, uint64_t *written, turbo_error *);
int32_t turbo_tokenizer_count(turbo_tokenizer *, const turbo_text *, uint32_t *n_tokens, turbo_error *);

int32_t turbo_chunk_plan(const turbo_chunk_desc *, const turbo_text *, turbo_tokenizer *, turbo_chunk_plan **, turbo_error *);
```

Convenience wrappers (`turbo_embed(context, bundle, texts, n, opts, float *out)`)
are provided in the header as thin sequences of the calls above and are
measured against them so their overhead is known.

## 6. Bundles and the per-model contract

A bundle is a directory with `bundle.json` (version 2) written last:

- identity: `model_id`, `revision`, `license`, `task`, `kind`, `family`
  (bert, xlm-roberta, mpnet, qwen3, llama, ...); the source checkpoint's
  hash is recorded by the zoo catalog (P11), not here.
- tokenizer: files with hashes, `tokenizer_kind` (wordpiece, bpe, unigram,
  sentencepiece, gguf-vocab), `chat_template` for generative models.
- contract: for classifiers `labels[]`, `id2label`, `activation` (softmax,
  sigmoid, none), and for token classifiers `aggregation` (none, simple,
  first, max) with the tagging scheme (BIO, BILOU); for embedders
  `pooling` (mean, cls, last, mean_sqrt_len, weighted_mean),
  `normalize` (none, l2), `max_seq`, `dim`, `truncate_dim` list for
  Matryoshka, `prompts.query`, `prompts.document`, `similarity_fn`, `dtype`.
- artifacts, one entry per format, each with hash and provenance: `onnx`,
  `openvino_ir`, `gguf`, `hef` (+ `embedding_tables.bin`, arch tag
  `hailo8|hailo8l|hailo10h`), `mlx_safetensors`, `tensorrt_plan` (SM- and
  version-locked, optional cache only).
- limits: `max_batch`, fixed-shape flag for NPUs.

`turbo-bundle import` derives the contract from the model's own files:
`modules.json`, `1_Pooling/config.json` in both the sentence-transformers v6
schema (`pooling_mode`, `embedding_dimension`) and the legacy six-boolean
schema with its documented precedence and `mean` as the no-flag default,
`config_sentence_transformers.json` prompts, `sentence_bert_config.json`
`max_seq_length` when present, GGUF metadata for generative models. The
importer refuses ambiguous inputs rather than guessing. Providers read the
contract; they never infer pooling from an alias.

The existing fetch manifests remain the hash-pinned source list. A catalog
(alias to bundle path) exists only in the server layer until P11, which
adds the model zoo: a published catalog of bundles per model and target,
with the recipe that produced each artifact and the receipts that prove
it (section 10, P11).

## 7. Providers and the lowest layer on each device

| provider | hardware / machine | runtime (version, license) | lowest layer used | salvaged from | known limits to report |
|---|---|---|---|---|---|
| `static` | any CPU | none (pure Rust/C++; model2vec-style static token embeddings, Apache-2 models) | table lookup + mean + L2 on host; one capability cell `EMBED x TEXT x CPU` | new | no context, no attention; documented quality gap vs transformer embedders; first real conformance target after mock |
| `cpu` | any; explicit only | ORT 1.30 CPU EP; ggml CPU for GGUF (MIT) | host arena, write-through tokens | `ort_cuda.rs` CPU path, `backend-llamacpp` | none; never AUTO |
| `cuda` | an x86_64 host with an RTX 4080 SUPER; an NVIDIA Jetson Orin Nano Super 8 GB (aarch64) | ORT 1.30 CUDA EP (CUDA 13 build for x86; JetPack 7.2.1 with CUDA 13.2 / TensorRT 10.16 on Jetson, using the `sbsa/cu130` ORT wheel if it carries sm_87 kernels, otherwise a pinned ORT source build on the board); TensorRT EP via session option; llama.cpp CUDA arch 87/89 | IoBinding on pinned/device arena, `user_compute_stream` import, `gpu_external_alloc` pool, device mean+L2 kernel, ORT 1.30 `CreateSyncStreamForEpDevice` for external queues | `ort_cuda.rs`, `ort_allocator.rs`, `pool_cuda.cu`, `turbo_buffer/cuda.cpp` | TensorRT plan cache is SM-locked; Jetson wheels unverified for sm_87 |
| `openvino` | an x86_64 host with an Intel Arc B70 (Battlemage); Intel NPU when the Core Ultra host arrives | OpenVINO 2026.4.0 (Apache-2) | `ov::Core` compiled model with mean+L2 fused into the graph; `ClContext` USM/`cl_mem` remote tensors on GPU; `ZeroContext` remote tensors on NPU; explicit `"CPU"` | `prepared.cpp` (graph fusion, OpenCL lease), `turbo_buffer/ze.cpp`, `wordpiece` | NPU is static-shape only (fixed `max_seq`, batch from bundle); OpenVINO GenAI RAG pipelines are not used on the hot path because they allocate their own outputs |
| `metal` | Apple M2 | Metal directly (Objective-C++; MSL kernels compiled at load; no MLX, no Xcode) | shared `MTLBuffer`s for tokens, weights, scratch and results; encoder, pooling, L2 and the reranker head as kernels; results resident in unified memory as `TURBO_PLACE_SHARED` | `providers/metal` (landed 2026-09-22; the PoC kernels from `native/turborerank`) | no CPU device (the CPU is reached through `ggml` or OpenVINO); WordPiece on the host; F32 weights only |
| `hailo` | two Pis with Hailo-8 (HailoRT 4.24.0, `hailo8` branch); one Pi with Hailo-10H (HailoRT 5.1.1 from the Pi OS packages today; 5.4 is the planned upgrade); also x86_64 hosts with a PCIe Hailo-8 card | HailoRT (MIT); DFC is proprietary and used offline only | `VDevice` + `InferModel` + `ConfiguredInferModel::Bindings` with `dma_map` on page-aligned arena rows; async `run_async` behind `CAP_ASYNC` | `hailo.cpp` split pipeline | encoder body only on NPU: host gather and host pooling reported as `fully_accelerated = 0`; batch 1, fixed seq 128; HEF locked to chip and HailoRT line; Hailo-10H embedding HEF needs a DFC 5 compile (open) |
| `ggml` | every machine | llama.cpp v0.4.1 / ggml 0.24 (MIT) with CUDA, SYCL, Metal, CPU backends | `ggml_backend_dev` registry for device identity; `llama_batch` decode; embeddings copied once from `llama_get_embeddings_seq` into the result buffer (no caller-owned output in llama.h); KV cache owned by the generation handle | `backend-llamacpp` | generation is the primary use; GGUF embeddings are a secondary path with `pooling_type` from the bundle |
| `hailo` GenAI | Hailo-10H Pi | `hailort::genai::LLM` in HailoRT 5.4 | native LLM on the NPU with its own sampler | new | model set limited to the Hailo GenAI zoo (Qwen2.5/3 1.5B, Llama 3.2 1B); ~8 to 10 tok/s |

Hailo-8 is not ARM-only: HailoRT ships x86_64 packages and the PCIe/M.2
card is supported on Ubuntu x86_64. The provider builds for both.

## 8. Bindings

- **Rust** (`crates/turbo`): the safe API. Types enforce lifetimes
  (`Result` borrows `Session`, `Session` borrows `Model`, all `Arc`-retained),
  single-owner sessions enforced at run time with `BUSY`, callback reentry rejected, panics
  caught at the boundary. The C ABI is exported from `crates/turbo-abi`.
- **C/C++**: the header set plus a CMake package, `-fvisibility=hidden`,
  a version script exporting only `turbo_*`, SONAME `libturbo.so.2`,
  `abidiff` gate in CI.
- **Java desktop** (`bindings/java/turbo-api`, `turbo-ffm`): JDK 25 LTS
  floor; jextract `25-jextract+2-4` used at build time only, output vendored.
  `Arena.ofConfined` per call, `Arena.ofShared` for model handles used from
  pools, `MemorySegment.reinterpret` with cleanup for native-owned memory,
  upcalls only to static methods, `Enable-Native-Access` in the manifest, CI
  runs under `--illegal-native-access=deny`. `turbo-api` is the boundary the
  Android JNI adapter implements later. Java package `ai.pipestream.turbo`;
  Maven group `ai.pipestream`, artifacts `turbo-api`, `turbo-ffm`, later
  `turbo-android`, natives per platform classifier plus `turbo-native-auto`.
  The old `ai.pipestream.turboembed` package is not carried forward.
- **Swift** (`bindings/swift`): SwiftPM package with a C module map over the
  header; `final class` wrappers with `deinit` release; callbacks as
  file-level `@convention(c)` functions with `Unmanaged` user data; shipped
  as an xcframework `binaryTarget` with headers nested under
  `Headers/Turbo/`. The Metal provider is a Swift dynamic library exporting
  `turbo_provider_get` via `@c` (SE-0495, Swift 6.3).
- **Android** (P10): JNI shim over the same header, NDK r30, `arm64-v8a` and
  `x86_64`, 16 KB page alignment, minSdk 24, targetSdk 36. FFM does not exist
  on Android. GPU on device is a later capability (LiteRT-Next or Vulkan).
- **GraalVM native-image** (P10): GraalVM 25.2 supports FFM downcalls and
  upcalls on linux-x64, linux-aarch64, and macos-aarch64. Foreign calls are
  registered through `reachability-metadata.json`; handles are not created in
  build-time-initialized statics. This is the OpenNLP path.
- UniFFI is not used: its per-call `RustBuffer` copies, JNA-based Kotlin, and
  lack of a streaming primitive do not fit a hot path, and it cannot host a
  Swift-implemented provider.

## 9. Repository layout after the refactor

```
include/turbo/           generated, committed headers (turbo.h, turbo_provider.h)
crates/turbo-abi/        #[repr(C)] types, cbindgen config, extern "C" exports
crates/turbo-core/       registry, device discovery, buffers, bundles, tokenizers, chunker, sessions
crates/turbo/            safe Rust API
crates/turbo-conformance/ provider-agnostic contract suite (runs against any provider)
crates/turbo-bench/      matched-native benchmark harness and receipt writer
providers/cpu/  providers/cuda/  providers/openvino/  providers/hailo/  providers/ggml/
providers/metal/         Objective-C++ provider over Metal directly (make + clang++)
native/                  shared C++: wordpiece, turbo_buffer, pooling kernels
bindings/java/  bindings/swift/  bindings/android/
tools/turbo-bundle/      import, verify, fetch; `zoo` subcommands (P11) behind a `net` feature
proto/zoo.proto          the zoo catalog schema (P11); Java and Swift types ship as `turbo-zoo`
server/                  Inferstream: OIP v2 over gRPC and REST, OpenAI-shaped routes (landed 2026-09-22)
docs/  testdata/  scripts/
```

The proof-of-concept tree is tagged `poc-2026-09-21` at the start of P0 and
then removed from the working tree. Salvage pulls files from that tag, so
each move is reviewable against a fixed reference and the new tree starts
clean. Receipts, fixtures, manifests, and `proto/` stay in place.

## 10. Milestones and acceptance gates

Each milestone is landed when its scoped changes are merged, the listed
gates pass on the named machine, and a dated receipt is committed under
`testdata/receipts/`. Local runs, hosted CI, and device runs are recorded
separately.

### P0 Contract, mock provider, conformance suite, Rust API
Deliver `include/turbo/*.h` generated from `turbo-abi`, the `mock` provider
(deterministic 8-d vectors, only for the `mock` bundle), `turbo-core` with
runtime, device enumeration, contexts, host buffers, sessions, results,
errors, and the safe Rust crate. Deliver the conformance suite with these
groups: contract (empty text, embedded NUL, invalid UTF-8 rejected, unknown
`struct_size`, unknown option constants, oversized shapes), lifetime (release
order in every permutation, results outliving sessions and models, two
contexts), capability honesty (every `OPT_*` bit either passes its honor test
or its rejection test), device policy (AUTO never CPU, explicit CPU, absent
device fails), threading (two sessions concurrently, BUSY on overlap,
reentry rejected), allocation (`allocs_per_run == 0` after warmup where the
provider claims it).
Gate: suite passes on mock; header parity test; `cargo test --locked
--workspace` and clippy `-D warnings`; Linux CI. Machine: any.

### P1 Core services
Buffers for all placements (salvage `turbo_buffer`), bundle format v2 with
importer and verifier, the `static` embedding provider as the first non-mock
provider (one capability cell, full conformance, reference goldens from the
model2vec reference implementation), tokenizers (native WordPiece with the gated loader;
HF `tokenizers` Rust crate for BPE/Unigram; GGUF vocab via llama.cpp), chunk
planner, `turbo_can_run`, provider plugin loading with vtable versioning.
Gate: `static` provider passes the full suite and ships a precision
receipt; token-ID parity against HF `tokenizers` for MiniLM, BGE, E5, XLM-R
across ASCII, CJK, accents, emoji, long words; bundle importer round-trips
both sentence-transformers schemas; conformance still green on mock.
Machine: any.

### P2 OpenVINO provider (first GPU baseline)
Embed, rerank, classify, and token-classify on GPU with the fused graph and
remote tensors; explicit CPU; generic `RUN` task for named-tensor models
including BYTES inputs; device-resident results with OpenCL export. NPU path compiled behind the same code with static
shapes, validated when hardware arrives.
Gate: conformance green on GPU and CPU; MiniLM and BGE goldens within
tolerance; `d2h_hidden_bytes == 0`; matched-native overhead within the
budget set by the first benchmark run (recorded, then held); two-context
isolation; receipt from the Intel Arc B70 host. This is the baseline all
other providers
are compared against for correctness.

### P3 CUDA provider
x86_64 first (the RTX 4080 SUPER host), then the Jetson Orin Nano on
JetPack 7.2.1. The first
Jetson step is a runtime probe of the `sbsa/cu130` ORT wheel for sm_87
kernels; on `cudaErrorNoKernelImageForDevice` the pinned source build
(ORT 1.30, CUDA 13.2, TensorRT 10.16, sm_87) is used instead. IoBinding, stream
import, external allocator pool, device pooling kernel, TensorRT EP as a
session option, generic `RUN`, classify and token-classify with device
post-processing. Stretch, measured and reported separately: GPU WordPiece
behind `CAP_DEVICE_TOKENIZE`, compared against host tokenization at batch
1, 8, 32.
Gate: conformance green on both machines; goldens; `d2h_hidden_bytes == 0`;
two engines with interleaved create/run/destroy under the allocator pool;
receipts from both machines.

Status (2026-09-21): x86_64 has landed on the RTX 4080 SUPER host
(`providers/cuda/`).
Embed, rerank, classify, and token-classify run through the ONNX Runtime
CUDA execution provider with IoBinding and the provider's own device
kernels for pooling, L2 normalization, sigmoid, and softmax; results stay
on the device and are exported as `TURBO_HANDLE_CUDA_PTR`. Precision
matches the FP32 reference vectors at cosine 1.000
(`testdata/receipts/turbo/cuda-2026-09-21.json`). Every cell stays
`EXPERIMENTAL`: the matched-native benchmark and the two-engine
interleaving test above are not yet done. The Jetson Orin Nano has moved past
"not started": device enumeration originally used the runtime's
`cudaGetDeviceProperties_v2`, which CUDA 13 does not export under that
name, so the provider failed to load there; it now reads compute
capability through `cudaDeviceGetAttribute` and the device name through
the driver library's `cuDeviceGetName`, both stable across CUDA toolkit
majors (`providers/cuda/src/cuda.rs`). With that fixed, and building
`--no-default-features` against a dynamically linked ONNX Runtime 1.24.0
via `ORT_LIB_LOCATION` (`TURBO_CUDA_ARCHS=87`, JetPack R39 rev 2.0, CUDA
13.2), all 12 live embedding tests pass on the Orin Nano at cosine 1.000; there
is no committed receipt for that machine yet and the task suite (rerank,
classify, token-classify) is still being verified there. TensorRT EP,
`user_compute_stream` import, and the GPU WordPiece stretch goal are not
implemented on either machine yet.

Status (2026-09-23): the ONNX Runtime engine described above is the
fallback, not the target. Section 0 and roadmap item R4 make the direct
CUDA encoder on the provider's own kernels the cuda provider's path, with
ONNX Runtime kept only for a model the direct path does not implement.
The matched benchmarks in `testdata/receipts/turbo/bench/` compare the
fallback engine with its own native loop; the direct path is measured
against TensorRT and TEI (R3).

### P4 Metal provider
Swift provider library exporting the plugin vtable; MLX arrays over the
shared arena without copies (pointer-verified); pooling and L2 on the GPU
stream; fixed create policy (HAILO and unknown constants rejected); result
struct laid out in C, not as a Swift struct.
Gate: conformance green on M2; goldens; no host copy of the hidden state
(measured); receipt.

Status (2026-09-22): landed as `providers/metal`, an Objective-C++
provider over Metal directly rather than the Swift-plus-MLX design above.
The change of route is deliberate: the plan's rule is the lowest, fastest
layer the platform allows, and on Apple silicon that is Metal itself, with
no MLX runtime in between and no dependency on Xcode (the provider builds
with `make` and the Command Line Tools' `clang++`, and compiles its Metal
Shading Language kernels at load time). The kernels are the ones the
2026-09-21 proof of concept validated (`native/turborerank`, tag
`poc-2026-09-21`) plus pooling, L2, the NSP pooler and the classifier
head, so every stage after WordPiece runs on the GPU and `turbo_model_info`
says so (`fully_accelerated = 0` for the tokenizer, every other stage
`DEVICE`). Weights come from an F32 `safetensors` artifact and the
architecture from the model's `config.json` (`hf_config` artifact),
cross-checked at load. Unified memory is reported honestly: the device is
`IGPU` with `DEVICE_RESULT`, `UNIFIED_MEMORY` and `HOST_PTR_IMPORT`; token
rows are written straight into shared `MTLBuffer`s, results are
`TURBO_PLACE_SHARED` (exportable as `TURBO_HANDLE_MTL_BUFFER` or a host
pointer), and `h2d_bytes` and `d2h_bytes` stay at zero, which the live
suite now checks for unified-memory devices instead of demanding an
upload. On an Apple M2 Mac (macOS 27): the 15 vtable tests
(`providers/metal/tests/provider_test.cpp`) pass; `live_embed` passes 16
of 16 at cosine 1.000 against the FP32 references with the STS ranking
gate; the rerank cases of `live_tasks` pass with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (whose contract declares the
Identity activation, so scores are logits and the suite now follows the
bundle's `activation` rather than assuming a sigmoid). Receipt:
`testdata/receipts/turbo/metal-2026-09-22.json`; throughput under
`testdata/receipts/turbo/bench/metal-mac-*-2026-09-22.json`: 221 rows/s
at batch 32 by 32 tokens (5.5k tokens/s), 46 rows/s at 32 by 128, 15
rows/s at 32 by 256, and 23 documents/s reranking 32 documents at 128
tokens with the 12-layer cross-encoder; the whole batch is one set of
dispatches per layer with a 16 by 16 tiled matmul, which took the first
build from 21 to 221 rows/s; the same evening the linear layers moved to
simdgroup matrix units (`linear_nt_simd_kernel`, 32 by 32 tiles of 8 by 8
accumulators on Apple GPU family 7 and later, the tiled kernel elsewhere),
622 rows/s at 32 by 32 and 24 rows/s at 32 by 256
(`metal-mac-embed-2026-09-22-simdgroup.json`), with the precision gates
unchanged (cosine 1.000, Spearman 0.9438) and the matched comparison
against the kernels run directly at 0.97x to 1.01x. Not yet: the
attention and layer-norm kernels, which are one thread per output and now
dominate at 128 and 256 tokens; GPU-side WordPiece; F16 weights;
classification heads with more than one logit; and generation (which
reaches the M2 through `ggml`'s Metal backend).

### P5 Hailo provider
Hailo-8/8L on HailoRT 4.24 with `dma_map` zero-copy and async behind
`CAP_ASYNC`; Hailo-10H on HailoRT 5 (5.1.1 on the board today, 5.4 planned)
for the same encoder split once a DFC 5 HEF exists (tracked as open, P11); `fully_accelerated = 0` with stage placement
reported; x86_64 build for PCIe cards.
Gate: conformance green on both Hailo-8 Pis (capability tests assert the
honest limits); goldens within the INT8 tolerance recorded in the bundle;
receipt per board.

Status (2026-09-21): the `hailo` provider (`providers/hailo`, C++ over the
HailoRT 4.23 C API) serves `EMBED x TEXT` on a Raspberry Pi 5 with the AI
HAT+ 26 TOPS and on a Raspberry Pi CM5 with a Hailo-8 M.2 module with the Model Zoo INT8 MiniLM HEF: the HEF's fixed-shape encoder
body runs through f32 vstreams, WordPiece, the word-embedding gather (the
`hailo_tables` artifact), pooling, L2, and `output_dim` run on the host,
and `turbo_model_info` says so (`fully_accelerated = 0`, encode on the
device, everything else host). The capability cell reports `dtype = I8`,
`reference_dtype = F32`, and the measured `cosine_floor` (0.30); the live
suite now gates on the floor the cell states instead of a fixed 0.9995 and
adds a ranking gate (Spearman over `testdata/corpus/sts-pairs.jsonl`,
0.937 on the Hailo-8 against 0.944 for FP32). The 14 vtable tests
(`providers/hailo/tests/provider_test.cpp`) and the 14 live embedding
tests pass on both boards. Receipt:
`testdata/receipts/turbo/hailo-2026-09-21.json`. Not as planned: the
provider uses vstreams with HailoRT's scheduler rather than `InferModel`
with `dma_map` (the 4.23 packages on the Pis; `dma_map` zero-copy and
`CAP_ASYNC` stay open), and HailoRT 4.23 rather than 4.24. Not yet:
Hailo-8L (no board), Hailo-10H (needs a DFC 5 HEF), the x86_64 PCIe build,
and the x86_64 PCIe build. Throughput (2026-09-21, `turbo-bench`,
`testdata/receipts/turbo/bench/hailo-pi5-hailo8-embed-2026-09-22b.json`): 76
rows/s at every batch and sequence length (the HEF is batch 1 with a
128-token frame; 13.2 ms per row).

### P6 Generation
`ggml` provider for GGUF generation across CUDA, SYCL, Metal, and CPU using
the pull iterator, chat templates from the bundle, cancellation, logprobs;
then the push `turbo_generate` wrapper over the same iterator (declared
since P0, stubbed until here);
MLX generation on Apple through the same provider vtable; Hailo-10H GenAI
LLM. Tokenize/detokenize for generative bundles.
Gate: streaming conformance (token order, stop strings, cancel mid-stream,
seed reproducibility on CPU, pull and push producing identical token
sequences for one seed), throughput receipts on the RTX 4080 SUPER host, the
Intel Arc B70 host, the M2, the Jetson Orin Nano, and the Hailo-10H Pi.

Status (2026-09-21): the `ggml` provider (`providers/ggml`, llama.cpp through
`llama-cpp-2`) generates from GGUF bundles on the CUDA and CPU devices of
the RTX 4080 SUPER host and, through llama.cpp's own Metal backend (the provider's `metal`
Cargo feature, not the separate MLX-based `metal` provider this section
scopes), on an Apple M2 Mac; all with the pull iterator, chat
templates from the bundle or the GGUF, stop strings and tokens,
cancellation, logprobs, seeded sampling, and GBNF grammars.
`turbo_generate` (push) is implemented over the pull iterator and its C
conformance test checks the two forms yield one token sequence. Receipt:
`testdata/receipts/turbo/ggml-2026-09-21.json`. GGUF embedding bundles
run through the same provider (pooling in the graph, L2 and `output_dim`
on the host, results placed in host memory) with the 13 live embedding
checks green on the CUDA and CPU devices of the RTX 4080 SUPER host and the
Metal and CPU devices of an Apple M2, all at cosine 0.99999 or better against the
FP32 references; the runs are in the same receipt. Not yet: MLX generation
through the dedicated `metal` provider, Hailo-10H generation,
tokenize/detokenize for GGUF vocabularies, JSON-schema constrained output,
and the throughput receipts.

### P7 Java FFM and Swift packages
Port the conformance suite to Java and Swift (the same cases through the
bindings). Publish native, prepared-token Java, and text Java timings
separately against the C numbers.
Gate: Java suite green under `--illegal-native-access=deny` on the Intel Arc
B70 host and the RTX 4080 SUPER host; Swift suite green on M2; documented
binding overhead.

Status (2026-09-21): the Java binding (`bindings/java`, `ai.pipestream:turbo`)
is in, with the raw layer generated by jextract from `include/turbo/turbo.h`
and the conformance cases passing through it on the mock provider on the
RTX 4080 SUPER host and the Intel Arc B70 host under
`--illegal-native-access=deny`: twelve cases
including generation (pull iterator, cancel, seeded repeatability, refusal
by field) and the tokenizer, then fifteen once the second review added a
held-result BUSY case, a cross-thread cancel case, and a regression case
for the jextract indexed-accessor fix (H-6). The Swift
package (`bindings/swift`, module `PipestreamTurbo`) is in, with its
generation and tokenizer wrappers mirroring the Java ones and the same
fourteen cases as an executable runner, all passing on an Apple M2 Mac
(2026-09-22). Not yet: the push form `turbo_generate` and the
chunk planner in the bindings, and the timings.

### P8 Packaging and SDK
Per-platform archives built in containers (manylinux_2_28 floor for Linux
x86_64 and aarch64; macOS arm64 notarized xcframework; Debian packages for
the Pi), version script, SONAME, `abidiff` gate, Maven classifier artifacts
plus an `-auto` aggregator through the Central Publisher Portal, a
clean-consumer install test per platform.
Gate: a fresh machine per platform installs, verifies a bundle, and runs
the conformance smoke without a source checkout.

Status (2026-09-21): `scripts/package.sh` builds the per-machine archive
(libturbo, headers, `turbo-bundle`, `turbo-bench`, and every provider the
machine can build), gates `libturbo.so` and the mock provider on `ldd`,
compiles and runs the C smoke test against the extracted archive, and
dlopens every packaged provider through `turbo-bench discover --strict`.
`scripts/package-container.sh` builds the Linux x86_64 archive on the
manylinux_2_28 floor in a container (`packaging/Dockerfile`; aarch64 through
qemu binfmt, untested). Not yet: the macOS xcframework, Debian packages for
the Pi, the SONAME and `abidiff` gate, Maven classifier artifacts and the
Central Publisher Portal, and the clean-consumer install test per platform.

### Dependency budget
Fewer crates is faster to build, smaller to ship and easier to audit, so
every crate declares only what it uses and test-only crates live in
`[dev-dependencies]`. Counted with `cargo tree -e normal` on 2026-09-22:
`libturbo` (turbo-shared) resolved 86 crates in the morning, 72 of them
through the Hugging Face `tokenizers` crate; by the evening it resolves
26. The core tokenizer is now native (`crates/turbo-core/src/wordpiece.rs`
over generated Unicode tables, `scripts/gen-unicode-nfd.py`): it reads
BERT-family `tokenizer.json` files (BertNormalizer, BertPreTokenizer,
WordPiece, TemplateProcessing or BertProcessing, the WordPiece decoder)
and a parity test holds it to the Hugging Face crate's ids, type ids,
byte offsets and decoded text over the STS corpus and adversarial strings
(`cargo test -p turbo-core --features hf-tokenizers`). It is also faster:
one thread on the RTX 4080 SUPER host, release build, the native path encodes 5.1M
tokens/s on the STS sentences (2.5 us per text) and 4.7M tokens/s on a
1000-token paragraph, 3.2x and 2.3x the Hugging Face crate on the same
texts (`cargo test -p turbo-core --features hf-tokenizers --release speed
-- --ignored --nocapture`).

Later on 2026-09-22 the byte-level BPE of the GPT-2 family is native too
(`crates/turbo-core/src/bpe.rs`): an NFC normalizer over the generated
composition table, the GPT-2, Qwen2 and cl100k (GPT-4, Llama 3) split
patterns matched by hand (`\p{L}`, `\p{N}` and `\s` from generated
class tables), the byte-to-character table, ranked merges with the
crate's pre-token cache, `ByteLevel`, `TemplateProcessing` and
`RobertaProcessing` post-processors and the `ByteLevel` decoder. The same
parity test holds it to the crate over the STS corpus and a multilingual
corpus (`testdata/corpus/multilingual.jsonl`: 26 languages and scripts,
emoji, decomposed accents, Hangul jamo, special tokens in text, CRLF) on
Qwen3-Embedding-0.6B's file, and it encodes 5.9M tokens/s on the
sentences and 5.5M on the paragraph, 4.1x and 3.4x the crate. Files
neither native path serves (Unigram, sentencepiece BPE with byte
fallback, as e5-mistral and bge-m3 declare) still go to the Hugging Face
crate behind `hf-tokenizers`, or are `TURBO_E_UNSUPPORTED` naming what
they declare.
`turbo-inferstream` resolves tokio, axum, tonic and their runtime on top
of that; nothing else.

### P9 Inferstream on the new ABI
Rebuild the server as a consumer: OIP v2 for `ModelInfer`/metadata over the
`RUN` and `EMBED` tasks, plus OpenAI-shaped `/v1/embeddings`, `/v1/rerank`,
`/v1/chat/completions` with streaming, since that is the ecosystem's
embedding lingua franca (KServe 0.15+ exposes the same routes). `info`
mirrors TEI's fields from `turbo_model_info`. Sessions are fixed-shape, so
the server keeps a session pool per model keyed by (batch, seq) bucket and
pads within a bucket; a request larger than the largest bucket is rejected
with the limit, never silently truncated. Classification and token
classification are served through OIP `ModelInfer` and a `/v1/classify`
extension route.
Gate: existing e2e parity suites pass against the new server on all three
GPU machines; catalog aliases resolve to bundles.

Status (2026-09-22): landed as `server/` (`turbo-inferstream`, binary
`inferstream`, Rust): one engine, two listeners. gRPC serves
`inference.GRPCInferenceService` generated from the protocol's own
`open_inference_grpc.proto` (tonic 0.14, the current line); HTTP serves
the OIP v2 REST binding under `/v2`, the OpenAI-shaped `/v1/embeddings`,
`/v1/rerank`, `/v1/chat/completions` (SSE streaming with cancellation on
disconnect) and `/v1/classify`, and `/info` with the
text-embeddings-inference fields. Every model kind the ABI has is mapped
(embedding, reranker, classifier, token classifier, generative, generic
RUN with typed tensors). Sessions are pooled per `(batch, seq)` bucket
as planned: a request is served by the smallest fitting bucket, split by
the widest batch, and rejected with `TURBO_E_CAPACITY` naming the limit
when its longest text exceeds the longest bucket unless it sets
`truncate` to `right` or `left`; the default never cuts. Errors carry the
Turbo status name and field index on both bindings. Direct dependencies
are tokio, axum, tonic, tonic-prost, prost, serde, serde_json, clap and
futures-core (all already in the tree). Verified on the RTX 4080 SUPER host:
every route
on the six mock bundles, and MiniLM plus the ms-marco reranker through
`cuda` on the RTX 4080 SUPER with Qwen2.5-0.5B through `ggml`, over both
bindings (`server/README.md`). Also landed (2026-09-22): gRPC server
reflection; the `turbo.inferstream.InferstreamExtension` service with
`ModelStreamInfer` (streamed generation over gRPC) and the model
repository (`RepositoryIndex`, `RepositoryModelLoad`,
`RepositoryModelUnload`, and `/v2/repository/...` over REST), so bundles
load and unload while the server runs; 16 more tests for them; the
`turbo-inferstream:cpu` image (`packaging/inferstream/Dockerfile`, about
200 MB, the ggml provider on the CPU) with a KServe `ClusterServingRuntime`
and `InferenceService` under `packaging/kserve/`; and `demo/rag/`, embed,
rerank and a streamed cited answer through the OpenAI SDK and through
KServe's own OIP clients; and `--pages DIR`, which serves a directory of
static files at `/`, for `demo/search/`, a page that embeds 48 passages
in 8 languages with MiniLM and Qwen3-Embedding-0.6B (both through `ggml`
on the 4080) and ranks them for a query in any language side by side,
where Qwen3 returns all 8 same-topic passages for each of 8 queries and
MiniLM 1 to 5. Not yet: the parity suites on all three GPU
machines, a catalog of aliases (the repository extension is where they
would resolve), and the CUDA and OpenVINO providers in the image.

### P10 Android JNI and GraalVM/OpenNLP
JNI adapter implementing `turbo-api`, AAR with `arm64-v8a` and `x86_64`, one
physical device named before work starts. GraalVM 25.2 native-image sample
that loads a bundle and runs `EMBED` and a token-classification `RUN` model,
with foreign-call metadata generated by the tracing agent. This is the
OpenNLP integration shape; the OpenNLP provider itself is written in the
OpenNLP repository against `turbo-api`.
Gate: Android conformance subset on the device; native-image sample runs on
linux-x64, linux-aarch64, macos-aarch64.

### P11 Model zoo and repositories
Scope note (2026-09-23, later the same day): section 0 cuts this milestone
down to the local catalog that the selection calls read, built from the
bundles on disk and the committed receipts. The repository layer, the DJL
view, signing and mirrors below are deferred and kept here as the design
that was reviewed, not as work in order.

Decided 2026-09-23; revised the same day after a review against the
Deep Java Library source, the KServe storage initializer and this tree's
dependency graph. A zoo is the layer that produces bundle artifacts
reproducibly, says what has been proven about each one, and serves them
from ordinary storage. The bundle (section 6) stays the unit every binding
opens; the zoo is an index over bundles plus the recipes and receipts
behind them. Nothing in it touches the run path.

Schema. `proto/zoo.proto` owns the catalog and nothing else: `Catalog`,
`Entry`, `Target`, `Source`, `Recipe`, `ReceiptRef`, `License`. On disk
the catalog is proto3 JSON written with the proto field names preserved
(snake_case, like `bundle.json`) and parsed with unknown fields ignored;
over gRPC the same messages travel as binary. `bundle.json` stays serde
and stays the one source of truth for the contract: the catalog records
the bundle's `manifest_sha256` and its artifact hashes, never a copied
contract. An entry names the model (`group_id`, `artifact_id`,
`version`, `application` in DJL's taxonomy such as `nlp/text_embedding`),
the target (`arch` such as `sm_89`, `aarch64-cuda`, `openvino-gpu`,
`metal`, `hailo8`, `hailo10h`; `format` such as `onnx`, `openvino_ir`,
`gguf`, `hef`; `runtime` and its version range; `quantized`), the source
(checkpoint, revision, sha256; this is where the checkpoint hash lives,
not in `bundle.json`), the licence (SPDX id, url, `redistributable`),
the recipe, the files (path, size, `sha256`; `sha1` computed at index
time for the DJL view), and receipt references. Rust reads and writes the
JSON through `pbjson`, counted in the Dependency budget before it lands;
the Java and Swift zoo types ship as separate artifacts (`turbo-zoo`) so
`turbo-api` gains no protobuf dependency.

Status is derived, never stored. A `ReceiptRef` is `{kind, path, sha256
of the receipt file, machine label}`. `turbo-bundle zoo verify` marks an
entry SUPPORTED only when all three kinds (conformance, precision,
matched-native) are present, each receipt's hash matches, each names the
entry's `manifest_sha256` and artifact hashes, each names the entry's
arch and runtime version, and the matched-native receipt has verdict
SUPPORTED with an empty `dirty` list. Anything less is EXPERIMENTAL with
the missing or failing item named. That is principle 7 applied
mechanically, and it is the same rule the capability cells follow.
Receipts name the files they compare by file name, never by absolute
path (the compare tool was corrected on 2026-09-23 after five receipts
carried a checkout path).

Signing and licences. The catalog is signed (an ed25519 detached
signature over the canonical JSON; Sigstore model-signing when KServe
settles on it) and `zoo fetch` verifies the signature before trusting
any hash in it; a sha256 in an unsigned index fetched from a mirror
proves only agreement with that mirror. `zoo index` refuses to publish
an artifact whose licence forbids redistribution and publishes its
recipe instead; a Hub fetch of a gated checkpoint needs a token and the
entry records that it did.

DJL compatibility, stated exactly. A stock Deep Java Library client can
consume two things without any code on its side: a mirror of our
catalog laid out as a DJL local repository
(`model/<application>/<group id as a path>/<artifact id>/metadata.json`,
artifact files under `<version>/`), loaded through the
`ai.djl.repository.zoo.location=file://...` property, where DJL lists
every `metadata.json`, downloads items and verifies `sha1Hash`; and a
single bundle archive URL passed to `Criteria.optModelUrls`, which DJL
downloads without a checksum. Listing over HTTPS is not something stock
DJL does for a third-party index: `djl://` is wired to DJL's own
repository, `ModelZoo.listModels()` only sees zoos registered on the
classpath, and a bare `https://` index URL is treated as an RPC
endpoint. So we ship a small `turbo-djl-zoo` jar that registers a
`ModelZoo` over a `RemoteRepository` at our base URL; with it on the
classpath the public HTTPS index lists and downloads the same way. The
DJL view is produced by an explicit transform (`zoo index --djl`) from
its own message set (`DjlMetadata`, `DjlArtifact`, `DjlItem`), with
enum-to-DJL-string tables and a golden test that round-trips DJL's own
sample `metadata.json`; `json_name` alone cannot regroup entries by
artifact group or key `files` by item id, so it is not the mechanism.
The view publishes only what DJL can run: ONNX artifacts for embed,
classify and token-classify, each bundle as one zip item with
`arguments.engine=OnnxRuntime`, DJL's translator factory, `pooling` in
DJL's spelling, `normalize` as a boolean and `maxLength`. HEF, GGUF and
OpenVINO artifacts are not listed there, because DJL has no engine for
them; running them from DJL needs a DJL `Engine` over `turbo-ffm`, which
is a separate deliverable and not part of this claim. SageMaker's DJL
Serving is not claimed until that jar has been run inside it. The
reverse direction, DJL's repository and the Hugging Face Hub as sources
for `turbo-bundle import`, uses the repository layer below.

Repositories. One `Repository` interface (resolve, fetch, verify) in
`turbo-bundle` behind a `net` feature; libturbo never fetches, and the
mobile bindings open bundles already on disk. The tree has no HTTP
client or TLS stack of its own today (the server's hyper is plain HTTP
through tonic), so the feature brings one blocking client, `ureq` with
`rustls` and `webpki-roots`, counted in the Dependency budget before it
lands. Implementations are thin: local directory (`file://`); HTTPS
static (DJL's repository, GitHub releases, any web server); S3 through
the REST API with SigV4 hand-written over the `sha2` already present
and a small HMAC, no vendor SDK; the Hugging Face Hub through its
resolve endpoint (a redirect to a CDN, a bearer token for gated repos);
OCI artifact registries in the ORAS layout (token flow, blobs by
digest). Google Cloud Storage and Azure Blob follow S3. `zoo mirror
<dir>` writes a self-contained tree with the catalog, the bundles and
the receipts, which is both the offline answer for an air-gapped Pi and
the DJL local-repository tree above; `--offline` never opens a socket.
Every file is content-addressed by sha256 in the index; the repository
type never changes what verified means, and `Bundle::open` hashes every
file against `bundle.json` at load on every path regardless.

KServe, stated exactly. `s3://`, `gs://` and `https://` (a bundle
archive, since the initializer unpacks a single file) mean the same
bytes to KServe's storage initializer and to `zoo fetch`. `pvc://` is
KServe-only: the controller mounts the claim and `turbo-bundle` cannot
resolve a claim name outside a cluster. `oci://` in KServe is a Modelcar
image (a container with `/models`, run as a sidecar), not an ORAS pull,
so a Modelcar image is published beside our OCI artifact. `hf://` in
KServe fetches a Hub snapshot, a checkpoint source, not our bundle.
`file://` is a path inside the initializer container and is only useful
with a hostPath, so it is not in the gate. The initializer verifies no
checksums, so the serving runtime takes `--expect-manifest-sha256` and
refuses a bundle that is not the qualified one.

Recipes. A recipe takes a source checkpoint and a target and yields the
artifact plus the facts the bundle needs (frame length, fixed shape,
quantization). No Python enters this tree for it. A recipe is a
declarative record in the catalog: the toolchain as a container image,
the command lines, the inputs by hash (checkpoint, model script,
calibration set), the host architecture it runs on, and the outputs.
`turbo-bundle zoo build` runs the container, captures the artifact,
hashes it and writes the entry; `fetch`, `verify`, `index` and `mirror`
are the other verbs. The vendor's Python runs inside the vendor's image
and our side is Rust. Import beats convert wherever an artifact is
already published: ONNX and GGUF from the Hugging Face Hub, HEFs from
Hailo's zoo where they exist; a toolchain runs only for what nobody
publishes, which today is the Hailo-10H compile. Reproducibility is
stated honestly: Hailo's AI Software Suite image is downloaded from the
Developer Zone under an EULA that bars redistribution, so there is no
public registry digest to pin; the recipe records the archive's sha256
and the loaded image id, rerunning it needs the vendor's account, it
runs on x86_64 only, and quantization is not promised bit-identical, so
a rebuilt artifact is accepted by its precision and conformance
receipts, with byte equality recorded when it happens. llama.cpp's GGUF
quantizer is public and is pinned by digest. There is no Core ML
recipe, because there is no Core ML provider; the Metal provider uses
Metal directly (section 7). The one conversion the Hailo bundles need
in-house, the fp32 embedding tables exported from the checkpoint
(`scripts/export-hailo-tables.py` today), moves into `turbo-bundle`
over the safetensors reader it already has
(`tools/turbo-bundle/src/safetensors.rs`), and the script is removed.
The Python that remains in the tree is the reference edge (PyTorch
reference generators, the Hailo native receipt over `hailortcli`) and
demos over the OpenAI SDK; none of it is distributed and none of it is
on a build path.

Vector storage is outside this library. The library produces vectors and
stays storage-agnostic. The search demo gains a `sqlite-vec` backend
(plain C, runs on the Pi, on mobile and in the browser); usearch and
LanceDB are the other embedded options a consumer may choose; pgvector and
the server-class stores are the deployment's business.

First entry: MiniLM for `hailo10h`, compiled with Dataflow Compiler 5.1 to
match the HailoRT 5.1.1 on the Hailo-10H Pi (5.4 is a later upgrade of
both, taken together), since no public HEF exists and it exercises every
field the schema has. Then the bundles already in use (MiniLM ONNX and
GGUF, the ms-marco reranker, SST-2, BERT NER, Qwen2.5 GGUF, MiniLM
`hailo8`) get entries with the receipts they already have.

Order: the schema, `verify` and the local repository with `zoo mirror`;
the DJL local-repository view with its golden test; the `net` feature
with HTTPS and S3; the Hailo-10H recipe and entry; OCI plus the Modelcar
image; Hub import; the `turbo-djl-zoo` jar last.
Gates: a stock DJL 0.39 client with `ai.djl.repository.zoo.location`
pointed at a `zoo mirror` tree lists the MiniLM entry, downloads it with
`sha1Hash` verified, and fails on a tampered file; with `turbo-djl-zoo`
on the classpath the same holds against the public HTTPS index. `zoo
fetch` refuses a file whose sha256 does not match and refuses a catalog
whose signature does not verify. `zoo verify` reports the Hailo-10H
entry SUPPORTED only with its three receipts bound to its hashes, and
names what is missing otherwise. The KServe manifest serves the same
bundle through `pvc://` and through `s3://` (a MinIO in the test
cluster) and refuses a mismatched manifest hash. No hostname, home path
or checkout path appears in the catalog, the receipts or the mirror.

Dependencies: P0 then P1 are strictly first. P2, P3, P4, P5 are independent
after P1 and can run in parallel on their machines; P2 sets the correctness
baseline so it starts first. P6 needs P1 and at least one GPU provider. P7
needs P2 or P3. P8 needs P7. P9 needs P8. P10 needs P8. P11 needs P8 for
`turbo-bundle`, P5 for its first entry, and P7 for the `turbo-djl-zoo`
jar.

## 11. Conformance and benchmark protocol

Conformance is one suite, parameterized by provider and device, run through
the C ABI (a C test binary) and through each binding. A provider is listed
as supported only when every group passes or is explicitly excluded by a
capability bit that the suite verified is reported.

Benchmarks are matched: for each provider, a direct-native reference program
using the runtime alone, and the same workload through `libturbo`. Workloads
are batch {1, 8, 32} by sequence {32, 128, 256} for embeddings, 32 documents
for rerank, and 128 new tokens for generation. Reported: p50/p99 latency,
tokens/s (when the bundle's tokenizer counted the tokens), H2D/D2H bytes
and host allocations per run, device memory; p99 from 100 iterations up.
Receipts carry machine ID, runtime versions, driver versions, bundle hashes,
and the commit the tool was built from. Budgets are set from the first run
per provider and then held: the budget must name the same provider, device
and bundle, and a cell the budget has that a later run did not measure is a
violation.

The native side of a pair is the runtime driven the way its own users
drive it (the `reference/` programs: the ONNX Runtime CUDA execution
provider through the `ort` crate, llama.cpp through `llama-cpp-2`,
OpenVINO's C++ API, `hailortcli benchmark`, the Metal proof of concept),
on the exact token rows the `libturbo` run used (`turbo-bench embed
--dump-tokens` writes them, the reference reads them, both receipts name
the same bundle hashes), with pooling and normalization done where such a
user would do them (on the host). `turbo-bench compare` reads the two
receipts, matches cells, and reports `libturbo` throughput as a fraction
of native per cell; a (device, task) is SUPPORTED when every cell reaches
0.95 of native, nothing is unmatched, and the comparison receipt is
committed under `testdata/receipts/turbo/bench/compare-*.json`. The first
such receipt was `compare-cuda-rtx4080-embed-2026-09-22.json`, since
re-run from commit 8d7b6dc as `compare-cuda-rtx4080-embed-2026-09-23.json`:
the cuda provider on the RTX 4080 SUPER against ONNX Runtime 1.28's CUDA
execution provider on the same 9 cells, `libturbo` at 1.00x to 1.15x of native
(the provider keeps the hidden state on the device and pools with its own
kernels; the plain loop copies it back), verdict SUPPORTED for EMBED on
that device. On the Jetson (`compare-cuda-orin-nano-embed-2026-09-22c.json`)
the same pair is 0.92x to 1.05x with 1x32 at 0.916x, 1x128 at 0.947x under the line, so the integrated GPU
stays EXPERIMENTAL. The same day: openvino embeddings on the B70 (1.15x to
1.55x) SUPPORTED, ggml generation on the 4080 (0.99x) SUPPORTED, hailo
embeddings on the Hailo-8 (1.00x of `hailortcli benchmark`) SUPPORTED,
metal embeddings on the M2 (1.00x of the kernels run directly)
SUPPORTED. openvino on the Ryzen CPU first read 0.91x on the larger
cells; the reference's attribution knobs (`--static`, `--fuse`,
`--i32`, which reproduce the provider's graph) put the graph choices
at a 4 percent gain, so the gap was the provider's: its token writer
built an error message string per token (about 1 us each) whether or
not the check failed. Built only on failure, the pair is 1.03x to
1.34x, SUPPORTED (`compare-openvino-rtx4080-cpu-embed-2026-09-22c.json`, the re-run from a clean commit);
the Metal and Hailo token writers had the same pattern and the same
fix. ggml GGUF embeddings on the GPU first read 0.94x on 1x32
and 8x128; both causes were in the protocol, not the provider: the
llama.cpp reference tokenized outside its timed loop while the
provider's text path tokenizes inside it, and a warm-up counted in
iterations left the first cells of a run on a GPU still raising its
clocks. With the reference tokenizing in the loop and a warm-up of at
least 0.5 s on both sides (`turbo_bench::receipt::warm_up`), the pair
is 0.99x to 1.89x, SUPPORTED
(`compare-ggml-rtx4080-gpu-embed-2026-09-22c.json`, re-run from a clean commit).

## 12. Risks and defaults chosen

| risk | default in this plan |
|---|---|
| Hailo-10H embedding HEF needs a DFC 5 encoder compile | Ship Hailo-10H generation first (HailoRT GenAI); track the embedding HEF as an open item with the DFC steps documented |
| No JetPack 7 ORT wheel channel; the `sbsa/cu130` aarch64 wheel may lack sm_87 kernels | The Orin Nano stays on JetPack 7.2.1 (owner keeps it current); probe the wheel first, fall back to a pinned ORT source build (CUDA 13.2, TensorRT 10.16, sm_87); the recipe is committed under `providers/cuda/jetson/` |
| Intel NPU hardware not yet available | NPU code path built and unit-tested with static shapes; supported status withheld until a receipt exists |
| A Metal result that is not the GPU's own memory | Every buffer is `MTLResourceStorageModeShared`; the result's `host_ptr` is the `MTLBuffer` contents and `d2h_bytes` is 0, which the live suite asserts on unified-memory devices |
| OpenVINO GenAI RAG pipelines allocate outputs | Not used; the fused-graph path is the provider |
| Swift `@c` needs Swift 6.3 | Toolchain floor Swift 6.3; `@_cdecl` only as a temporary shim |
| GraalVM has no macOS x64 FFM | Documented; macOS Intel is not a target |
| `--illegal-native-access=deny` may become default in a later JDK | CI already runs under `deny` |
| KServe OIP is frozen and unused by the embedding ecosystem | Keep OIP for predictive compatibility; OpenAI-shaped routes are the primary embedding surface |

## 13. Decisions for the owner

1. Prefix and library name `turbo_` / `libturbo` with TurboEmbed as the
   project name. Decided 2026-09-21.
2. Refactor in place: tag the PoC and remove it from the tree. Decided
   2026-09-21.
3. Jetson runs JetPack 7.2.1, which is what the Orin Nano has. Decided 2026-09-21.
4. Generation streaming: both forms in the header; pull iterator first,
   push wrapper stubbed until pull passes conformance. Decided 2026-09-21.
5. Java package `ai.pipestream.turbo`, Maven group `ai.pipestream`.
   Decided 2026-09-21.

All five were decided on 2026-09-21; execution proceeds from P0.

## 14. Sources consulted (2026-09-21)

ONNX Runtime 1.30.0 release notes and `onnxruntime_c_api.h`; ONNX Runtime
GenAI 0.16.0 `ort_genai_c.h`; OpenVINO 2026.4.0 release notes, NPU device
docs, `intel_gpu/ocl/ocl.hpp`, `intel_npu/level_zero/level_zero.hpp`;
OpenVINO GenAI 2026.4 `rag/text_embedding_pipeline.hpp`; HailoRT 5.4.0 and
4.24.0 (`hailo8` branch), `hailo/genai/llm/llm.hpp`; NVIDIA JetPack 7.2.1
and 6.2.3 release pages, `pypi.jetson-ai-lab.io`; MLX 0.32.2, mlx-c 0.6.0,
mlx-swift 0.31.6; llama.cpp v0.4.1 `llama.h` and `ggml-backend.h`; TEI
1.9.4 router and `tei.proto`; DJL 0.38.0 `Engine.java` and `Criteria`;
sentence-transformers 6.1.0 pooling and config schemas; KServe 0.20 and the
open-inference-protocol repository; ExecuTorch 1.5 `Error`; LiteRT 2.2 C
API; IREE runtime C API; JEP 454 and JDK 25 restricted-methods docs;
jextract `25-jextract+2-4`; GraalVM 25.2 FFM reference; Android NDK r30 and
16 KB page-size guide; SE-0495; UniFFI 0.32.1; cbindgen 0.29.4; manylinux
policy (PEP 600); libabigail `abidiff`; Maven Central Publisher Portal.
