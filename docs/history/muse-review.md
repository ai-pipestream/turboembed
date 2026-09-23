# MUSE-REVIEW — turboembed / inferstream

Historical review supplied before the 2026-09-13 planning update. Its findings
and proposed priorities are preserved below; they are not current validation.
The [roadmap](ROADMAP.md) supersedes its delivery order, and the
[subsequent code review](docs/code-review-2026-09-13.md) corrects several claims.

**Verdict: not bullshit — but "all platforms, mostly done" is overclaimed.**

This iteration is substantially real where the last two allegedly were not:
real tokenizer → ONNX/GenAI/MLX → mean+L2 → FP32 embeddings, a frozen C ABI
kept 1:1 in Rust and Swift, a gRPC server that routes to real backends, and
explicit fail-loud guards that keep the 8-d FNV mock from masquerading as a
model. I verified each of those in file bodies (paths below).

What is *not* done is the entire Java side, and there is zero scaffolding
for the native OpenNLP future. "Mostly done except the Java bridge" is only
true if you mean the Rust/C/gRPC core. For the stated goal — "all-encompassing
inference library for all platforms" with C/Rust + gRPC + Java SPI — the Java
bridge is at 0%, and the docs currently plan it in the opposite order from
what you asked for.

How this was checked: 6-angle workflow sweep (build, core pipeline, C/Rust
ABI, gRPC, Java bridge, tests/bullshit-detector) + skeptic + synthesis, then
hand-verified by reading the cited bodies, grepping for Java/SPI/Panama,
OpenNLP, and "closest to metal", diffing the Swift header copy, and
attempting `cargo test` (blocked by sandbox — see §7).

## 1. What Cursor got right (real, not stubbed)

- **Workspace/build is real.** 20-member workspace, resolver 2, edition 2021,
  rust 1.85, tonic 0.12 / prost 0.13 / tokio / tokenizers 0.23 (fancy-regex,
  no C++ deps) — [Cargo.toml](Cargo.toml).
- **Text → embeddings path is real.** HF `tokenizers` server-side +
  int64 tensors → hidden state → mask-weighted mean/CLS + L2 → FP32;
  ORT CUDA path uses IoBinding with PINNED/DEVICE arenas, warmup, and
  zero-alloc hot path (`d2h_hidden_bytes == 0`) —
  [lib.rs](crates/turboembed/src/lib.rs),
  [engine.rs](crates/backend-ort/src/engine.rs),
  [ort_cuda.rs](crates/turboembed/src/ort_cuda.rs),
  [pool_cuda.cu](native/turboembed/src/pool_cuda.cu).
- **C ABI v1 is frozen and honest.** 8 statuses, 9 devices, pooling/output
  enums, pointer+length views, engine-owned outputs, "not thread-safe on one
  engine, serialize calls" — and the Rust `extern "C"` block matches it 1:1:
  [turboembed.h](include/turboembed.h),
  [ffi.rs](crates/turboembed/src/ffi.rs).
  Swift copy is byte-identical (I ran `diff -q` → IDENTICAL):
  `swift/Sources/TurboEmbedC/include/turboembed.h`.
- **Mock isolation is the strongest anti-bullshit signal.** The 8-d FNV-1a
  mock exists ([lib.rs](crates/backend-mock/src/lib.rs))
  but catalog aliases *refuse* it: `device=mock` → `Unavailable`, aliases
  without a provider feature → `NotImplemented`, GPU requests never silently
  fall back to CPU — [lib.rs](crates/turboembed/src/lib.rs#L26-L57),
  [lib.rs](crates/backend-turboembed/src/lib.rs#L1-L60),
  [turboembed.md](docs/turboembed.md#L1-L30).
- **gRPC is real, two services.** Vendored OIP (`GRPCInferenceService`:
  ServerLive/Ready, ModelInfer, bidi ModelStreamInfer) plus
  `InferstreamService` (Tokenize, Detokenize, Embed, EmbedStream, ListModels,
  Rerank) — [open_inference_grpc.proto](proto/open_inference_grpc.proto),
  [inferstream_extension.proto](proto/inferstream_extension.proto).
  Server resolves `BackendKind` per model and maps `BackendError` →
  `Status` (not_found/invalid_argument/unavailable/internal), Bearer auth —
  [extension.rs](crates/server/src/extension.rs),
  [catalog.rs](crates/server/src/catalog.rs).
- **Tests guard against the exact fakes you worried about.** Cosine-vs-golden
  floors, "FAKE / BERT pooler suspected" assertions, `mock_kill`,
  `abi_smoke`, `device_policy` —
  `crates/turboembed/tests/{abi_smoke.rs,mock_kill.rs,device_policy.rs,apple_minilm_metal.rs,nvidia_minilm.rs}`.

No `unimplemented!`/`todo!` in `crates/turboembed/src`,
`native/turboembed`, `crates/backend-ort/src`, `crates/server/src` (grepped —
zero hits). The stub that *does* exist ([stub.cpp](native/turboembed/src/stub.cpp))
fails loud by design.

## 2. "Closest to the metal" — partly earned, phrase itself absent

`grep -rn "closest.to.metal"` over the repo returns **zero hits** — the
marketing phrase lives only in your prompt, not the code. The nearest real
techniques (arena-rented ZE SHARED/HOST USM rows, `InferRequest.set_tensor`
wrapping pointers instead of copying, CUDA IoBinding DEVICE buffers with
on-device mean+L2, `genai_xfer_*` counters proving zero-copy in
[genai.cpp](native/turboembed/src/genai.cpp))
are the right *kind* of claim, but I could not execute them here (no GPU,
and `cargo test` is sandbox-blocked — §7). Treat "closest to metal" as
**plausible but unproven in this environment**; the live-GPU proof docs
(`docs/*-machine-a.md`, `docs/turboembed.md` NVIDIA section) are the evidence
to demand on real hardware.

## 3. Java bridge — the gap (this is the "mostly done" caveat)

- `find . -iname '*java*' -o -iname '*jni*' -o -iname '*jextract*'` →
  **empty**. `grep ServiceLoader|panama|java.lang.foreign` → **empty**.
  There is no JNI layer, no Panama FFM layer, no `module-info`, no SPI
  descriptor, no `turboembed-java` module. Swift (Package.swift + Sources +
  Tests) is the only non-Rust/C client — and it works, which makes Java's
  absence sharper.
- Worse, the recorded plan contradicts yours. [turboembed-architecture.md](docs/turboembed-architecture.md#L160-L173)
  says: Java comes **later via grpc-java stubs**, and **"JNI / Panama …
  is optional later … Do not start there."** You asked for the opposite
  (in-process modern bridge first, as an SPI plugin). Neither path is started.
- The name you forgot: **Project Panama — Foreign Function & Memory (FFM)
  API, `java.lang.foreign` (JEP 454, final in JDK 22), plus the `jextract`
  tool to generate bindings from `turboembed.h`.** That is the correct
  "more modern one" to build first per your instruction — not JNI.

To close it (in-process, SPI, Panama-first):

1. `jextract --output src/main/java -t org.turboembed.ffi
   include/turboembed.h` → check in or vendor generated sources (don't rely
   on every user having jextract).
2. `Arena`/`Linker`/`SymbolLookup` wrapper opening `libTurboEmbed.so/.dylib`
   (arch-specific: CUDA EP on nvidia, GenAI on intel, MLX dylib on mac).
3. SPI: `META-INF/services/<your.spi.EmbedderProvider>` + `ServiceLoader`
   implementation delegating to the FFM wrapper; honor the ABI's ownership
   rules (caller-owned views, engine-owned outputs, serialize calls per
   engine — the ABI is `Send` but **not `Sync`**).
4. Fail-loud parity: replicate the Rust device policy (AUTO = host GPU, never
   silent CPU/mock; surface `NOT_IMPLEMENTED`/`UNAVAILABLE` as typed
   exceptions).
5. grpc-java stubs stay as the *remote* fallback, not the SPI path.

## 4. Native OpenNLP future — no scaffolding at all

`grep -rni opennlp .` → **zero hits**. No crate, no header, no doc, no issue
stub. Nothing about the current architecture blocks it (the provider vtbl in
[ffi.rs](crates/turboembed/src/ffi.rs#L104-L124)
is the obvious seam — "later model2vec" is already named there), but there
is nothing to review yet. If you want it planned, the seam is
`turboembed_provider_vtbl` + `turboembed_register_provider` (currently
reserved/stub-NotImplemented per
[lib.rs](crates/turboembed/src/lib.rs#L589)).

## 5. Smaller correctness / hygiene notes

- **C ABI thread contract limits "all-encompassing".** One engine = serialize
  calls; scale via distinct engines on distinct threads. Correct and clearly
  documented, but any Java SPI wrapper must enforce it (easy to get wrong
  with a shared `Arena`/executor).
- **Packed-bytes aliasing** (typed `f32` + LE-packed `u8` alias one
  allocation, one free) is documented and matched in Rust — good, but JNI and
  FFM wrappers must not double-free the two views.
- **Naming drift:** the wire is `inferstream.*`, the lib is `turboembed`,
  the repo URL in [Cargo.toml](Cargo.toml#L30)
  is still `https://example.invalid/inferstream`, version `0.1.0`. Fine for
  now, but it will confuse the Java/SPI naming decision — settle
  `org.turboembed` vs `org.inferstream` before generating the Java package.
- **No silent-fallback policy is load-bearing — keep it.** Any future Java
  convenience overload ("just give me embeddings") must not reintroduce
  CPU/mock fallback behind a default argument.

## 6. Prioritized fix list

1. Decide Java order explicitly (your stated order contradicts the docs):
   Panama FFM SPI first, grpc-java remote second — then update
   [turboembed-architecture.md](docs/turboembed-architecture.md#L160-L173).
2. Ship `turboembed-java` FFM bindings + SPI plugin with fail-loud device
   parity (§3).
3. Add a Java-side `device_policy` equivalent test (mock-kill: assert `minilm`
   on `mock` fails, never returns 8-d vectors).
4. Settle group/artifact naming (`turboembed` vs `inferstream`) and fix the
   placeholder repository URL before publishing anything Maven-shaped.
5. Run the GPU golden suites on real nvidia/intel/apple hardware and paste
   the cosine tables into the review record — this sandbox cannot (see §7).
6. When OpenNLP starts, register it behind `turboembed_provider_vtbl`
   rather than forking the header (ABI v1 stays frozen).

## 7. What I could not verify (honest limits)

- `cargo test -p inferstream-backend-mock -p turboembed --lib` **did not run**:
  sandbox denied `rustc` execution (`Operation not permitted` on every crate
  compile). Build/test status is therefore **unverified by me**, not green.
  The workflow's "CI fmt+clippy+test" bullet is unconfirmed here — check
  `.github/` on a real runner.
- GPU-dependent claims (IoBinding zero-copy counters, cosine floors,
  `d2h_hidden_bytes == 0`) are code- and doc-supported but **not executed**
  here. The fraud guards look right, but the numbers need hardware.
- Workflow synthesis noted two residual gaps I carry forward unchanged:
  "OpenNLP: no evidence" (confirmed — zero hits) and one truncated result
  tail (irrelevant — I re-read the bodies directly).
