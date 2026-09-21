# TurboEmbed review, 2026-09-13

Reviewed commit: `9d03af93fbf683d4f76d79b2ee6e4dc1e22422cb` on `main`.
The checkout matched its local `origin/main` tracking reference; remote HEAD and
hosted CI were not refreshed. `MUSE-REVIEW.md` was already untracked and was left
unchanged. This review adds documentation only.

The core is substantial and worth building on. The C API, native providers,
buffer arena, Rust adapters, Swift implementation, and gRPC integration are
real code. However, the public wrappers have memory-safety defects, independent
CUDA engines share allocator state incorrectly, and tokenization and options
are not yet a consistent cross-platform contract. Resolve these before treating
the library as ready for additional language bindings.

Scope: close reading of the embedding ABI, Rust and Swift wrappers, CUDA arena
hooks, Intel dispatch, shared tokenizer, gRPC embedding adapter, build setup,
tests, and selected receipts. Reranking and generation received orientation-level
inspection, not a full kernel or numerical audit. Unless identified as a test
result below, findings are from source inspection, not executed reproducers.

## Findings

### 1. High: embedding results can outlive their engine in safe wrappers

Rust `Embeddings` stores only a raw result pointer, with no engine borrow or
retained owner. `embed_one`, `embed`, and `embed_stream` return it independently
of `Engine`. Safe Rust can obtain a result, drop the engine, then read or drop
the result. The native engine destructor destroys the arena, including rented
slabs. Reading the result then accesses freed memory; freeing it calls back into
the destroyed arena. A documentation instruction to drop results first cannot
make this a safe API.

The Swift wrapper has the same ownership gap for Metal results: its `Embeddings`
does not retain `Engine`, and result records retain a raw arena pointer.

Evidence: [Rust result](../crates/turboembed/src/lib.rs#L325),
[native destructor](../native/turboembed/src/stub.cpp#L166),
[result free](../native/turboembed/src/stub.cpp#L1242),
[arena destruction](../native/turbo_buffer/src/arena.cpp#L434), and
[Swift wrapper](../swift/Sources/TurboEmbed/TurboEmbed.swift#L77).

Required direction: enforce a lifetime relationship or retain an owner until
all results are released. A future Java `close()` contract must prevent engine
destruction while native result views remain usable. Include close/read/drop
ordering in the eventual contract tests.

### 2. High: empty Rust aliases can trigger an out-of-bounds native read

`Engine::load_model` passes `alias.as_ptr()` and `alias.len()`. The C header
explicitly defines a zero alias length as a NUL-terminated-string convention,
so the C++ implementation calls `strlen` before reporting an empty alias. Rust
strings do not promise a terminating NUL, and an empty string's pointer need
not point to readable storage. Thus `load_model("")` can fault in safe Rust;
an empty subslice can instead cause bytes outside the supplied view to be read
as an alias. The safe wrapper fails to uphold that native precondition.
Embedding dispatch repeats this behavior. Apple alias helpers also interpret
length zero as a request for `strlen`; Swift's `withCString` wrapper supplies
termination, unlike the Rust wrapper.

Evidence: [native contract](../include/turboembed.h#L241),
[Rust call](../crates/turboembed/src/lib.rs#L449),
[native load](../native/turboembed/src/stub.cpp#L689),
[native embed](../native/turboembed/src/stub.cpp#L899), and
[Swift alias helper](../swift/Sources/TurboEmbed/ABI.swift#L597).

Required direction: reject empty aliases in the safe Rust wrapper, or explicitly
supply a terminated string when using this convention. Clarify the alias
exception alongside the header's general pointer-plus-length rules; do not
silently change the existing C calling convention.

### 3. High: CUDA allocation callbacks are not isolated per engine

`EXT_ARENA` holds one process-wide arena pointer. Loading a CUDA session replaces
it. Both `gpu_external_alloc` and `gpu_external_free` consult that pointer;
`EXT_SLABS` records a view but not its owning arena. After loading engines A and
B, a later allocation for A can rent from B, and freeing an earlier allocation
from A attempts to return it to B. Destroying B can invalidate allocations still
used by A. The registry mutex protects access to a pointer, not its lifetime.

This conflicts with the header's independent-engine concurrency contract and
the server's one-engine-per-alias design. It can also occur with sequential
calls across multiple live engines. No live CUDA reproduction was run.

Evidence: [callback implementations](../crates/turboembed/src/ort_cuda.rs#L174),
[arena selection at load](../crates/turboembed/src/ort_cuda.rs#L642), and
[server engine ownership](../crates/backend-turboembed/src/lib.rs#L87).

Required direction: design allocator ownership around the callback facilities
actually provided by ORT. Verify interleaved creation, inference, and destruction
of two engines; adding another mutex around individual embedding calls is not
sufficient to establish ownership.

### 4. High: Swift input copying breaks embedded-NUL strings

The Swift wrapper uses `strdup(text)` but advertises `text.utf8.count` to the C
API. A string such as `a\u0000bc` is copied only through its first NUL, while
the advertised view covers the full original UTF-8 length. Native consumers
can read beyond the allocated copy. This affects ordinary Swift `String`
inputs, including the mock path.

Evidence: [Swift input construction](../swift/Sources/TurboEmbed/TurboEmbed.swift#L43).
Required direction: copy the actual UTF-8 byte span and pass its exact length.
JNI must likewise convert to standard UTF-8 rather than treating JNI Modified
UTF-8 as the native ABI's encoding.

### 5. High: Swift concurrency promises exceed the native implementation

The public Swift `Engine` is `@unchecked Sendable` and has no lock or actor
isolation. Callers can share it across tasks, concurrently mutating native
engine state despite the ABI's serialization requirement. Separately, the
Apple `TLS.shared` create-error slot is global, not thread-local. Its getter
returns a pointer after releasing the lock; another thread can replace and
free the string before the first caller copies it. Distinct engines therefore
do not isolate creation errors either.

Evidence: [public Swift engine](../swift/Sources/TurboEmbed/TurboEmbed.swift#L12)
and [create-error storage](../swift/Sources/TurboEmbed/ABI.swift#L54).
Required direction: enforce the promised sharing model and make error-pointer
lifetimes match the header. Review callback reentry and panic/exception handling
at the same boundary; Rust currently invokes the user closure directly from an
`extern "C"` callback.

### 6. High: the optimized tokenizer is selected without verifying compatibility

The native JSON loader searches for a WordPiece marker but falls back to any
`"vocab"` object when it cannot find one. The ORT path prefers this loader to
the HF tokenizer. A BPE tokenizer with an object vocabulary can therefore be
accepted and processed as uncased BERT WordPiece, bypassing its own tokenizer
configuration. Even a WordPiece vocabulary is insufficient to establish
normalization and special-token behavior.

The custom normalizer also differs from BERT normalization. For example it
maps `ß` to `s` and `ø` to `o`. HF's BERT normalizer uses Unicode decomposition
and removal of combining marks, which does not make those substitutions.
Such changes alter token IDs before inference; vector dimension checks and a
few cosine examples will not catch every case.

Evidence: [JSON selection](../native/wordpiece/vocab_load.cpp#L357),
[ORT selection](../crates/turboembed/src/ort_cuda.rs#L613),
[native normalization](../native/wordpiece/encode.cpp#L63), and
[HF BERT normalizer source](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/normalizers/bert.rs).

Required direction: explicitly gate the fast path to supported tokenizer
configurations and compare token IDs against the pinned reference implementation.
Include Unicode, cased models, special tokens, and long-word limits. This
review did not run a multilingual model-parity experiment.

### 7. Medium: per-call embedding options differ across providers

Linux ORT dispatch forwards pooling and normalization but never `truncate_to`.
Intel dispatch also does not read that field; both use load-time sequence
configuration. Swift forwards truncation to its provider. A caller therefore
gets different behavior for the same public option depending on the host.
Intel also only checks explicit CLS/LAST overrides, so requesting MEAN for a
BGE alias configured for CLS is silently accepted without changing the pooling.

Evidence: [ORT dispatch](../native/turboembed/src/stub.cpp#L919),
[Intel option checks](../native/turboembed/src/stub.cpp#L1016),
[Intel default pooling](../native/turboembed/src/stub.cpp#L252), and
[Apple option forwarding](../swift/Sources/TurboEmbed/ABI.swift#L573).

Required direction: define whether each option overrides the loaded model or
must match its configuration, then implement or explicitly reject it on every
provider. Do not expose silently ineffective settings through Java.

### 8. Medium: default parallel tests interfere through global counters

The default workspace command failed in
`output_scratch::tests::bytes_clone_returns_once`: actual `returns()` was 4,
expected 2. Tests share `ALLOCS`, `RENTS`, `RETURNS`, and the pool, reset them
independently, then assert exact values or deltas while other tests run.
The serial workspace command passed. This supports a test-isolation diagnosis,
not a claim that pooled response ownership is broken.

Evidence: [global counters](../crates/protocol/src/output_scratch.rs#L42) and
[tests](../crates/protocol/src/output_scratch.rs#L253).
Required direction: isolate measurements or serialize the relevant tests.
Keep the default CI command as the acceptance criterion.

## Build and documentation gaps

There is no Java module, JNI bridge, Android project, or FFM binding in this
checkout. The C ABI is a useful boundary, but it is not yet a packaged native
SDK for those consumers. The Rust crate defaults to an rlib, and Linux native
objects are assembled by Cargo build scripts. The standalone Make target builds
a mock static archive. Apple has a shared-library product. The native README's
handwritten shared-library command compiles only `stub.cpp` and omits the arena
and tokenizer objects now required by it.

Before generating bindings, define reproducible native artifacts, exported
symbols, transitive runtime dependencies, model lookup, and supported targets.
Build scripts currently include host compiler/library probes and checkout paths;
those need a deliberate cross-compilation and relocation strategy for Android.

MUSE correctly identified absent Java support and reserved provider registration,
but its endorsement of wrapper safety was too strong. Its Intel description is
also stale: the current native path reads and compiles an OpenVINO model with a
custom tokenizer rather than calling `TextEmbeddingPipeline` for each embed.
The existing Java architecture note recommends gRPC first; the current user
request establishes interest in FFM and JNI, not a finalized delivery order.
MUSE's OpenNLP discussion does not by itself establish additional project scope.

Documentation should distinguish TurboEmbed from Inferstream, separate historical
machine receipts from current support, and describe exactly what an allocation
counter measures. The ORT path still creates vectors, strings, and result
records during embedding; zero arena-allocation counts are not a whole-process
zero-allocation claim. Preserve useful evidence while removing repetitive
status slogans and unexplained milestone labels. The root Cargo repository URL
is still a placeholder and needs correction before publication.

## FFM, JNI, and Android direction

Recommendation, not an implementation commitment: expose one Java-facing API
with separate local FFM, local JNI, and optional remote adapters. Keep the core
Java API free of `java.lang.foreign` types so Android and older JVM targets can
use it without loading the FFM implementation. Decide whether an SPI is needed
once the intended host applications are known.

FFM is available as `java.lang.foreign` in JDK 22. It can call the shared C ABI;
generated low-level bindings still need a wrapper for ownership, serialization,
error translation, and native-library discovery. JDK support policy and native
access configuration remain packaging decisions.
[Oracle FFM documentation](https://docs.oracle.com/en/java/javase/22/core/foreign-function-and-memory-api.html).

JNI is the established Android Java/Kotlin-to-native boundary. Keep calls
coarse-grained, batch work where useful, and perform native inference off the UI
thread. Handle standard UTF-8 explicitly, result lifetimes, and thread-local JNI
state. Do not assume Android's runtime provides the desktop JDK FFM API.
[Android JNI guidance](https://developer.android.com/ndk/guides/jni-tips).

Android devices can run GPU compute. Vulkan compute is a native option, subject
to the device's driver, precision support, and resource limits. This provides
an execution mechanism, not an existing TurboEmbed transformer backend.
[Khronos Android compute guide](https://github.khronos.org/Vulkan-Site/tutorial/latest/Advanced_Vulkan_Compute/12_Mobile_and_Embedded_Compute/02_android_compute.html).

LiteRT's GPU delegate is another candidate for compatible converted models.
Its Interpreter API delegate requires creation and execution on the same thread,
so a dedicated worker may be required rather than merely a mutex. Model/operator
coverage and embedding parity must be tested before selecting it. NNAPI is
deprecated and should not be the default foundation for new Android work.
[LiteRT GPU documentation](https://developers.google.com/edge/litert/android/gpu),
[Android NNAPI status](https://developer.android.com/ndk/guides/neuralnetworks).

The first Android evaluation should use a named device and model, compare CPU
and GPU correctness and latency, and measure memory, sustained thermal behavior,
and battery impact. Preserve explicit device selection and report partial CPU
execution if a runtime partitions the graph. JNI alone does not port CUDA,
OpenVINO, or Metal to Android. The device enums and reserved provider vtable also
need a compatibility plan for a new provider rather than reuse of a misleading
existing device value.

## Validation performed

- `cargo test --locked --workspace`: failed with exit 101 in the counter test
  described above. Build phase reported 13.16 seconds.
- `cargo test --locked --workspace -- --test-threads=1`: exited 0; 279 harness
  passes, 0 failures, 8 ignored. Some tests return early when model files are
  absent, so these counts are not 279 executed hardware/model validations.
- Environment: Rust/Cargo 1.97.1 and protoc 25.1 on Linux. This does not validate
  the declared Rust 1.85 minimum.
- Inspected CI configuration and selected committed receipts. No GPU suites,
  Swift execution, Android execution, sanitizers, hosted CI checks, model
  downloads, or benchmarks were run. No new test or implementation code was
  written.

Local logs: `/tmp/turboembed-review-default-tests.log` and
`/tmp/turboembed-review-serial-tests.log` (temporary, not committed evidence).

Next planning decisions are the Java support baseline, the first Android
device/API target, and which model must retain equivalent behavior across
providers. The immediate technical prerequisite is a reliable native ownership
and execution contract, followed by reproducible native packaging.
