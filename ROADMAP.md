# TurboEmbed roadmap

Planning baseline: 2026-09-13, code at `9d03af9`. No implementation is authorized
by this document alone. The [library definition](docs/library-design.md) states
the intended contracts; the [review](docs/code-review-2026-09-13.md) records known
defects. This roadmap governs library direction over older server-first plans.

Current implementation and usage are maintained in the [native SDK guide](docs/native-sdk.md),
with linked Rust/Java guides and dated validation receipts. The planning baseline
below does not serve as a current feature-status list.

## Release objective

Ship an embeddable native library that applications can run directly on a GPU
machine, with control over tokens, buffers, and execution, and measured overhead
close to the underlying native implementation. Provide a convenient text API
over the same path. Preserve embedding, reranking, and shared-buffer boundaries;
make language bindings and the gRPC server optional consumers.

The first supported model family is MiniLM embeddings, with the existing MiniLM
cross-encoder as the reranking track. Broader model support is earned through
tokenizer, numerical, device, and packaging conformance. Existing catalog entries
and historical receipts do not automatically qualify a model for release.

Confirmed platform order: Intel OpenVINO on a GPU host establishes the first
release's correctness and performance baseline. NVIDIA CUDA and Apple Metal
follow as provider qualification tracks and receive common correctness fixes
throughout. The initial Intel packaging target is Linux; the exact reference
device and runtime version will be recorded before measurement.

Confirmed Java target: JDK 25+ on desktop through Panama FFM. JNI is delivered
later for Android, sharing the language-neutral native contract. Older desktop
JVM compatibility is outside the initial release scope.

## What we already have

- C contracts for embedding, reranking, and native arenas, with Rust and Swift
  callers and real CUDA, OpenVINO, and Metal implementation paths.
- Model manifests and fetching tools, reference fixtures, hardware receipts,
  device-policy checks, and a working server architecture.
- Native WordPiece and an E2E chunking helper. Both require additional contract
  work before general-purpose library use.
- A benchmark harness that measures selected existing paths, but not yet the
  matched native/ABI/language overhead experiment needed for the product claim.

The review found ownership, string, concurrency, tokenizer, option, and test
isolation defects. The ordinary workspace test run failed; a serial run passed
279 tests with 8 ignored, including conditional model skips. These are recorded
baseline observations, not current GPU certification.

## M0: make the existing contracts safe and reproducible

Land narrow fixes before new public surfaces:

1. Enforce Rust and Swift engine/result lifetime rules, including release order.
2. Repair empty-alias handling in safe Rust and embedded-NUL copying in Swift;
   define input bounds and error behavior at every language boundary.
3. Correct CUDA allocator ownership across engines; enforce Swift engine sharing
   rules and isolate creation-error storage. Define callback reentry and failure
   containment rather than relying on callers to avoid them.
4. Gate optimized tokenizer selection to compatible configurations and verify
   exact token IDs. Make unsupported pooling/normalization/truncation explicit.
5. Isolate shared test counters and restore the ordinary parallel test command.
   Update affected documentation and remove misleading standalone build commands.

Acceptance: ordinary default-feature tests pass; relevant lifetime/string tests
run under available native sanitizers; Rust negative compile tests establish
the safe ownership boundary where applicable; two-engine create/run/destroy
scenarios pass on the affected GPU providers. Record hardware-unverified fixes
as such until exercised. Use the reference model tests without changing goldens
to accommodate failures.

Each numbered item is a reviewable change group, split further by language or
provider when needed. Do not combine them into an unrelated kernel rewrite.

## M1: define and prove the prepared execution path

After M0's core ownership work, agree the versioned ABI extension and implement
the smallest useful direct path:

- Device discovery and explicit selection, including device identity and
  provider/model capabilities.
- Prepared model and execution workspace with bounded shape and memory limits.
- Token/mask input views and caller-reusable outputs, including a device-result
  option. Keep text convenience on the same execution implementation.
- Explicit failure for unsupported placement, data type, shape, or options.
- An optimized direct-native reference harness and ABI comparison, following the
  [performance acceptance plan](docs/library-design.md#native-performance-acceptance).

Use direct OpenVINO execution on the selected Intel GPU as the initial native
reference. Verify resource context compatibility and actual transfers using the
installed runtime; a host pointer wrapped in a tensor is not proof of remote
device binding. The baseline runs in process without an inference server.

Acceptance: a native application supplies valid tokens, performs repeated GPU
inference, and consumes the output on the GPU without mandatory readback. A
second example requests a host result. Numerical parity holds, lifetimes are
enforced, and the measured overhead meets the predeclared budget. Small and
mixed-length requests must be represented; a large-batch throughput result alone
does not establish low latency.

Start synchronously with owned resources. External buffer/queue import and
asynchronous completion follow as separately advertised capabilities. Do not
reuse a host-pointer result field to expose device memory or alter ABI v1 structs.

## M2: ship the first native GPU SDK

Package the Intel OpenVINO provider as a reproducible shared library and headers,
with a native link example and Rust bindings. Preserve an explicit CPU reference
path for correctness and environments that deliberately select it. Ship reranking
as an optional surface when it passes the same lifecycle and packaging gates.

- Build artifacts outside a particular developer's directory; define exported
  symbols, loader paths, runtime versions, and redistribution requirements.
- Model loading takes an explicit bundle path. Catalog aliases remain convenient
  names, with no dependency on a developer's external model-cache path.
- Reuse the existing manifest/fetch tools for explicit provisioning. A bundle
  records model/tokenizer revisions, hashes, conversion provenance, precision,
  prefixes, pooling, normalization, limits, and license metadata.
- Verify complete bundles before exposing them to inference; interrupted or
  mismatched downloads cannot become a loadable model. Run offline after setup.
- Provide one maintained getting-started page and a support list naming tested
  models, devices, runtime versions, and capabilities.

Acceptance: a clean consumer project installs the artifact, verifies and loads
the model, embeds text and prepared tokens without a server or source checkout,
and reproduces correctness and bounded performance checks. Dependencies resolve
from the documented package, not the author's environment.

This is the first native preview. External async interop, all Java bindings,
Android, and OpenNLP analysis are not prerequisites for this checkpoint.

## M3: deliver desktop Java on JDK 25+ through FFM

After the native ABI and packaging are proven, add the Java API and its
in-process Panama FFM adapter for JDK 25+. Preserve a boundary that the later
Android JNI adapter can implement. Measure FFM against the native ABI; JNI
implementation and measurement belong to M5.

- Deterministic ownership and close behavior, typed errors, UTF-8 correctness,
  and safe concurrent use under the declared execution model.
- Text convenience plus reusable native-buffer/prepared-input calls; batch
  crossings instead of per-token or per-value calls.
- Adapter selection at initialization, with no mandatory runtime download and
  no FFM classes loaded on Android or an unsupported desktop JVM.
- Reproducible generated FFM declarations where useful, with header/layout
  conformance. End users do not need a binding generator to run the library.
- Native artifacts selected by platform and ABI; missing accelerator support
  remains an explicit error. A remote adapter is optional and separately named.

Acceptance: identical native/FFM contract cases pass, including empty text,
embedded NUL, Unicode, invalid options, output aliasing, close during use, and
multiple engines. Publish native, prepared Java, and ordinary text Java timings
separately. Demonstrate an ordinary Java application on the GPU host.

An OpenNLP integration can consume this API later without requiring a dependency
on a particular Java inference framework. Do not add it to OpenNLP in this phase.

## M4: qualify NVIDIA and Apple; extend model coverage

Bring the existing NVIDIA and Apple providers through M0-M2's contracts, examples,
packaging, and matched native measurements. Adapt the language surface appropriate
to each platform, including Java where selected. A provider can pass synchronous
inference while still reporting external-buffer or async interop unsupported.

For NVIDIA, prove CUDA allocator isolation and device-buffer integration under
multiple live engines. For Apple, prove Metal resource ownership and Swift
concurrency without relying on unchecked sharing declarations. Compare each
provider's overhead against its own direct-native implementation, while using
the Intel-first model contract for cross-provider correctness.

Expand with one deliberately different model contract at a time: CLS pooling,
different tokenization, then multilingual inputs. Match the actual checkpoint
and precision across providers; returning a different model with the same vector
dimension is not portability. Precision/quantization changes need separate
accuracy and performance evidence.

Acceptance: each published provider has its own passing conformance and native
overhead results. A cross-platform SDK milestone requires the advertised common
MiniLM surface on NVIDIA, Intel, and Apple. Do not require every catalog alias
on every provider or hide differences behind automatic model substitution.

## M5: add an Android JNI surface and qualify one mobile GPU backend

Choose a physical Android device, minimum API level, and supported ABI before
implementation. Start with arm64 as a proposal, not an existing support claim.
Cross-compile and package an AAR with native artifacts, model provisioning, and
a Kotlin/Java example. Implement JNI against the native contract and the Java
API boundary established in M3, without loading desktop FFM classes on Android.
Port the same ownership, encoding, option, and device-policy conformance cases
and measure native-versus-JNI overhead on the phone.

Evaluate one viable backend on that device before building a broader platform
layer. Vulkan compute offers native GPU execution but requires model kernels
or a compatible inference implementation; LiteRT is a candidate when model
conversion and operator coverage fit. Benchmark the same model against explicit
CPU execution and check actual GPU work, partial CPU execution, initialization,
memory use, sustained latency, and thermal/battery effects.

Acceptance: inference runs on the named phone GPU, the UI stays responsive,
resource/thread lifetimes are correct, and the model meets parity gates.
Missing GPU capability gives the documented error unless the caller explicitly
selected an allowed CPU policy. An emulator or successful NDK compilation is
not mobile GPU proof. Keep NNAPI outside the default plan because it is deprecated.
[Android JNI](https://developer.android.com/ndk/guides/jni-tips),
[Vulkan compute](https://github.khronos.org/Vulkan-Site/tutorial/latest/Advanced_Vulkan_Compute/12_Mobile_and_Embedded_Compute/02_android_compute.html),
[LiteRT GPU](https://developers.google.com/edge/litert/android/gpu),
[NNAPI status](https://developer.android.com/ndk/guides/neuralnetworks).

## M6: optional chunking, OpenNLP integration, and analysis acceleration

A basic native chunker can land after M1 and ship as an optional utility without
waiting for Android. Require model token budgets, source offsets, explicit
overlap, and deterministic behavior; avoid promising linguistic analysis from
the E2E helper. Callers can always provide their own chunks or prepared tokens.

After M3, design the optional OpenNLP embedding/reranking provider around the
Java API. Keep the base OpenNLP dependency policy intact; downloads and native
provider dependencies belong to opt-in integration/provisioning modules. The
planned roughly 50 analysis features retain their own roadmap and acceptance
criteria. They do not all become TurboEmbed work or native ports automatically.

GPU tokenization, segmentation, and model-based native/GPU analysis remain
unsupported extension points until selected for a measured experiment. Promote
one only when it preserves tokenizer/annotation semantics and improves the
complete pipeline after transfer and scheduling costs. Separate Java analysis,
native algorithm ports, and model inference in the integration design.

Acceptance: OpenNLP runs without this integration installed; an opt-in provider
executes the selected embedding model in process; model provenance and source
offsets survive the pipeline. Broader analysis acceleration is a later gate.

## Sequence and definition of landed

The main sequence is M0, M1, M2, then M3. Common fixes for all providers belong in
M0; provider qualification in M4 can proceed after the common contracts stabilize.
M5 adds JNI after the native contract and Java API work; it does not depend on
completion of OpenNLP.
M6's chunker and OpenNLP adapter have the separate dependencies described above.
There are no calendar estimates until the first hardware baseline and ABI
decisions are complete.

A milestone is landed when its scoped changes are reviewed and merged, required
checks pass, and its acceptance example works. An SDK release additionally needs
published artifacts and a clean consumer install test. Local tests, hosted CI,
merge, artifact publication, and device validation are distinct recorded states.
Do not mark an experimental or stubbed capability as supported.

Before M1 implementation, record the exact Intel GPU/runtime and settle Java
package identity, ABI extension strategy, and buffer-owner/close policy. Android minimums
and the OpenNLP SPI mapping can wait for their respective phases. Performance
budgets are calibrated once in the bounded pilot and then held fixed for the
implementation being evaluated.

Existing server work such as TLS/mTLS, model ACLs, protocol adapters, richer
stream metadata, and generation backends remains a separate backlog. AMD,
Windows, additional mobile vendors, multi-GPU scheduling, and arbitrary analysis
plugins are expansion candidates requiring explicit scope and validation.
