# Prepared native execution contract

Implementation direction for the first Intel SDK, 2026-09-14. This document
records the extension and ownership choices before new public symbols are
implemented. The existing text ABI v1 remains compatible. Support is established
by the roadmap's runnable examples and measurements, not by this design alone.

## ABI and resource ownership

Add a separate `turboembed_prepared.h` extension with versioned symbols and
opaque context, model, input, execution, and result handles. Name the initial
extension symbols `turboembed_prepared_v1_*`; its version is independent of the
existing text ABI version. Do not change
`turboembed.h` structs or repurpose its host result pointer as device memory.
Descriptors start with explicit byte size and extension version. The first
implementation accepts only its supported descriptor versions and rejects
unknown options or reserved fields. Integer-valued fields use fixed-width
integers; malformed enum representations must not invoke C++ undefined behavior.

Contexts select an explicit provider/device and report its resolved identity.
AUTO retains the GPU-only policy. A model loads from an explicit bundle path;
it records tokenizer identity, pooling, normalization, precision, dimensions,
and supported shapes. The first prepared shape is fixed when an execution slot
is created. An inference call cannot unexpectedly compile another shape.

Native handles retain their dependencies. Releasing a context or model handle
drops that reference; dependent resources remain usable until their last owner
is released. An execution slot owns its mutable inference request and workspace.
Different slots permit concurrency. Calls on one slot serialize or fail while
busy; they must never race its tensor bindings or buffers.

Inputs can be populated once and reused. The first version owns its allocations;
importing foreign queues/buffers and asynchronous submission remain separate
capabilities. Input IDs, masks, and model-required type IDs carry shape, data
type, capacity, and context identity. Validate ranges and arithmetic before
accessing pointers or narrowing to provider dimensions. Token identity remains
the prepared-input caller's responsibility; text calls use the validated native
model tokenizer over the same execution implementation.

A result leases an output buffer. Re-executing a slot must not overwrite a live
result: the initial implementation rejects that reuse until the result is
released. Preallocate the lease descriptor and native bookkeeping with the
slot so repeated execution reuses them as well as the tensor buffers.
Reusable caller-owned result storage can be a separately advertised
capability. Host copying is explicit. Error outputs are cleared, exceptions are
contained, and error text is copied into caller-provided storage rather than
returning mutable process-wide error buffers.

## First Intel implementation

Reference host: Intel Battlemage G31 with OpenVINO 2026.3.1 on Linux x86_64,
as recorded in the [hardware receipt](intel-native-baseline-2026-09-14.md).
Use OpenVINO's OpenCL remote-context API and actual OpenCL buffers. TurboEmbed
owns and retains the context, in-order queue, and buffers; OpenVINO wraps them
as remote tensors. Host-accessible Level Zero memory wrapped as an ordinary
`ov::Tensor` does not satisfy this path.

Transform MiniLM's graph before compilation to perform masked mean pooling and
L2 normalization on the GPU. Clamp the token-count divisor and match the
existing `max(sqrt(sum(x*x)), 1e-12)` normalization denominator. Bind the final
`[batch, dimension]` output to a remote buffer. Hidden state is not a required
host output.

Synchronous inference returns after provider work completes. An explicitly
queried OpenCL view identifies the buffer and its context; it is a borrowed
resource valid while the result lease is held. A following GPU operation must
complete before the lease is released. The first interoperability example uses
the library's shared in-order queue and an explicit completion boundary. Raw
OpenCL handles are resource identifiers, never host-addressable data pointers.
A host result request performs an explicit readback and completion wait.

The direct OpenVINO reference uses equivalent graph operations, fixed shapes,
reused resources, and synchronization. Establish numerical parity before
calibrating the roadmap's performance budgets. Record host/device transfer
bytes and allocation scopes. A remote buffer alone does not prove the provider
performs no internal copies.

## Packaging and Java

The native SDK must export an intentional symbol list and work in a clean
consumer project. Bundle verification and dependency packaging are required
before calling the SDK deliverable complete. Provisioning is explicit; inference
never downloads models. Native runtime, tokenizer data, and model licenses and
versions accompany the artifacts.

Use `ai.pipestream.turboembed` as the Java package and Maven group identity,
consistent with the repository organization. Target JDK 25+ through FFM.
Bindings retain native owners, enforce deterministic close and UTF-8 semantics,
and batch native calls. Native-buffer access must share the native resource's
lifetime. Android JNI stays outside this first desktop artifact and cannot load
FFM classes. OpenNLP integration remains optional and later.
