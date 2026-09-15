# Java on JDK 25

The `turboembed-ffm` artifact calls the [installed Intel SDK](native-sdk.md)
in process through Panama FFM. `turboembed-api` contains the common Java
interfaces with no FFM or vendor-runtime dependency. Select the FFM adapter
explicitly at initialization. Android JNI is a later adapter; this provider
currently supports Linux x86_64 with JDK 25+.

Build the two jars with JDK 25 and Maven:

```bash
mvn -f java/pom.xml --batch-mode verify
```

This compiles the common API with Java 17 bytecode and the FFM provider with
Java 25 bytecode. It does not claim an older-JVM runtime implementation.
The layout test needs a C11 compiler (`cc`) and checks every descriptor against
the canonical header. GPU tests are skipped unless the SDK environment variable
is set; ordinary builds do not download models or start inference.

The jars are development artifacts under each module's `target/`, not published
Maven coordinates. The FFM provider depends only on the common API at runtime.
Supply the separately installed SDK and a verified model bundle.

```java
try (var runtime = FfmTurboEmbed.open(Path.of("/path/to/sdk"));
     var context = runtime.context(Device.OPENVINO_GPU, 0);
     var model = context.loadModel(Path.of("/path/to/minilm-bundle"));
     var slot = model.slot(1, 32)) {
    var output = FloatBuffer.allocate(model.info().dimension());
    slot.writeText("hello world");
    try (var result = slot.execute()) {
        result.readInto(output);
    }
}
```

Imports are from `ai.pipestream.turboembed`,
`ai.pipestream.turboembed.ffm.FfmTurboEmbed`, `java.nio.FloatBuffer`, and
`java.nio.file.Path`. Run with `--enable-native-access=ALL-UNNAMED` when using
the classpath. The [standalone example](../java/examples/README.md) gives complete
compile/run commands and checks prepared-token/text agreement.

## Ownership and buffers

Close resources explicitly, normally with try-with-resources. Contexts and
models serialize their own handle operations and may be shared across threads.
They retain the loaded native library independently; their children continue
working after parent close. Closed handles reject new operations.

Each execution slot and its results belong to the thread that created the slot.
Use one slot per worker. Its editable native `IntBuffer` views enforce thread
and lifetime checks even after they escape the slot object. Release a result
before reusing or closing its slot; early slot close is rejected. An old result
stays invalid when the native slot lends its output to a new result.

`inputs()` exposes three fixed row-major i32 staging buffers. Fill their complete
contents, including padding and masks, then call `upload()`. Upload ignores the
views' positions and limits and copies the fixed shape. The native buffers can
then serve repeated execution. `writeText` instead encodes Java strings to UTF-8
and invokes native tokenization; it does not populate the editable Java staging.
Empty strings and embedded NUL are supported. Unpaired UTF-16 surrogates are
rejected; failed writes invalidate previously prepared inputs.

`readInto` writes `batch * dimension` floats at the destination's current
position and advances that position. Insufficient or read-only destinations are
rejected. Aligned, native-order direct buffers receive the explicit native
readback directly. Heap, unaligned, or opposite-byte-order buffers use reusable
native staging followed by a Java bulk copy. Execution alone does not read the
GPU result back to the host.

`openCl()` exposes borrowed native resource identifiers while the result is
open. Use the supplied context and in-order queue, treat the output buffer as
read-only, and never release the borrowed handles. Result close waits for that
queue. Raw identifiers must not escape their lease into later native work.
External queue imports and asynchronous submission are unsupported.

The prepared path reuses input/output storage but creates a Java result wrapper
and FFM address objects. Text encoding also allocates. Native runtime allocations
and transfers are separate from Java allocations; the
[binding pilot](../java/benchmarks/README.md) measures these boundaries separately.

## Run the hardware contracts

On the Intel GPU host:

```bash
TURBOEMBED_PREPARED_SDK=/path/to/sdk \
TURBOEMBED_PREPARED_BUNDLE=/path/to/minilm-bundle \
  mvn -f java/pom.xml --batch-mode verify
```

This requires both Intel GPU and explicit CPU execution. It checks layouts,
model metadata, numerical parity, prepared/text agreement, invalid inputs,
Unicode, buffer bounds/order/alignment, concurrent slots and close/thread
ownership. Missing GPU capability fails the test; it does not fall back to CPU.
The [validation receipt](intel-bindings-2026-09-14.md) records actual runs and
performance limits.
