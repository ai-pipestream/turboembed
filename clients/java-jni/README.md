# Java JNI binding

This module binds the `libturbo` C ABI from Java 21 or newer, and from
Android, through the Java Native Interface. It is the bridge for the
runtimes the Panama binding (`clients/java-panama`) cannot serve: Android's
runtime has no `java.lang.foreign`, and this project's Panama implementation
requires Java 25. Add `ai.pipestream:turboembed-api` and
`ai.pipestream:turboembed-jni` at the same version. Import the common
`ai.pipestream.turboembed` interfaces and values; the JNI package contains
only their native implementation. The adapters now live once in
`ai.pipestream.turboembed.opennlp` in `clients/java-api`.

`TurboRuntime.load()` selects JNI on Java 21 through 24 and Android. On
Java 25+, use `-Dturbo.bridge=jni` or `TurboRuntime.load(TurboBridge.JNI)`
to choose JNI explicitly, even if Panama is installed. The API reports a
missing selected jar instead of silently choosing another bridge.

The native side is part of `libturbo` itself: `core/src/jni` exports the
`native` methods of `ai.pipestream.turboembed.jni.TurboJni` beside the C
ABI, each a thin shim over the `turbo_*` function a C caller would use. No
C glue is compiled, and one `libturbo` per platform serves both bindings.

No accuracy or speed comparison with Apache OpenNLP is claimed by this
binding. The tests run over the sealed small bundles in `testdata`, whose
seeded heads cover correctness and lifecycle, not accuracy.

## Requirements

Build the native library first, then run Maven with a Java 21 or newer JDK:

```sh
cargo build -p turbo --locked
mvn -f clients/pom.xml -pl java-jni -am test
```

No `--enable-native-access` flag is needed. Library discovery, in order:

1. `turbo.library.path` system property or `TURBO_LIBRARY_PATH`: a shared
   library file or a directory containing it. A configured missing path is
   an error.
2. `turbo.library.dir` system property or `TURBO_LIBRARY_DIR`: one or more
   path-separator-delimited directories. A configured set with no library
   is an error.
3. `target/debug` and `target/release` under the working directory or one
   of its ancestors.
4. `System.loadLibrary("turbo")`: the platform library lookup, which on
   Android is the application's native libraries for the device's ABI.

`TurboLibrary.load()` loads the library early; the first native call loads
it otherwise.

## Android

`build-android.sh` builds `libturbo.so` for the `arm64-v8a` and `x86_64`
ABIs through the NDK, at API level 26, into a `jniLibs` directory:

```sh
ANDROID_NDK_HOME=/path/to/ndk clients/java-jni/build-android.sh app/src/main/jniLibs
```

The release workflow publishes the same two libraries as
`turboembed-v<version>-android.tar.gz` (docs/release.md). The Maven jars target Java 21. The Android conformance harness compiles
the same sources at language level 17 against `android.jar` (exhaustive
enum switches would otherwise require `java.lang.MatchException`, which
that jar does not declare) and uses no platform API above Android API 26,
so an application module with `minSdk 26` and desugaring for records
consumes the API and JNI code on ART. On Android, bundles live where the
application can read them as directories, such as the application's files
directory, and are named by their absolute path.

### Conformance on an emulator or device

`android-conformance/` is a Gradle harness kept apart from the Maven build. It
packs the sealed `testdata` bundles as APK assets, copies them into the
application files directory at test start, and runs `AnalyzerConformance`,
`SessionConformance`, and the JNI bridge tests as instrumented tests. The
loader path under test is `System.loadLibrary("turbo")` from the packaged
`jniLibs`; the harness clears `turbo.library.path` and `turbo.library.dir`
so a host-JVM library cannot satisfy the load.

```sh
ANDROID_NDK_HOME=/path/to/ndk ANDROID_HOME=/path/to/sdk \
  clients/java-jni/run-android-conformance.sh
```

`TURBO_ANDROID_ABI` defaults to `x86_64,arm64-v8a`. Use `x86_64` alone for
an emulator. The opt-in CI job `.github/workflows/android-conformance.yml`
runs the same script on an `x86_64` emulator at API 26 (and 35 on labeled
pull requests) via `workflow_dispatch` or the `ci:android-emulator` label.
The host-only `:helpers:test` target checks the asset-staging layout without
talking to ART. The Gradle wrapper jar under `android-conformance/gradle/wrapper/`
is the stock Gradle 8.11.1 wrapper (`distributionSha256Sum` is set in
`gradle-wrapper.properties`).

ART results are recorded from that opt-in CI job (API 26 and API 35 on
`x86_64`), not from a host JVM stand-in. After a green emulator run, paste
the job URL and the instrumented-test summary into the pull request that
carried the harness change.

The first labeled emulator run on this harness failed compiling
`clients/java-api` against `android.jar`: `RuntimeLoader` called
`Runtime.version()`, which that jar does not declare. Selection now reads
`java.specification.version` instead so the shared source compiles on
Android and on the host.

## Primary typed API

Primary operations return `TurboAnalysisResult`, which owns one native
result reference:

```java
import ai.pipestream.turboembed.*;

try (TurboAnalyzer analyzer = TurboAnalyzer.create();
     TurboAnalysisResult result = analyzer.tokenize(text)) {
    for (TurboSpanView token : result.spans(TurboLayerKind.TOKENS)) {
        String surface = token.coveredSourceText(text);
        TurboSpanData.CoordinateRange source = token.sourceRange().orElseThrow();
        TurboSpanData.CoordinateRange view = token.viewRange().orElseThrow();
    }
}
```

Where the Panama binding reads the native span array and string pool in
place, this binding copies them into Java once, when the result is made:
the rows as one `int[]` and one `double[]`, the pool as one `byte[]`. The
graph crosses on its first use. The native result handle is kept for
`retain()` and the graph and is released by `close()`; no other native
pointer outlives the call that read it. `TurboSpanView` reads the copied
arrays and decodes one pool slice per label or value. Views stop working
when their owning result is closed, as in the Panama binding, so the two
bindings keep the same lifetime rules.

Use explicit copy-out when data must outlive the result:

```java
List<TurboSpanData> copied = analyzer.tokenizeCopied(text);
TurboSpanData one = result.copySpan(0);
byte[] graphBytes = graphResult.copyGraphBytes();
```

Methods ending in `Copied` release the native result before returning.

The Panama binding's `withSpansSegment`, `withStringPoolSegment` and
`withGraphSegment` callbacks hand out `MemorySegment`s, which this binding
cannot: they are the one part of its API without a counterpart here.

## Coordinates, rows, bundles and graphs

The rows, their coordinate spaces, the bundle rules, the explicit unknowns
and the graph round trip are the Panama binding's, documented in
`clients/java-panama/README.md`. In short: every row names a
`TurboLayerKind`, a layer id and instance, item and fragment indexes, a view
id, optional view and source ranges in UTF-8 bytes, UTF-16 code units and
Unicode scalars, an optional token range, an optional label and value, and
an optional probability. The learned layers come from the heads of the
bundle named by `TurboAnalyzer.Config.forBundle(bundleDir)`; without one,
their tasks throw `TurboException` with `TURBO_E_UNSUPPORTED_TASK`.
`inspectGraph(byte[])` decodes a graph against the source stored in it.

Text crosses to the library as UTF-8 encoded by `String.getBytes`, as the
Panama binding encodes it, so an unpaired surrogate reaches `libturbo` as
the same replacement in both.

## Conformance

`clients/java-conformance` holds the suite both bindings run: the same rows,
labels, probabilities, vectors and errors over the sealed bundles in
`testdata`. This module runs it from `JniAnalyzerConformanceTest` and
`JniSessionConformanceTest`; the Panama module runs it from its own
subclasses. A caller sees the same result whichever bridge is loaded.

## Lifetime and threading

- `TurboAnalyzer` serializes calls on one native analyzer.
- `TurboAnalyzer.retain()` creates another owner sharing that serialization.
- A `TurboAnalysisResult` remains usable after its analyzer closes.
- `TurboAnalysisResult.retain()` creates another independent result owner
  over the same copied rows.
- Closing one owner does not invalidate views belonging to a retained owner.
- `TurboSession` serializes embedding and reranking calls; the native side
  checks the result's dtype, shape and byte count before the vectors cross.
- Every `close()` is idempotent.

## OpenNLP-style adapters

The shared adapters in `java-api` use this binding's analyzer: they
return copied arrays, their `Span` names its coordinate space (UTF-16 or
token), absent probabilities are `Double.NaN`, unknown tags and lemmas are
`null` elements, and discontinuous native items are rejected rather than
collapsed. Adapter `probs()` state belongs to the most recent completed call
on that adapter.
