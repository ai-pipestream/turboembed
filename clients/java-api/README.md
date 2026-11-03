# Shared in-process Java API

`ai.pipestream:turboembed-api` is a pure Java 21+ jar. It contains the
analyzer, result, span-view and session contracts; immutable spans and
coordinate ranges; typed layer and PII values; native error details; and
the adapters in `ai.pipestream.turboembed.opennlp`. It has no native code
or production dependency on either bridge. The adapters are ours, not
Apache OpenNLP classes or model readers.

Add the API and one matching-version bridge:

- `ai.pipestream:turboembed-jni`: Java 21+ or Android with the documented
  Java toolchain/desugaring setup in `clients/java-jni/README.md`.
- `ai.pipestream:turboembed-panama`: Java 25+ and
  `--enable-native-access=ALL-UNNAMED`.

Both call the same platform `libturbo`. Bundle/model files are separate.
The generated RPC artifact `turboembed-grpc` is a different client.

## Select a runtime

`TurboRuntime.load()` reads `turbo.bridge` (`auto`, `jni`, `panama`,
case-insensitive). With `auto`, Java 25+ with `java.lang.foreign` selects
Panama; older supported Java runtimes and Android select JNI. Only the
chosen implementation class is loaded. A missing selected jar or an
incompatible explicit request fails with a diagnostic, without silent
fallback. A Java 25 application with only the JNI jar must choose `jni`.

`TurboRuntime.load(TurboBridge.JNI)` selects JNI explicitly regardless of
the property. A factory does not load the native library until an analyzer
or session is opened. Native library discovery belongs to the bridge and
is documented in its README.

```java
import ai.pipestream.turboembed.*;
import ai.pipestream.turboembed.opennlp.TurboTokenizerME;

TurboRuntime runtime = TurboRuntime.load();
try (TurboAnalyzer analyzer = runtime.createAnalyzer(TurboAnalyzer.Config.defaults());
     TurboAnalysisResult result = analyzer.tokenize("Café 🚀!")) {
    for (TurboSpanView row : result.spans(TurboLayerKind.TOKENS)) {
        System.out.println(row.coveredSourceText("Café 🚀!"));
    }
    TurboSpanData copied = result.copySpan(0);
    // copied survives result.close(); a row view does not.
}
```

`TurboAnalyzer.create()` and `TurboSession.open(bundle)` use the default
loader. To choose a bridge per adapter, open an analyzer through that
runtime and pass it to the adapter constructor. The adapter does not own
an analyzer supplied by its caller.

## Ownership and coordinates

Each analyzer, result and session is closeable. Retaining an analyzer or
result creates an independent owner; closing one does not close the
other. Calls sharing a native analyzer are serialized. The common API
does not add result copies: Panama reads native rows and strings on
access; JNI copies them once into Java arrays when constructing a result.
Explicit `copy*`/`*Copied` methods return independent Java values.

Every span names its source/view/token coordinate spaces. Text ranges
carry byte, UTF-16 and Unicode scalar offsets; no conversion or joined
text is introduced by this module. Native errors retain their status
code, symbolic name, field and message in the shared `TurboException`.
Panama's `withSpansSegment`, `withStringPoolSegment` and `withGraphSegment`
callbacks remain on its concrete result type, outside this API.

## Build and migration

```sh
cargo build --locked -p turbo
# Java 21+: API tests, conformance, JNI tests
mvn -f clients/pom.xml -pl java-jni -am test
# Java 25+: all four modules and both native suites
mvn -f clients/pom.xml test
```

Use `-Dturbo.library.path=...` for a library outside the standard discovery
locations. `mvn -f clients/pom.xml -pl java-api -am test` needs no native
library or bridge. On Java 21 the reactor does not load the Panama module.

Replace bridge-local imports of `TurboSpanData`, `TurboLayerKind`,
`TurboPiiConfidence` and `TurboException` with the shared package. Replace
`ai.pipestream.turboembed.jni.opennlp` with
`ai.pipestream.turboembed.opennlp`. Concrete analyzers/results still exist
in bridge packages and implement the common interfaces; their `Config`
member type is the shared `TurboAnalyzer.Config`. Recompile callers after
this package migration. The old JNI exception name remains a deprecated
subclass solely because existing native entry points construct it.
