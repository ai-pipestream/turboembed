# Java binding conformance suite

The tests every in-process Java binding of `libturbo` runs, so a caller sees
the same spans, labels, probabilities, vectors and errors whichever bridge
is loaded: the Panama binding (`clients/java-panama`, Java 25 or newer) and
the JNI binding (`clients/java-jni`, Java 21 or newer and Android).

The suite is a test dependency, built in the `clients/pom.xml` reactor.
Both bindings extend `AnalyzerConformance`, `SessionConformance` and
`RuntimeConformance` and provide the actual `TurboRuntime` implementation.
There is no per-binding data adapter: calls use the shared Java API.
`Row` and `Graph` are only test fixture snapshots used to compare the
native results, not a second public application API.

The expectations cover the seeded heads in `testdata/tiny-bert-heads-bundle`
and `testdata/tiny-bert-segment-bundle`,
rule tokenization and sentence splitting, PII, graph round trips, native
error details, retained owners and closed views. Loader tests run fresh
JVM processes for automatic selection, explicit overrides and missing jars.

`mvn -f clients/pom.xml -pl java-conformance -am test` builds the API and
suite on Java 21. Add `-pl java-jni -am` to run the JNI suites or use
`-pl java-panama -am` with Java 25+ to run Panama. Build `libturbo` first.
