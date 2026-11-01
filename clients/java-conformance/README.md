# Java binding conformance suite

The tests every in-process Java binding of `libturbo` runs, so a caller sees
the same spans, labels, probabilities, vectors and errors whichever bridge
is loaded: the Panama binding (`clients/java-panama`, Java 25 or newer) and
the JNI binding (`clients/java-jni`, Java 21 or newer and Android).

The suite is source, not a jar: each binding's `pom.xml` adds
`src/main/java` here as a test source root, and a binding runs the suite
by extending `AnalyzerConformance` and `SessionConformance` and returning
its `Binding`, a small adapter from its own analyzer, session and exception
types to the neutral `Analyzer`, `Session` and `Row` here. The expectations
are exact: the labels and probabilities the seeded heads of
`testdata/tiny-bert-heads-bundle` produce, the rows of the rule splitter and
tokenizer, the PII layer, the graph round trip, and the native status code,
name and field of each refusal.

`mvn -f clients/java-conformance/pom.xml compile` checks the suite compiles
on Java 21 by itself; the bindings' own `mvn test` runs it.
