# Java Panama binding

This module binds the `libturbo` C ABI from Java 25 or newer through the
Panama Foreign Function and Memory API. It contains:

- `ai.pipestream.turboembed.panama`: the typed primary API.
- `ai.pipestream.turboembed.opennlp`: OpenNLP-style adapters in this
  project's namespace. These are not Apache OpenNLP classes and are not
  source- or binary-compatible drop-in replacements for `opennlp.tools`.

No accuracy or speed comparison with Apache OpenNLP is claimed by this
binding. The tests run over the sealed small bundles in `testdata`, whose
seeded heads cover correctness and lifecycle, not accuracy.

## Requirements

Build the native library first, then run Maven with a Java 25 or newer JDK:

```sh
cargo build -p turbo --locked
mvn -f clients/java-panama/pom.xml test
```

Applications must run with:

```text
--enable-native-access=ALL-UNNAMED
```

Library discovery, in order:

1. `turbo.library.path` system property or `TURBO_LIBRARY_PATH`: a shared
   library file or a directory containing it. A configured missing path is
   an error.
2. `turbo.library.dir` system property or `TURBO_LIBRARY_DIR`: one or more
   path-separator-delimited directories. A configured set with no library
   is an error.
3. `target/debug` and `target/release` under the working directory or one
   of its ancestors.
4. The platform library lookup for the system name of `libturbo`.

## Primary typed API

Primary operations return `TurboAnalysisResult`, which owns one native
result reference:

```java
try (TurboAnalyzer analyzer = TurboAnalyzer.create();
     TurboAnalysisResult result = analyzer.tokenize(text)) {
    for (TurboSpanView token : result.spans(TurboLayerKind.TOKENS)) {
        String surface = token.coveredSourceText(text);
        TurboSpanData.CoordinateRange source = token.sourceRange().orElseThrow();
        TurboSpanData.CoordinateRange view = token.viewRange().orElseThrow();
    }
}
```

`TurboSpanView` reads the native span array in place. It does not copy the
span array, the string pool, or the graph. A label or value accessor decodes
only that row's pool slice. Views stop working when their owning result is
closed.

Use explicit copy-out when data must outlive the result:

```java
List<TurboSpanData> copied = analyzer.tokenizeCopied(text);
TurboSpanData one = result.copySpan(0);
byte[] graphBytes = graphResult.copyGraphBytes();
```

Methods ending in `Copied` release the native result before returning.

## Coordinates and rows

Every row names:

- a `TurboLayerKind`;
- the full layer id and instance;
- item and fragment indexes;
- the native view id;
- an optional view range in that view's coordinates;
- an optional checked source range;
- an optional token range;
- an optional label and normalized value;
- an optional probability.

View and source ranges each carry UTF-8 byte, UTF-16 code-unit, and Unicode
scalar bounds. They are never interchanged. Covered sentence and token text
is read through `coveredSourceText` or `coveredViewText`; it is not copied
into the label field.

Token-only POS, lemma, morphology, and token-sequence NER rows have token
ranges and no text ranges. Dependency rows are aligned to tokens: the
relation is the label and the head token is read through `dependencyHead()`,
which is empty for the root of a sentence. PII rows carry their kind as the
label, their normalized form as the value, and a `TurboPiiConfidence`
evidence classification. PII classification is not represented as a
probability.

## Bundles and explicit unknowns

The default analyzer contains the rule sentence splitter, rule tokenizer,
and the separate PII task. It does not contain hidden POS, lemma,
morphology, or NER fallbacks. The learned layers come from the heads of a
bundle directory: its encoder runs once per sentence and every head reads
that pass, so the part-of-speech, lemma, morphology, dependency and entity
layers of `analyzeGraph` cost one encoder pass together. Without a bundle,
or with a bundle that has no head for a layer, that layer's tasks throw
`TurboException` with `TURBO_E_UNSUPPORTED_TASK` naming what is missing.

Name the bundle explicitly:

```java
TurboAnalyzer.Config config = TurboAnalyzer.Config.forBundle(bundleDir);
TurboAnalyzer.Config french = new TurboAnalyzer.Config("fr", bundleDir);
```

The five per-component model path fields of the native descriptor are
retired. The binding never writes them; a program writing the raw layout
gets `TURBO_E_INVALID_ARGUMENT` naming the field for a non-empty value.

NER uses only the bundle's entity head and fresh context for each
sequence. No dictionary, pattern, or PII recognizer output is merged into
NER. `detectPii` remains a separate operation and layer.

`lemmatize` takes the tags the bundle's lemmatizer head reads: they must
be labels of that head's tag head, and a lemmatizer reading no tags
ignores them. A row with no label is an explicit miss. It is never
replaced by a lowercased token. Rule sentence and token rows have no
probability; the binding does not invent one.

## Graphs

`analyzeGraph` returns typed rows plus the serialized binary graph in the
same `TurboAnalysisResult`. Use the read-only `withSpansSegment`,
`withStringPoolSegment`, and `withGraphSegment` callbacks to inspect native
buffers without copying them, or `copyGraphBytes` for explicit copy-out.

`inspectGraph(byte[])` does not accept caller-supplied text. It validates
and decodes against the source and views stored in the graph itself, then
reports both view coordinates and checked source projections.

## Lifetime and threading

- `TurboAnalyzer` serializes calls on one native analyzer.
- `TurboAnalyzer.retain()` creates another owner sharing that serialization.
- A `TurboAnalysisResult` remains usable after its analyzer closes.
- `TurboAnalysisResult.retain()` creates another independent result owner.
- Closing one owner does not invalidate views belonging to a retained owner.
- `TurboSession` serializes embedding and reranking calls and validates the
  native result shape and byte count before constructing Java arrays.

## OpenNLP-style adapters

The adapters return copied arrays because their interfaces require arrays.
Their `Span` type explicitly names either UTF-16 or token coordinate space:

- token spans cannot be used to slice text and have no byte offsets;
- UTF-16 spans carry corresponding source byte and codepoint offsets;
- absent probabilities are `Double.NaN`, not zero;
- unknown POS tags and lemmas are `null` array elements;
- discontinuous native items are rejected rather than collapsed into a
  bounding span.

Adapter `probs()` state belongs to the most recent completed call on that
adapter. Use one adapter per thread, or use the primary typed result API,
when probability association matters under concurrency.
