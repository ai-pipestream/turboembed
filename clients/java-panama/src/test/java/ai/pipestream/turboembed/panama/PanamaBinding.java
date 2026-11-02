package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.conformance.Binding;
import ai.pipestream.turboembed.conformance.Graph;
import ai.pipestream.turboembed.conformance.Row;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** This binding under the shared conformance suite. */
final class PanamaBinding implements Binding {
    static final PanamaBinding INSTANCE = new PanamaBinding();

    private PanamaBinding() {}

    @Override
    public String name() {
        return "panama";
    }

    @Override
    public ai.pipestream.turboembed.conformance.Analyzer openAnalyzer(String language, Path bundlePath) {
        TurboAnalyzer analyzer = TurboAnalyzer.create(new TurboAnalyzer.Config(language, bundlePath));
        return new ai.pipestream.turboembed.conformance.Analyzer() {
            @Override
            public List<Row> detectSentences(String text) {
                return rows(analyzer.detectSentencesCopied(text));
            }

            @Override
            public List<Row> tokenize(String text) {
                return rows(analyzer.tokenizeCopied(text));
            }

            @Override
            public List<Row> posTag(String[] tokens) {
                return rows(analyzer.posTagCopied(tokens));
            }

            @Override
            public List<Row> lemmatize(String[] tokens, String[] tags) {
                return rows(analyzer.lemmatizeCopied(tokens, tags));
            }

            @Override
            public List<Row> findTokens(String[] tokens) {
                return rows(analyzer.findTokensCopied(tokens));
            }

            @Override
            public List<Row> findText(String text) {
                return rows(analyzer.findTextCopied(text));
            }

            @Override
            public List<Row> detectPii(String text) {
                return rows(analyzer.detectPiiCopied(text));
            }

            @Override
            public Graph analyzeGraph(String text) {
                try (TurboAnalysisResult result = analyzer.analyzeGraph(text)) {
                    return new Graph(rows(result.copySpans()), result.copyGraphBytes());
                }
            }

            @Override
            public List<Row> inspectGraph(byte[] graph) {
                try (TurboAnalysisResult result = analyzer.inspectGraph(graph)) {
                    return rows(result.copySpans());
                }
            }

            @Override
            public void close() {
                analyzer.close();
            }
        };
    }

    @Override
    public ai.pipestream.turboembed.conformance.Session openSession(Path bundleDir) {
        TurboSession session = TurboSession.open(bundleDir);
        return new ai.pipestream.turboembed.conformance.Session() {
            @Override
            public float[][] embed(List<String> texts) {
                return session.embed(texts);
            }

            @Override
            public float[] rerank(String query, List<String> passages) {
                return session.rerank(query, passages);
            }

            @Override
            public void close() {
                session.close();
            }
        };
    }

    @Override
    public int statusCode(Throwable failure) {
        return failure instanceof TurboException e ? e.code() : -1;
    }

    @Override
    public String statusName(Throwable failure) {
        return failure instanceof TurboException e ? e.statusName() : null;
    }

    @Override
    public int statusField(Throwable failure) {
        return failure instanceof TurboException e ? e.field() : -1;
    }

    static List<Row> rows(List<TurboSpanData> spans) {
        List<Row> out = new ArrayList<>(spans.size());
        for (TurboSpanData span : spans) {
            out.add(new Row(
                    span.layerKind().nativeCode(),
                    span.itemIndex(),
                    span.fragmentIndex(),
                    span.viewId(),
                    span.classification(),
                    span.viewRange().map(PanamaBinding::range).orElse(null),
                    span.sourceRange().map(PanamaBinding::range).orElse(null),
                    span.tokenRange().map(tokens -> new Row.Tokens(tokens.start(), tokens.end())).orElse(null),
                    span.layer(),
                    span.instance(),
                    span.label().orElse(null),
                    span.value().orElse(null),
                    span.probability().isPresent() ? span.probability().getAsDouble() : null));
        }
        return out;
    }

    private static Row.Range range(TurboSpanData.CoordinateRange range) {
        return new Row.Range(
                range.byteStart(),
                range.byteEnd(),
                range.utf16Start(),
                range.utf16End(),
                range.codepointStart(),
                range.codepointEnd());
    }
}
