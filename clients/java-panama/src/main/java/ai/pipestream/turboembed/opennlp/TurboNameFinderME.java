package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalysisResult;
import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboLayerKind;
import ai.pipestream.turboembed.panama.TurboSpanView;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * OpenNLP-style name finder backed only by a configured statistical model.
 * PII is a separate {@link TurboAnalyzer#detectPii(String)} task and is never merged here.
 * There is deliberately no no-model constructor or recognizer fallback.
 */
public final class TurboNameFinderME extends TurboAdapterSupport implements TokenNameFinder, AutoCloseable {

    /** Creates a name finder using an OpenNLP {@code TokenNameFinderModel} {@code .bin} file. */
    public TurboNameFinderME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", null, null, null, null, Objects.requireNonNull(modelPath, "modelPath"))),
                true);
    }

    /** Creates a name finder sharing an analyzer without owning its lifecycle. */
    public TurboNameFinderME(TurboAnalyzer analyzer) {
        this(analyzer, false);
    }

    private TurboNameFinderME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        super(analyzer, ownsAnalyzer);
    }

    @Override
    public synchronized Span[] find(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        try (TurboAnalysisResult result = analyzer().findTokens(tokens)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.ENTITIES);
            requireSingleFragmentRows(rows, "find");
            Span[] spans = new Span[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                spans[i] = Span.fromToken(rows.get(i));
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return spans;
        }
    }

    /** Finds named entities directly in text and returns source UTF-16 spans. */
    public synchronized Span[] findInText(String text) {
        Objects.requireNonNull(text, "text");
        try (TurboAnalysisResult result = analyzer().findText(text)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.ENTITIES);
            requireSingleFragmentRows(rows, "findInText");
            Span[] spans = new Span[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                spans[i] = Span.fromSource(rows.get(i));
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return spans;
        }
    }

    /** Returns probabilities from the most recent call on this adapter; absent values are NaN. */
    public synchronized double[] probs() {
        return copyProbs();
    }

    @Override
    public synchronized void clearAdaptiveData() {
        analyzer();
        // Native calls construct fresh model context and retain no adaptive data.
    }

    @Override
    public synchronized void close() {
        closeAdapter();
    }
}
