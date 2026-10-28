package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalysisResult;
import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboLayerKind;
import ai.pipestream.turboembed.panama.TurboSpanView;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/** OpenNLP-style sentence detector backed by {@code libturbo}. */
public final class TurboSentenceDetectorME extends TurboAdapterSupport implements SentenceDetector, AutoCloseable {

    /** Creates a sentence detector with the default English rule splitter. */
    public TurboSentenceDetectorME() {
        this(TurboAnalyzer.create(), true);
    }

    /** Creates a sentence detector using an OpenNLP {@code SentenceModel} {@code .bin} file. */
    public TurboSentenceDetectorME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", Objects.requireNonNull(modelPath, "modelPath"), null, null, null, null)),
                true);
    }

    /** Creates a detector sharing an analyzer without owning its lifecycle. */
    public TurboSentenceDetectorME(TurboAnalyzer analyzer) {
        this(analyzer, false);
    }

    private TurboSentenceDetectorME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        super(analyzer, ownsAnalyzer);
    }

    @Override
    public synchronized String[] sentDetect(CharSequence s) {
        String text = text(s);
        try (TurboAnalysisResult result = analyzer().detectSentences(text)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.SENTENCES);
            requireSingleFragmentRows(rows, "sentDetect");
            String[] sentences = new String[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                sentences[i] = rows.get(i).coveredSourceText(text);
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return sentences;
        }
    }

    @Override
    public synchronized Span[] sentPosDetect(CharSequence s) {
        String text = text(s);
        try (TurboAnalysisResult result = analyzer().detectSentences(text)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.SENTENCES);
            requireSingleFragmentRows(rows, "sentPosDetect");
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
    public synchronized void close() {
        closeAdapter();
    }

    private static String text(CharSequence s) {
        Objects.requireNonNull(s, "s");
        return s instanceof String string ? string : s.toString();
    }
}
