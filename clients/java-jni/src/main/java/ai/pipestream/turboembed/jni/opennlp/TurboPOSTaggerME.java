package ai.pipestream.turboembed.jni.opennlp;

import ai.pipestream.turboembed.jni.TurboAnalysisResult;
import ai.pipestream.turboembed.jni.TurboAnalyzer;
import ai.pipestream.turboembed.jni.TurboLayerKind;
import ai.pipestream.turboembed.jni.TurboSpanView;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * OpenNLP-style POS tagger backed only by the part-of-speech head of a bundle.
 * There is deliberately no no-bundle constructor or lexical fallback.
 */
public final class TurboPOSTaggerME extends TurboAdapterSupport implements POSTagger, AutoCloseable {

    /** Creates a POS tagger over the part-of-speech head of the bundle directory at {@code bundlePath}. */
    public TurboPOSTaggerME(Path bundlePath) {
        this(TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(bundlePath)), true);
    }

    /** Creates a POS tagger sharing an analyzer without owning its lifecycle. */
    public TurboPOSTaggerME(TurboAnalyzer analyzer) {
        this(analyzer, false);
    }

    private TurboPOSTaggerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        super(analyzer, ownsAnalyzer);
    }

    @Override
    public synchronized String[] tag(String[] sentence) {
        Objects.requireNonNull(sentence, "sentence");
        try (TurboAnalysisResult result = analyzer().posTag(sentence)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.POS);
            requireAlignedRows(rows, sentence.length, "POS");
            String[] tags = new String[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                tags[i] = rows.get(i).label().orElse(null);
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return tags;
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

    static void requireAlignedRows(List<TurboSpanView> rows, int expected, String task) {
        requireSingleFragmentRows(rows, task);
        if (rows.size() != expected) {
            throw new IllegalStateException(task + " returned " + rows.size() + " rows for " + expected + " tokens");
        }
        for (int i = 0; i < rows.size(); i++) {
            TurboSpanView row = rows.get(i);
            if (row.itemIndex() != i
                    || row.tokenRange().isEmpty()
                    || row.tokenRange().get().start() != i
                    || row.tokenRange().get().end() != i + 1L) {
                throw new IllegalStateException(task + " row " + i + " is not aligned to token " + i);
            }
        }
    }
}
