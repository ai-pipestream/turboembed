package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalysisResult;
import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboLayerKind;
import ai.pipestream.turboembed.panama.TurboSpanView;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * OpenNLP-style lemmatizer backed only by the lemmatizer head of a bundle.
 * A miss returns {@code null}; there is deliberately no no-bundle constructor or fallback form.
 */
public final class TurboLemmatizerME extends TurboAdapterSupport implements Lemmatizer, AutoCloseable {

    /** Creates a lemmatizer over the lemmatizer head of the bundle directory at {@code bundlePath}. */
    public TurboLemmatizerME(Path bundlePath) {
        this(TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(bundlePath)), true);
    }

    /** Creates a lemmatizer sharing an analyzer without owning its lifecycle. */
    public TurboLemmatizerME(TurboAnalyzer analyzer) {
        this(analyzer, false);
    }

    private TurboLemmatizerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        super(analyzer, ownsAnalyzer);
    }

    @Override
    public synchronized String[] lemmatize(String[] toks, String[] tags) {
        Objects.requireNonNull(toks, "toks");
        Objects.requireNonNull(tags, "tags");
        try (TurboAnalysisResult result = analyzer().lemmatize(toks, tags)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.LEMMAS);
            TurboPOSTaggerME.requireAlignedRows(rows, toks.length, "lemmatizer");
            String[] lemmas = new String[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                lemmas[i] = rows.get(i).label().orElse(null);
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return lemmas;
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
}
