package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboSpanView;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Shared lifecycle and last-call probability state for the OpenNLP-style adapters.
 *
 * <p>Subclass operations are synchronized on the adapter. That makes the native call and the
 * update of {@code probs()} atomic with respect to other calls on the same adapter. It does not
 * make a caller's separate operation-then-{@code probs()} pair atomic across threads; use one
 * adapter per thread or the typed {@link TurboAnalyzer} result API for concurrent work.
 */
abstract class TurboAdapterSupport {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private boolean closed;
    private double[] lastProbs = new double[0];

    TurboAdapterSupport(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = Objects.requireNonNull(analyzer, "analyzer");
        this.ownsAnalyzer = ownsAnalyzer;
    }

    final TurboAnalyzer analyzer() {
        if (closed) {
            throw new IllegalStateException(getClass().getSimpleName() + " is already closed");
        }
        return analyzer;
    }

    final void rememberProbs(double[] probs) {
        lastProbs = Objects.requireNonNull(probs, "probs").clone();
    }

    final double[] copyProbs() {
        return lastProbs.clone();
    }

    final boolean adapterClosed() {
        return closed;
    }

    final void closeAdapter() {
        if (closed) {
            return;
        }
        closed = true;
        if (ownsAnalyzer) {
            analyzer.close();
        }
    }

    static double probabilityOrNaN(TurboSpanView span) {
        return span.probability().orElse(Double.NaN);
    }

    static void requireSingleFragmentRows(List<TurboSpanView> rows, String operation) {
        Set<Long> items = new HashSet<>();
        for (TurboSpanView row : rows) {
            if (row.fragmentIndex() != 0L || !items.add(row.itemIndex())) {
                throw new IllegalStateException(
                        operation
                                + " cannot represent a discontinuous native item as one OpenNLP-style span; "
                                + "use TurboAnalysisResult instead");
            }
        }
    }
}
