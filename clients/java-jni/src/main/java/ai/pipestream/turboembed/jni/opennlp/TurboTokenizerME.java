package ai.pipestream.turboembed.jni.opennlp;

import ai.pipestream.turboembed.jni.TurboAnalysisResult;
import ai.pipestream.turboembed.jni.TurboAnalyzer;
import ai.pipestream.turboembed.jni.TurboLayerKind;
import ai.pipestream.turboembed.jni.TurboSpanView;
import java.util.List;
import java.util.Objects;

/** OpenNLP-style tokenizer backed by {@code libturbo}. */
public final class TurboTokenizerME extends TurboAdapterSupport implements Tokenizer, AutoCloseable {

    /** Creates a tokenizer with the default rule tokenizer. */
    public TurboTokenizerME() {
        this(TurboAnalyzer.create(), true);
    }

    /** Creates a tokenizer sharing an analyzer without owning its lifecycle. */
    public TurboTokenizerME(TurboAnalyzer analyzer) {
        this(analyzer, false);
    }

    private TurboTokenizerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        super(analyzer, ownsAnalyzer);
    }

    @Override
    public synchronized String[] tokenize(String s) {
        Objects.requireNonNull(s, "s");
        try (TurboAnalysisResult result = analyzer().tokenize(s)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.TOKENS);
            requireSingleFragmentRows(rows, "tokenize");
            String[] tokens = new String[rows.size()];
            double[] probs = new double[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                tokens[i] = rows.get(i).coveredSourceText(s);
                probs[i] = probabilityOrNaN(rows.get(i));
            }
            rememberProbs(probs);
            return tokens;
        }
    }

    @Override
    public synchronized Span[] tokenizePos(String s) {
        Objects.requireNonNull(s, "s");
        try (TurboAnalysisResult result = analyzer().tokenize(s)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.TOKENS);
            requireSingleFragmentRows(rows, "tokenizePos");
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
}
