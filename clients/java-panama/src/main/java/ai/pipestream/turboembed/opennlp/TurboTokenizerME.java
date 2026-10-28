package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboSpan;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code TokenizerME} adapter backed in-process by {@code libturbo} via
 * Panama FFM.
 */
public final class TurboTokenizerME implements Tokenizer, AutoCloseable {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private double[] lastProbs = new double[0];

    /**
     * Creates a tokenizer using the built-in {@code libturbo} tokenizer.
     */
    public TurboTokenizerME() {
        this(TurboAnalyzer.create(), true);
    }

    /**
     * Creates a tokenizer using an OpenNLP {@code TokenizerModel} {@code .bin} file.
     *
     * @param modelPath path to the {@code .bin} tokenizer model
     */
    public TurboTokenizerME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", null, Objects.requireNonNull(modelPath, "modelPath"), null, null, null)),
                true);
    }

    /**
     * Creates a tokenizer sharing an existing {@link TurboAnalyzer} without taking ownership of its
     * lifecycle.
     *
     * @param analyzer shared analyzer instance
     */
    public TurboTokenizerME(TurboAnalyzer analyzer) {
        this(Objects.requireNonNull(analyzer, "analyzer"), false);
    }

    private TurboTokenizerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = analyzer;
        this.ownsAnalyzer = ownsAnalyzer;
    }

    @Override
    public String[] tokenize(String s) {
        Objects.requireNonNull(s, "s");
        List<TurboSpan> spans = analyzer.tokenizeSpans(s);
        String[] tokens = new String[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            tokens[i] = sp.label();
            probs[i] = sp.prob();
        }
        this.lastProbs = probs;
        return tokens;
    }

    @Override
    public Span[] tokenizePos(String s) {
        Objects.requireNonNull(s, "s");
        List<TurboSpan> spans = analyzer.tokenizeSpans(s);
        Span[] result = new Span[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            result[i] = new Span(
                    sp.utf16Start(),
                    sp.utf16End(),
                    "token",
                    sp.prob(),
                    sp.byteStart(),
                    sp.byteEnd(),
                    sp.codepointStart(),
                    sp.codepointEnd());
            probs[i] = sp.prob();
        }
        this.lastProbs = probs;
        return result;
    }

    /**
     * Returns the token probabilities from the most recent {@link #tokenize(String)} or
     * {@link #tokenizePos(String)} call.
     *
     * @return token probability array
     */
    public double[] probs() {
        return lastProbs.clone();
    }

    @Override
    public void close() {
        if (ownsAnalyzer) {
            analyzer.close();
        }
    }
}
