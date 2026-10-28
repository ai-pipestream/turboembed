package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboSpan;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code POSTaggerME} adapter backed in-process by {@code libturbo} via
 * Panama FFM.
 */
public final class TurboPOSTaggerME implements POSTagger, AutoCloseable {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private double[] lastProbs = new double[0];

    /**
     * Creates a POS tagger using the built-in {@code libturbo} POS model.
     */
    public TurboPOSTaggerME() {
        this(TurboAnalyzer.create(), true);
    }

    /**
     * Creates a POS tagger using an OpenNLP {@code POSModel} {@code .bin} file.
     *
     * @param modelPath path to the {@code .bin} POS model
     */
    public TurboPOSTaggerME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", null, null, Objects.requireNonNull(modelPath, "modelPath"), null, null)),
                true);
    }

    /**
     * Creates a POS tagger sharing an existing {@link TurboAnalyzer} without taking ownership of
     * its lifecycle.
     *
     * @param analyzer shared analyzer instance
     */
    public TurboPOSTaggerME(TurboAnalyzer analyzer) {
        this(Objects.requireNonNull(analyzer, "analyzer"), false);
    }

    private TurboPOSTaggerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = analyzer;
        this.ownsAnalyzer = ownsAnalyzer;
    }

    @Override
    public String[] tag(String[] sentence) {
        Objects.requireNonNull(sentence, "sentence");
        List<TurboSpan> spans = analyzer.posTag(sentence);
        String[] tags = new String[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            tags[i] = sp.label();
            probs[i] = sp.prob();
        }
        this.lastProbs = probs;
        return tags;
    }

    /**
     * Returns the probabilities of the tags assigned by the most recent {@link #tag(String[])}
     * call.
     *
     * @return tag probability array
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
