package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboSpan;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code LemmatizerME} / {@code DictionaryLemmatizer} adapter backed
 * in-process by {@code libturbo} via Panama FFM.
 */
public final class TurboLemmatizerME implements Lemmatizer, AutoCloseable {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private double[] lastProbs = new double[0];

    /**
     * Creates a lemmatizer using the built-in {@code libturbo} dictionary and statistical
     * lemmatizer.
     */
    public TurboLemmatizerME() {
        this(TurboAnalyzer.create(), true);
    }

    /**
     * Creates a lemmatizer using an OpenNLP {@code LemmatizerModel} {@code .bin} or dictionary
     * file.
     *
     * @param modelPath path to the {@code .bin} model or dictionary file
     */
    public TurboLemmatizerME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", null, null, null, Objects.requireNonNull(modelPath, "modelPath"), null)),
                true);
    }

    /**
     * Creates a lemmatizer sharing an existing {@link TurboAnalyzer} without taking ownership of
     * its lifecycle.
     *
     * @param analyzer shared analyzer instance
     */
    public TurboLemmatizerME(TurboAnalyzer analyzer) {
        this(Objects.requireNonNull(analyzer, "analyzer"), false);
    }

    private TurboLemmatizerME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = analyzer;
        this.ownsAnalyzer = ownsAnalyzer;
    }

    @Override
    public String[] lemmatize(String[] toks, String[] tags) {
        Objects.requireNonNull(toks, "toks");
        Objects.requireNonNull(tags, "tags");
        List<TurboSpan> spans = analyzer.lemmatize(toks, tags);
        String[] lemmas = new String[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            lemmas[i] = sp.label();
            probs[i] = sp.prob();
        }
        this.lastProbs = probs;
        return lemmas;
    }

    /**
     * Returns the probabilities associated with the most recent
     * {@link #lemmatize(String[], String[])} call.
     *
     * @return lemma probability array
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
