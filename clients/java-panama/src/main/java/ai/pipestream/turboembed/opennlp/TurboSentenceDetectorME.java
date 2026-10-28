package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboSpan;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code SentenceDetectorME} adapter backed in-process by {@code libturbo}
 * via Panama FFM.
 */
public final class TurboSentenceDetectorME implements SentenceDetector, AutoCloseable {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private double[] lastProbs = new double[0];

    /**
     * Creates a sentence detector with the default English rules and abbreviation inventory.
     */
    public TurboSentenceDetectorME() {
        this(TurboAnalyzer.create(), true);
    }

    /**
     * Creates a sentence detector using an OpenNLP {@code SentenceModel} {@code .bin} file.
     *
     * @param modelPath path to the {@code .bin} sentence model
     */
    public TurboSentenceDetectorME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", Objects.requireNonNull(modelPath, "modelPath"), null, null, null, null)),
                true);
    }

    /**
     * Creates a sentence detector sharing an existing {@link TurboAnalyzer} without taking
     * ownership of its lifecycle.
     *
     * @param analyzer shared analyzer instance
     */
    public TurboSentenceDetectorME(TurboAnalyzer analyzer) {
        this(Objects.requireNonNull(analyzer, "analyzer"), false);
    }

    private TurboSentenceDetectorME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = analyzer;
        this.ownsAnalyzer = ownsAnalyzer;
    }

    @Override
    public String[] sentDetect(CharSequence s) {
        Objects.requireNonNull(s, "s");
        List<TurboSpan> spans = analyzer.detectSentences(s.toString());
        String[] sentences = new String[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            sentences[i] = sp.label();
            probs[i] = sp.prob();
        }
        this.lastProbs = probs;
        return sentences;
    }

    @Override
    public Span[] sentPosDetect(CharSequence s) {
        Objects.requireNonNull(s, "s");
        List<TurboSpan> spans = analyzer.detectSentences(s.toString());
        Span[] result = new Span[spans.size()];
        double[] probs = new double[spans.size()];
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan sp = spans.get(i);
            result[i] = new Span(
                    sp.utf16Start(),
                    sp.utf16End(),
                    "sentence",
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
     * Returns the probabilities associated with the most recent {@link #sentDetect(CharSequence)}
     * or {@link #sentPosDetect(CharSequence)} call.
     *
     * @return sentence probability array
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
