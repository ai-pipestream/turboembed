package ai.pipestream.turboembed.opennlp;

import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboEntitySpan;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code NameFinderME} adapter backed in-process by {@code libturbo} via
 * Panama FFM.
 */
public final class TurboNameFinderME implements TokenNameFinder, AutoCloseable {
    private final TurboAnalyzer analyzer;
    private final boolean ownsAnalyzer;
    private double[] lastProbs = new double[0];

    /**
     * Creates a name finder using the built-in {@code libturbo} statistical, dictionary, regex,
     * and PII recognizers.
     */
    public TurboNameFinderME() {
        this(TurboAnalyzer.create(), true);
    }

    /**
     * Creates a name finder using an OpenNLP {@code TokenNameFinderModel} {@code .bin} file.
     *
     * @param modelPath path to the {@code .bin} name finder model
     */
    public TurboNameFinderME(Path modelPath) {
        this(
                TurboAnalyzer.create(new TurboAnalyzer.Config(
                        "en", null, null, null, null, Objects.requireNonNull(modelPath, "modelPath"))),
                true);
    }

    /**
     * Creates a name finder sharing an existing {@link TurboAnalyzer} without taking ownership of
     * its lifecycle.
     *
     * @param analyzer shared analyzer instance
     */
    public TurboNameFinderME(TurboAnalyzer analyzer) {
        this(Objects.requireNonNull(analyzer, "analyzer"), false);
    }

    private TurboNameFinderME(TurboAnalyzer analyzer, boolean ownsAnalyzer) {
        this.analyzer = analyzer;
        this.ownsAnalyzer = ownsAnalyzer;
    }

    @Override
    public Span[] find(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        List<TurboEntitySpan> entities = analyzer.findTokens(tokens);
        Span[] spans = new Span[entities.size()];
        double[] probs = new double[entities.size()];
        for (int i = 0; i < entities.size(); i++) {
            TurboEntitySpan e = entities.get(i);
            spans[i] = new Span(e.tokenStart(), e.tokenEnd(), e.kind(), e.prob());
            probs[i] = e.prob();
        }
        this.lastProbs = probs;
        return spans;
    }

    /**
     * Finds named entity and PII spans directly in raw UTF-8 text, returning character-level
     * {@link Span}s whose {@link Span#getStart()} and {@link Span#getEnd()} are UTF-16 code-unit
     * offsets compatible with {@link String#substring(int, int)}.
     *
     * @param text input text
     * @return character-indexed entity and PII spans
     */
    public Span[] findInText(String text) {
        Objects.requireNonNull(text, "text");
        List<TurboEntitySpan> entities = analyzer.findText(text);
        Span[] spans = new Span[entities.size()];
        double[] probs = new double[entities.size()];
        for (int i = 0; i < entities.size(); i++) {
            TurboEntitySpan e = entities.get(i);
            spans[i] = new Span(
                    e.utf16Start(),
                    e.utf16End(),
                    e.kind(),
                    e.prob(),
                    e.byteStart(),
                    e.byteEnd(),
                    e.codepointStart(),
                    e.codepointEnd());
            probs[i] = e.prob();
        }
        this.lastProbs = probs;
        return spans;
    }

    /**
     * Returns the probabilities associated with the spans returned by the most recent
     * {@link #find(String[])} or {@link #findInText(String)} call.
     *
     * @return entity probability array
     */
    public double[] probs() {
        return lastProbs.clone();
    }

    @Override
    public void clearAdaptiveData() {
        this.lastProbs = new double[0];
    }

    @Override
    public void close() {
        if (ownsAnalyzer) {
            analyzer.close();
        }
    }
}
