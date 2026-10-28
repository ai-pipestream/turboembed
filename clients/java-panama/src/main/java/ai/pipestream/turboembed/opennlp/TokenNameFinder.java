package ai.pipestream.turboembed.opennlp;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.namefind.TokenNameFinder} interface.
 */
public interface TokenNameFinder {
    /**
     * Generates name spans for the given sequence of tokens.
     *
     * @param tokens array of sentence tokens
     * @return array of token-indexed {@link Span}s
     */
    Span[] find(String[] tokens);

    /**
     * Forgets all adaptive data collected during previous calls to {@link #find(String[])}.
     */
    void clearAdaptiveData();
}
