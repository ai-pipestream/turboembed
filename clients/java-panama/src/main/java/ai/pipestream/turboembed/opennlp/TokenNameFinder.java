package ai.pipestream.turboembed.opennlp;

/**
 * OpenNLP-style token name finder interface in this project's namespace.
 * This is not {@code opennlp.tools.namefind.TokenNameFinder}.
 */
public interface TokenNameFinder {
    /**
     * Generates token-indexed name spans for one independent token sequence.
     *
     * @param tokens array of sentence tokens
     * @return array of token-indexed {@link Span}s
     */
    Span[] find(String[] tokens);

    /**
     * Clears adaptive data. The native binding constructs fresh model context for every
     * {@code find} call and retains no adaptive data, so implementations may make this a no-op.
     */
    void clearAdaptiveData();
}
