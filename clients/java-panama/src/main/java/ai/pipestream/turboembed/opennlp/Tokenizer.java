package ai.pipestream.turboembed.opennlp;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.tokenize.Tokenizer} interface.
 */
public interface Tokenizer {
    /**
     * Splits a string into its atomic token strings.
     *
     * @param s input string to tokenize
     * @return array of token surface strings
     */
    String[] tokenize(String s);

    /**
     * Finds the boundaries of atomic tokens in a string as UTF-16 code-unit {@link Span}s.
     *
     * @param s input string to tokenize
     * @return array of token spans
     */
    Span[] tokenizePos(String s);
}
