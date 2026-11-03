package ai.pipestream.turboembed.opennlp;

/**
 * OpenNLP-style tokenizer interface in this project's namespace.
 * This is not {@code opennlp.tools.tokenize.Tokenizer}.
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
     * Finds atomic token boundaries as source UTF-16 {@link Span}s with corresponding token
     * and source byte/codepoint ranges.
     *
     * @param s input string to tokenize
     * @return array of token spans
     */
    Span[] tokenizePos(String s);
}
