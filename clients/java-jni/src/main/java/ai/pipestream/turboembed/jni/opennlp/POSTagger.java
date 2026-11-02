package ai.pipestream.turboembed.jni.opennlp;

/**
 * OpenNLP-style POS tagger interface in this project's namespace.
 *
 * <p>This is not {@code opennlp.tools.postag.POSTagger}. A {@code null} element means the
 * configured model produced no tag for that token.
 */
public interface POSTagger {
    /**
     * Assigns part-of-speech tags to one token sequence.
     *
     * @param sentence array of tokens to be tagged
     * @return tag per token, with {@code null} for an explicit unknown
     */
    String[] tag(String[] sentence);
}
