package ai.pipestream.turboembed.jni.opennlp;

/**
 * OpenNLP-style lemmatizer interface in this project's namespace.
 *
 * <p>This is not {@code opennlp.tools.lemmatizer.Lemmatizer}. A {@code null} element means the
 * configured dictionary or model produced no lemma for that token; it is never replaced by a
 * lowercased token or another placeholder.
 */
public interface Lemmatizer {
    /**
     * Generates lemmas for aligned token and POS-tag arrays.
     *
     * @param toks array of tokens
     * @param tags array of POS tags for each token
     * @return lemma per token, with {@code null} for an explicit unknown
     */
    String[] lemmatize(String[] toks, String[] tags);
}
