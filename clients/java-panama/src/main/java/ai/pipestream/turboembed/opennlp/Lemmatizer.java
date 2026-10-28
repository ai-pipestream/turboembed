package ai.pipestream.turboembed.opennlp;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.lemmatizer.Lemmatizer} interface.
 */
public interface Lemmatizer {
    /**
     * Generates lemmas for the word and postag arrays, returning the result in an array.
     *
     * @param toks array of tokens
     * @param tags array of POS tags for each token
     * @return array of lemma strings for each token in {@code toks}
     */
    String[] lemmatize(String[] toks, String[] tags);
}
