package ai.pipestream.turboembed.opennlp;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.postag.POSTagger} interface.
 */
public interface POSTagger {
    /**
     * Assigns the sentence of tokens part-of-speech tags.
     *
     * @param sentence array of tokens to be tagged
     * @return array of part-of-speech tags for each token provided in {@code sentence}
     */
    String[] tag(String[] sentence);
}
