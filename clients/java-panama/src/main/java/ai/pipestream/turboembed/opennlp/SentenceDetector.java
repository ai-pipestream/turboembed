package ai.pipestream.turboembed.opennlp;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.sentdetect.SentenceDetector} interface.
 */
public interface SentenceDetector {
    /**
     * Detects sentences in a character sequence.
     *
     * @param s character sequence to segment into sentences
     * @return array of detected sentence strings
     */
    String[] sentDetect(CharSequence s);

    /**
     * Detects sentence spans (using UTF-16 code-unit offsets) in a character sequence.
     *
     * @param s character sequence to segment into sentences
     * @return array of detected sentence spans
     */
    Span[] sentPosDetect(CharSequence s);
}
