package ai.pipestream.turboembed.jni.opennlp;

/**
 * OpenNLP-style sentence detector interface in this project's namespace.
 * This is not {@code opennlp.tools.sentdetect.SentenceDetector}.
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
     * Detects source UTF-16 sentence spans in a character sequence.
     *
     * @param s character sequence to segment into sentences
     * @return array of detected sentence spans
     */
    Span[] sentPosDetect(CharSequence s);
}
