package ai.pipestream.turboembed.jni;

/**
 * The {@code native} methods {@code libturbo} exports for this binding, under the names the Java
 * Native Interface derives from this class. Text crosses as UTF-8 bytes; several texts cross
 * packed end to end with a length each; native handles cross as {@code long}. A failed call
 * throws {@link TurboException} with the native status, field and message.
 */
final class TurboJni {
    static {
        TurboLibrary.load();
    }

    private TurboJni() {
    }

    static native String version();

    static native String statusName(int code);

    static native String lastErrorMessage();

    static native long analyzerCreate(byte[] language, byte[] bundlePath);

    static native long analyzerRetain(long analyzer);

    static native void analyzerRelease(long analyzer);

    static native long analyzerDetectSentences(long analyzer, byte[] text);

    static native long analyzerTokenize(long analyzer, byte[] text);

    static native long analyzerPosTag(long analyzer, byte[] tokens, int[] lengths);

    static native long analyzerLemmatize(long analyzer, byte[] tokens, int[] tokenLengths, byte[] tags, int[] tagLengths);

    static native long analyzerFindTokens(long analyzer, byte[] tokens, int[] lengths);

    static native long analyzerFindText(long analyzer, byte[] text);

    static native long analyzerDetectPii(long analyzer, byte[] text);

    static native long analyzerAnalyzeGraph(long analyzer, byte[] text);

    static native long analyzerInspectGraph(long analyzer, byte[] graph, int offset, int length);

    static native int[] resultRows(long result);

    static native double[] resultProbabilities(long result);

    static native byte[] resultStrings(long result);

    static native long resultGraphLength(long result);

    static native byte[] resultGraph(long result);

    static native long resultRetain(long result);

    static native void resultRelease(long result);

    static native long sessionOpen(byte[] bundlePath);

    static native float[][] sessionEmbed(long session, byte[] texts, int[] lengths);

    static native float[] sessionRerank(long session, byte[] query, byte[] passages, int[] lengths);

    static native void sessionRelease(long session);
}
