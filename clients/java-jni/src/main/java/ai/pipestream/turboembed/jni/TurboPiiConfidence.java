package ai.pipestream.turboembed.jni;

/** Layer-specific evidence classification carried by PII rows. */
public enum TurboPiiConfidence {
    /** No PII evidence classification is present. */
    NONE(0),
    /** The value has the expected format. */
    FORMAT(TurboLibrary.TURBO_PII_CONFIDENCE_FORMAT),
    /** A checksum over the value passes. */
    CHECKSUM(TurboLibrary.TURBO_PII_CONFIDENCE_CHECKSUM),
    /** A nearby label names the kind and a checksum passes. */
    LABELLED_CHECKSUM(TurboLibrary.TURBO_PII_CONFIDENCE_LABELLED);

    private final int nativeCode;

    TurboPiiConfidence(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    /** Returns the native {@code TURBO_PII_CONFIDENCE_*} code. */
    public int nativeCode() {
        return nativeCode;
    }

    static TurboPiiConfidence fromNative(long nativeCode) {
        for (TurboPiiConfidence confidence : values()) {
            if (confidence.nativeCode == nativeCode) {
                return confidence;
            }
        }
        throw new IllegalStateException("Unknown native PII confidence classification: " + nativeCode);
    }
}
