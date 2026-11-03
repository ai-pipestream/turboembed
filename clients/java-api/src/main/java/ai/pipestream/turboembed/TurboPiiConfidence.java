package ai.pipestream.turboembed;

/** Layer-specific evidence classification carried by PII rows. */
public enum TurboPiiConfidence {
    /** No PII evidence classification is present. */
    NONE(0),
    /** The value has the expected format. */
    FORMAT(1),
    /** A checksum over the value passes. */
    CHECKSUM(2),
    /** A nearby label names the kind and a checksum passes. */
    LABELLED_CHECKSUM(3);

    private final int nativeCode;

    TurboPiiConfidence(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    /** Returns the native {@code TURBO_PII_CONFIDENCE_*} code. */
    public int nativeCode() {
        return nativeCode;
    }

    public static TurboPiiConfidence fromNative(long nativeCode) {
        for (TurboPiiConfidence confidence : values()) {
            if (confidence.nativeCode == nativeCode) {
                return confidence;
            }
        }
        throw new IllegalStateException("Unknown native PII confidence classification: " + nativeCode);
    }
}
