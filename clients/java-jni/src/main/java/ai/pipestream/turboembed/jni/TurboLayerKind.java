package ai.pipestream.turboembed.jni;

/** The typed layer family represented by one native analysis span row. */
public enum TurboLayerKind {
    /** Sentence spans. */
    SENTENCES(TurboLibrary.TURBO_ANALYSIS_LAYER_SENTENCES),
    /** Token spans. */
    TOKENS(TurboLibrary.TURBO_ANALYSIS_LAYER_TOKENS),
    /** Part-of-speech rows aligned to tokens. */
    POS(TurboLibrary.TURBO_ANALYSIS_LAYER_POS),
    /** Lemma rows aligned to tokens. */
    LEMMAS(TurboLibrary.TURBO_ANALYSIS_LAYER_LEMMAS),
    /** Named-entity rows. */
    ENTITIES(TurboLibrary.TURBO_ANALYSIS_LAYER_ENTITIES),
    /** Personally identifiable information rows. */
    PII(TurboLibrary.TURBO_ANALYSIS_LAYER_PII),
    /**
     * Dependency rows aligned to tokens: the relation is the label and the head token is the
     * classification, read through {@code dependencyHead()}.
     */
    DEPENDENCIES(TurboLibrary.TURBO_ANALYSIS_LAYER_DEPENDENCIES),
    /**
     * Morphology rows aligned to tokens: one row per token, with the feature string as the label.
     */
    MORPHOLOGY(TurboLibrary.TURBO_ANALYSIS_LAYER_MORPHOLOGY);

    private final int nativeCode;

    TurboLayerKind(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    /** Returns the native {@code TURBO_ANALYSIS_LAYER_*} code. */
    public int nativeCode() {
        return nativeCode;
    }

    static TurboLayerKind fromNative(long nativeCode) {
        for (TurboLayerKind kind : values()) {
            if (kind.nativeCode == nativeCode) {
                return kind;
            }
        }
        throw new IllegalStateException("Unknown native analysis layer kind: " + nativeCode);
    }
}
