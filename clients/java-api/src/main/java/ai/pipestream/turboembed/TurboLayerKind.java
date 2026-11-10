package ai.pipestream.turboembed;

/** The typed layer family represented by one native analysis span row. */
public enum TurboLayerKind {
    /** Sentence spans. */
    SENTENCES(1),
    /** Token spans. */
    TOKENS(2),
    /** Part-of-speech rows aligned to tokens. */
    POS(3),
    /** Lemma rows aligned to tokens. */
    LEMMAS(4),
    /** Named-entity rows. */
    ENTITIES(5),
    /** Personally identifiable information rows. */
    PII(6),
    /**
     * Dependency rows aligned to tokens: the relation is the label and the head token is the
     * classification, read through {@code dependencyHead()}.
     */
    DEPENDENCIES(7),
    /**
     * Morphology rows aligned to tokens: one row per token, with the feature string as the label.
     */
    MORPHOLOGY(8),
    /**
     * Syntactic word rows from a token split head: the tokens a word belongs to as its token range,
     * and its form as the label only when the form is not the text of its range.
     */
    WORDS(9);

    private final int nativeCode;

    TurboLayerKind(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    /** Returns the native {@code TURBO_ANALYSIS_LAYER_*} code. */
    public int nativeCode() {
        return nativeCode;
    }

    public static TurboLayerKind fromNative(long nativeCode) {
        for (TurboLayerKind kind : values()) {
            if (kind.nativeCode == nativeCode) {
                return kind;
            }
        }
        throw new IllegalStateException("Unknown native analysis layer kind: " + nativeCode);
    }
}
