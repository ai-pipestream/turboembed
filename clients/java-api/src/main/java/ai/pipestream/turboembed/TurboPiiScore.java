package ai.pipestream.turboembed;

import java.util.Objects;

/**
 * Optional numeric score attached to a PII mention or candidate by a learned or heuristic model.
 */
public record TurboPiiScore(float value, Kind kind, TurboPiiDetector model, boolean calibrated) {
    public TurboPiiScore {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException("PII score value must be finite: " + value);
        }
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(model, "model");
        if ((kind == Kind.PROBABILITY || kind == Kind.HEURISTIC) && (value < 0.0f || value > 1.0f)) {
            throw new IllegalArgumentException(kind + " PII score value must be in 0..=1: " + value);
        }
    }

    /** Native {@code TURBO_PII_SCORE_*} kinds. */
    public enum Kind {
        PROBABILITY(1),
        LOGIT(2),
        HEURISTIC(3);

        private final int nativeCode;

        Kind(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native {@code TURBO_PII_SCORE_*} code. */
        public int nativeCode() {
            return nativeCode;
        }

        public static Kind fromNative(long nativeCode) {
            for (Kind kind : values()) {
                if (kind.nativeCode == nativeCode) {
                    return kind;
                }
            }
            throw new IllegalStateException("Unknown native PII score kind: " + nativeCode);
        }
    }
}
