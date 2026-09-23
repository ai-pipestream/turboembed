package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Normalization; MODEL means the bundle contract. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Normalize {
    MODEL(TurboNative.TURBO_NORMALIZE_MODEL()),
    NONE(TurboNative.TURBO_NORMALIZE_NONE()),
    L2(TurboNative.TURBO_NORMALIZE_L2());

    private final int value;

    Normalize(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Normalize of(int v) {
        for (Normalize k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("normalize value " + v + " is not a known constant");
    }
}
