package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Pooling; MODEL means the bundle contract. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Pooling {
    MODEL(TurboNative.TURBO_POOLING_MODEL()),
    MEAN(TurboNative.TURBO_POOLING_MEAN()),
    CLS(TurboNative.TURBO_POOLING_CLS()),
    LAST(TurboNative.TURBO_POOLING_LAST());

    private final int value;

    Pooling(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Pooling of(int v) {
        for (Pooling k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("pooling value " + v + " is not a known constant");
    }
}
