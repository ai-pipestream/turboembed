package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Truncation policy; MODEL means the bundle contract. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Truncate {
    MODEL(TurboNative.TURBO_TRUNCATE_MODEL()),
    NONE(TurboNative.TURBO_TRUNCATE_NONE()),
    RIGHT(TurboNative.TURBO_TRUNCATE_RIGHT()),
    LEFT(TurboNative.TURBO_TRUNCATE_LEFT());

    private final int value;

    Truncate(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Truncate of(int v) {
        for (Truncate k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("truncate value " + v + " is not a known constant");
    }
}
