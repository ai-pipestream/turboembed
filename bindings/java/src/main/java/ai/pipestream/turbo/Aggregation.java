package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Token-classification span aggregation; MODEL means the bundle contract. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Aggregation {
    MODEL(TurboNative.TURBO_AGGREGATE_MODEL()),
    NONE(TurboNative.TURBO_AGGREGATE_NONE()),
    SIMPLE(TurboNative.TURBO_AGGREGATE_SIMPLE()),
    FIRST(TurboNative.TURBO_AGGREGATE_FIRST()),
    MAX(TurboNative.TURBO_AGGREGATE_MAX());

    private final int value;

    Aggregation(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Aggregation of(int v) {
        for (Aggregation k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("aggregation value " + v + " is not a known constant");
    }
}
