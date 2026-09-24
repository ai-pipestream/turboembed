package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Device selection policy. AUTO never selects a CPU. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum SelectPolicy {
    AUTO(TurboNative.TURBO_SELECT_AUTO()),
    EXPLICIT(TurboNative.TURBO_SELECT_EXPLICIT());

    private final int value;

    SelectPolicy(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static SelectPolicy of(int v) {
        for (SelectPolicy k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("selection policy value " + v + " is not a known constant");
    }
}
