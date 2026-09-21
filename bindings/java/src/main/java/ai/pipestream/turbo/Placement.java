package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Memory placement of a buffer. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Placement {
    HOST(TurboNative.TURBO_PLACE_HOST()),
    PINNED(TurboNative.TURBO_PLACE_PINNED()),
    DEVICE(TurboNative.TURBO_PLACE_DEVICE()),
    SHARED(TurboNative.TURBO_PLACE_SHARED());

    private final int value;

    Placement(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Placement of(int v) {
        for (Placement k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("placement value " + v + " is not a known constant");
    }
}
