package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Qualification status of a capability cell. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum CapStatus {
    UNSUPPORTED(TurboNative.TURBO_CAP_UNSUPPORTED()),
    PLANNED(TurboNative.TURBO_CAP_PLANNED()),
    EXPERIMENTAL(TurboNative.TURBO_CAP_EXPERIMENTAL()),
    SUPPORTED(TurboNative.TURBO_CAP_SUPPORTED());

    private final int value;

    CapStatus(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static CapStatus of(int v) {
        for (CapStatus k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("capability status value " + v + " is not a known constant");
    }
}
