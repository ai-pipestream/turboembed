package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Input modality. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Modality {
    TEXT(TurboNative.TURBO_MODALITY_TEXT()),
    AUDIO(TurboNative.TURBO_MODALITY_AUDIO()),
    IMAGE(TurboNative.TURBO_MODALITY_IMAGE()),
    VIDEO(TurboNative.TURBO_MODALITY_VIDEO());

    private final int value;

    Modality(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Modality of(int v) {
        for (Modality k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("modality value " + v + " is not a known constant");
    }
}
