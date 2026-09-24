package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Requested output element type; MODEL means the compute dtype. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum OutputDType {
    MODEL(TurboNative.TURBO_OUTPUT_MODEL()),
    F32(TurboNative.TURBO_OUTPUT_F32()),
    F16(TurboNative.TURBO_OUTPUT_F16()),
    I8(TurboNative.TURBO_OUTPUT_I8());

    private final int value;

    OutputDType(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static OutputDType of(int v) {
        for (OutputDType k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("output dtype value " + v + " is not a known constant");
    }
}
