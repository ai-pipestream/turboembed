package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Element type. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum DType {
    BOOL(TurboNative.TURBO_DTYPE_BOOL()),
    U8(TurboNative.TURBO_DTYPE_U8()),
    I8(TurboNative.TURBO_DTYPE_I8()),
    U16(TurboNative.TURBO_DTYPE_U16()),
    I16(TurboNative.TURBO_DTYPE_I16()),
    U32(TurboNative.TURBO_DTYPE_U32()),
    I32(TurboNative.TURBO_DTYPE_I32()),
    U64(TurboNative.TURBO_DTYPE_U64()),
    I64(TurboNative.TURBO_DTYPE_I64()),
    F16(TurboNative.TURBO_DTYPE_F16()),
    BF16(TurboNative.TURBO_DTYPE_BF16()),
    F32(TurboNative.TURBO_DTYPE_F32()),
    F64(TurboNative.TURBO_DTYPE_F64()),
    BYTES(TurboNative.TURBO_DTYPE_BYTES());

    private final int value;

    DType(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static DType of(int v) {
        for (DType k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("dtype value " + v + " is not a known constant");
    }
}
