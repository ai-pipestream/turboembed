package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Hardware class of a device. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum DeviceKind {
    CPU(TurboNative.TURBO_DEVICE_CPU()),
    GPU(TurboNative.TURBO_DEVICE_GPU()),
    IGPU(TurboNative.TURBO_DEVICE_IGPU()),
    NPU(TurboNative.TURBO_DEVICE_NPU()),
    ACCEL(TurboNative.TURBO_DEVICE_ACCEL());

    private final int value;

    DeviceKind(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static DeviceKind of(int v) {
        for (DeviceKind k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("device kind value " + v + " is not a known constant");
    }
}
