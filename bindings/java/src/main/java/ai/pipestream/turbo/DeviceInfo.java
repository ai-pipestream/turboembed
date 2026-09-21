package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_device_info;
import java.lang.foreign.MemorySegment;

/**
 * Static description of one device ({@code turbo_device_info}). {@code caps}
 * holds the {@code TURBO_CAP_*} bits; an option is honored only when its bit
 * is set, otherwise the call that sets it fails with
 * {@code TURBO_E_UNSUPPORTED_OPTION}.
 */
public record DeviceInfo(
        int index,
        DeviceKind kind,
        int ordinal,
        int vendorId,
        long caps,
        long memoryTotal,
        long memoryFree,
        String name,
        String vendor,
        String providerId,
        String providerVersion,
        String runtimeVersion,
        String driverVersion) {

    static DeviceInfo from(int index, MemorySegment s) {
        return new DeviceInfo(
                index,
                DeviceKind.of(turbo_device_info.kind(s)),
                turbo_device_info.ordinal(s),
                turbo_device_info.vendor_id(s),
                turbo_device_info.caps(s),
                turbo_device_info.memory_total(s),
                turbo_device_info.memory_free(s),
                Native.fixed(turbo_device_info.name(s)),
                Native.fixed(turbo_device_info.vendor(s)),
                Native.fixed(turbo_device_info.provider_id(s)),
                Native.fixed(turbo_device_info.provider_version(s)),
                Native.fixed(turbo_device_info.runtime_version(s)),
                Native.fixed(turbo_device_info.driver_version(s)));
    }

    /** True when every bit of {@code bits} is advertised. */
    public boolean has(long bits) {
        return (caps & bits) == bits;
    }
}
