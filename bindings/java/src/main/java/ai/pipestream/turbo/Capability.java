package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_capability;
import java.lang.foreign.MemorySegment;

/**
 * One cell of the capability matrix ({@code turbo_capability}): whether a
 * (device, task, modality) is offered, how far it is qualified, and the
 * measured precision against the reference dtype.
 */
public record Capability(
        CapStatus status,
        DType dtype,
        DType referenceDtype,
        float cosineFloor,
        float maxAbsError,
        boolean deterministic,
        String notes) {

    static Capability from(MemorySegment s) {
        int dtype = turbo_capability.dtype(s);
        int ref = turbo_capability.reference_dtype(s);
        return new Capability(
                CapStatus.of(turbo_capability.status(s)),
                dtype == 0 ? null : DType.of(dtype),
                ref == 0 ? null : DType.of(ref),
                turbo_capability.cosine_floor(s),
                turbo_capability.max_abs_error(s),
                turbo_capability.deterministic(s) != 0,
                Native.fixed(turbo_capability.notes(s)));
    }

    /** True unless the status is {@link CapStatus#UNSUPPORTED}. */
    public boolean offered() {
        return status != CapStatus.UNSUPPORTED;
    }
}
