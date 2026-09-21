package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_result_info;
import ai.pipestream.turbo.ffi.turbo_span;
import ai.pipestream.turbo.ffi.turbo_tensor_info;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * The outputs of one run. The provider's storage is leased to this result
 * until it is closed; reads copy from wherever the output lives (host or
 * device) into Java memory.
 */
public final class Result implements AutoCloseable {
    private final MemorySegment r;
    private boolean closed;

    Result(MemorySegment r) {
        this.r = r;
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("result is closed");
        }
        return r;
    }

    /** One output's description. */
    public record Output(String name, DType dtype, long[] shape) {
        /** Elements in the logical shape. */
        public long elements() {
            long n = 1;
            for (long d : shape) {
                n *= d;
            }
            return n;
        }
    }

    /** Number of outputs; output 0 is the primary. */
    public int outputCount() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = turbo_result_info.allocate(arena);
            turbo_result_info.struct_size(info, (int) turbo_result_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_get_info(handle(), info, err), err);
            return turbo_result_info.n_outputs(info);
        }
    }

    /** Placement of the primary output. */
    public Placement placement() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = turbo_result_info.allocate(arena);
            turbo_result_info.struct_size(info, (int) turbo_result_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_get_info(handle(), info, err), err);
            return Placement.of(turbo_result_info.placement(info));
        }
    }

    /** Description of output {@code index}. */
    public Output output(int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment t = turbo_tensor_info.allocate(arena);
            turbo_tensor_info.struct_size(t, (int) turbo_tensor_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_output_info(handle(), index, t, err), err);
            int ndim = turbo_tensor_info.ndim(t);
            // Read the array through its slice: the generated indexed
            // accessors assume the JDK 22 var-handle coordinates and read
            // the wrong offset on JDK 25.
            MemorySegment shapes = turbo_tensor_info.shape(t);
            long[] shape = new long[ndim];
            for (int i = 0; i < ndim; i++) {
                shape[i] = shapes.getAtIndex(ValueLayout.JAVA_LONG, i);
            }
            return new Output(Native.fixed(turbo_tensor_info.name(t)), DType.of(turbo_tensor_info.dtype(t)), shape);
        }
    }

    /** Copy output {@code index} as raw bytes. */
    public byte[] readBytes(int index) {
        Output o = output(index);
        long elem = elementSize(o.dtype());
        long bytes = o.elements() * elem;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dst = arena.allocate(Math.max(bytes, 1));
            MemorySegment written = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_read(handle(), index, dst, bytes, written, err), err);
            long n = written.get(ValueLayout.JAVA_LONG, 0);
            byte[] out = new byte[(int) n];
            MemorySegment.copy(dst, ValueLayout.JAVA_BYTE, 0, out, 0, (int) n);
            return out;
        }
    }

    /** Copy an {@code f32} output. */
    public float[] readFloats(int index) {
        Output o = output(index);
        if (o.dtype() != DType.F32) {
            throw new IllegalStateException("output " + index + " is " + o.dtype() + ", not F32");
        }
        try (Arena arena = Arena.ofConfined()) {
            long bytes = o.elements() * 4;
            MemorySegment dst = arena.allocate(Math.max(bytes, 4), 4);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_read(handle(), index, dst, bytes, MemorySegment.NULL, err), err);
            return dst.asSlice(0, bytes).toArray(ValueLayout.JAVA_FLOAT);
        }
    }

    /** Copy an {@code i32} output (for example the rerank {@code sorted} indices). */
    public int[] readInts(int index) {
        Output o = output(index);
        if (o.dtype() != DType.I32) {
            throw new IllegalStateException("output " + index + " is " + o.dtype() + ", not I32");
        }
        try (Arena arena = Arena.ofConfined()) {
            long bytes = o.elements() * 4;
            MemorySegment dst = arena.allocate(Math.max(bytes, 4), 4);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_read(handle(), index, dst, bytes, MemorySegment.NULL, err), err);
            return dst.asSlice(0, bytes).toArray(ValueLayout.JAVA_INT);
        }
    }

    /** Token-classification spans, in row order. */
    public List<Span> spans() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment count = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_result_spans(handle(), MemorySegment.NULL, 0, count, err), err);
            int n = count.get(ValueLayout.JAVA_INT, 0);
            if (n == 0) {
                return List.of();
            }
            MemorySegment arr = turbo_span.allocateArray(n, arena);
            Native.check(turbo_result_spans(handle(), arr, n, count, err), err);
            List<Span> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                MemorySegment s = turbo_span.asSlice(arr, i);
                out.add(new Span(turbo_span.row(s), turbo_span.byte_start(s), turbo_span.byte_end(s), turbo_span.label(s), turbo_span.score(s)));
            }
            return out;
        }
    }

    private static long elementSize(DType d) {
        return switch (d) {
            case BOOL, U8, I8 -> 1;
            case U16, I16, F16, BF16 -> 2;
            case U32, I32, F32 -> 4;
            case U64, I64, F64 -> 8;
            case BYTES -> throw new IllegalStateException("bytes outputs have no fixed element size");
        };
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_result_release(r);
        }
    }
}
