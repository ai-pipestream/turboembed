package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_context_desc;
import ai.pipestream.turbo.ffi.turbo_model_desc;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;

/**
 * A device plus its memory domain. Models are loaded into a context; the
 * context retains the runtime, and a model retains the context.
 */
public final class Context implements AutoCloseable {
    private final MemorySegment ctx;
    private final int deviceIndex;
    private boolean closed;

    private Context(MemorySegment ctx, int deviceIndex) {
        this.ctx = ctx;
        this.deviceIndex = deviceIndex;
    }

    static Context create(Turbo rt, int index) {
        return create(rt, index, Map.of());
    }

    /** Create with provider options; unknown keys are refused by the provider. */
    public static Context create(Turbo rt, int index, Map<String, String> options) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = turbo_context_desc.allocate(arena);
            turbo_context_desc.struct_size(desc, (int) turbo_context_desc.sizeof());
            turbo_context_desc.n_options(desc, options.size());
            turbo_context_desc.options(desc, Native.kvs(arena, options));
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_context_create(rt.handle(), index, desc, out, err), err);
            return new Context(out.get(ValueLayout.ADDRESS, 0), index);
        }
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("context is closed");
        }
        return ctx;
    }

    /** Index of the device this context runs on. */
    public int deviceIndex() {
        return deviceIndex;
    }

    /** Load and verify a bundle directory. */
    public Model loadModel(String bundlePath) {
        return loadModel(bundlePath, Map.of());
    }

    /** Load with provider options. */
    public Model loadModel(String bundlePath, Map<String, String> options) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = turbo_model_desc.allocate(arena);
            turbo_model_desc.struct_size(desc, (int) turbo_model_desc.sizeof());
            turbo_model_desc.n_options(desc, options.size());
            turbo_model_desc.options(desc, Native.kvs(arena, options));
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_model_load(handle(), Native.text(arena, bundlePath), desc, out, err), err);
            return new Model(out.get(ValueLayout.ADDRESS, 0));
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_context_release(ctx);
        }
    }
}
