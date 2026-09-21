package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_model_info;
import ai.pipestream.turbo.ffi.turbo_session_desc;
import ai.pipestream.turbo.ffi.turbo_text;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A loaded, immutable model. Sessions are created from it. */
public final class Model implements AutoCloseable {
    private final MemorySegment m;
    private final ModelInfo info;
    private boolean closed;

    Model(MemorySegment m) {
        this.m = m;
        this.info = readInfo();
    }

    private ModelInfo readInfo() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment s = turbo_model_info.allocate(arena);
            turbo_model_info.struct_size(s, (int) turbo_model_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_model_get_info(m, s, err), err);
            int n = turbo_model_info.n_labels(s);
            List<String> labels = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                MemorySegment t = turbo_text.allocate(arena);
                Native.check(turbo_model_label(m, i, t, err), err);
                labels.add(Native.textOf(t));
            }
            return ModelInfo.from(s, List.copyOf(labels));
        }
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("model is closed");
        }
        return m;
    }

    /** What loaded. */
    public ModelInfo info() {
        return info;
    }

    /** Create a session with fixed maxima; 0 means the model's default. */
    public Session createSession(int maxBatch, int maxSeq) {
        return createSession(maxBatch, maxSeq, Map.of());
    }

    /** Create a session with provider options. */
    public Session createSession(int maxBatch, int maxSeq, Map<String, String> options) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = turbo_session_desc.allocate(arena);
            turbo_session_desc.struct_size(desc, (int) turbo_session_desc.sizeof());
            turbo_session_desc.max_batch(desc, maxBatch);
            turbo_session_desc.max_seq(desc, maxSeq);
            turbo_session_desc.n_options(desc, options.size());
            turbo_session_desc.options(desc, Native.kvs(arena, options));
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_session_create(handle(), desc, out, err), err);
            return new Session(out.get(ValueLayout.ADDRESS, 0), this);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_model_release(m);
        }
    }
}
