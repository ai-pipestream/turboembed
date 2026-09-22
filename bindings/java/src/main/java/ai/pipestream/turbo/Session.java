package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_run_options;
import ai.pipestream.turbo.ffi.turbo_session_stats;
import ai.pipestream.turbo.ffi.turbo_token_batch;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;

/**
 * An execution workspace with fixed maxima. Single owner: a concurrent call
 * on the same session fails with {@code TURBO_E_BUSY} rather than racing,
 * and a write or run while a {@link Result} of this session is still open
 * fails the same way.
 */
public final class Session implements AutoCloseable {
    private final MemorySegment s;
    private final Model model;
    private boolean closed;

    Session(MemorySegment s, Model model) {
        this.s = s;
        this.model = model;
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("session is closed");
        }
        return s;
    }

    /** The model this session runs. */
    public Model model() {
        return model;
    }

    /** Write texts for embedding. */
    public void writeText(List<String> texts, EmbedOptions opts) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(turbo_session_write_text(handle(), Native.texts(arena, texts), texts.size(), opts.toNative(arena), err), err);
        }
    }

    /**
     * Write caller-prepared token rows ({@code ids} and {@code mask} are
     * {@code batch x seq}, row-major, with {@code rowStride} elements
     * between row starts; {@code types} may be null).
     */
    public void writeTokens(int batch, int seq, int rowStride, int[] ids, int[] mask, int[] types) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment b = turbo_token_batch.allocate(arena);
            turbo_token_batch.struct_size(b, (int) turbo_token_batch.sizeof());
            turbo_token_batch.batch(b, batch);
            turbo_token_batch.seq(b, seq);
            turbo_token_batch.row_stride(b, rowStride);
            turbo_token_batch.ids(b, arena.allocateFrom(ValueLayout.JAVA_INT, ids));
            turbo_token_batch.mask(b, arena.allocateFrom(ValueLayout.JAVA_INT, mask));
            turbo_token_batch.types(b, types == null ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, types));
            MemorySegment err = Native.error(arena);
            Native.check(turbo_session_write_tokens(handle(), b, err), err);
        }
    }

    /** Write a query and documents for reranking. */
    public void writePairs(String query, List<String> docs, RerankOptions opts) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(
                    turbo_session_write_pairs(handle(), Native.text(arena, query), Native.texts(arena, docs), docs.size(), opts.toNative(arena), err),
                    err);
        }
    }

    /** Write texts for classification or token classification. */
    public void writeTextClassify(List<String> texts, ClassifyOptions opts) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(
                    turbo_session_write_text_classify(handle(), Native.texts(arena, texts), texts.size(), opts.toNative(arena), err), err);
        }
    }

    /** Execute the written inputs. The result holds a lease on this session until closed. */
    public Result run() {
        return run(Map.of());
    }

    /** Execute with provider run parameters. */
    public Result run(Map<String, String> params) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment o = turbo_run_options.allocate(arena);
            turbo_run_options.struct_size(o, (int) turbo_run_options.sizeof());
            turbo_run_options.n_params(o, params.size());
            turbo_run_options.params(o, Native.kvs(arena, params));
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_session_run(handle(), o, out, err), err);
            return new Result(out.get(ValueLayout.ADDRESS, 0), this);
        }
    }

    /** Counters. */
    public SessionStats stats() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment st = turbo_session_stats.allocate(arena);
            turbo_session_stats.struct_size(st, (int) turbo_session_stats.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_session_get_stats(handle(), st, err), err);
            long allocs = turbo_session_stats.provider_allocs(st);
            long hostAllocs = turbo_session_stats.host_allocs(st);
            return new SessionStats(
                    turbo_session_stats.runs(st),
                    hostAllocs == -1L ? null : hostAllocs,
                    turbo_session_stats.h2d_bytes(st),
                    turbo_session_stats.d2h_bytes(st),
                    turbo_session_stats.input_bytes(st),
                    turbo_session_stats.output_bytes(st),
                    allocs == -1L ? null : allocs);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_session_release(s);
        }
    }
}
