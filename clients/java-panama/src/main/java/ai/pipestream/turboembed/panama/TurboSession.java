package ai.pipestream.turboembed.panama;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process model session over {@code libturbo} for dense embedding and cross-encoder reranking
 * using Java 22+ Panama FFM.
 */
public final class TurboSession implements AutoCloseable {
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantLock invocationLock = new ReentrantLock(true);
    private final MemorySegment runtime;
    private final MemorySegment context;
    private final MemorySegment model;
    private final MemorySegment session;

    private TurboSession(
            MemorySegment runtime,
            MemorySegment context,
            MemorySegment model,
            MemorySegment session) {
        this.runtime = runtime;
        this.context = context;
        this.model = model;
        this.session = session;
    }

    /**
     * Opens a bundle directory on the CPU device (device 0) and creates a ready-to-run session.
     *
     * @param bundleDir path to a verified {@code libturbo} bundle directory
     * @return open {@code TurboSession}
     */
    public static TurboSession open(Path bundleDir) {
        Objects.requireNonNull(bundleDir, "bundleDir");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = TurboLibrary.allocateError(arena);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);

            int code;
            try {
                code = (int) TurboLibrary.TURBO_RUNTIME_CREATE.invokeExact(MemorySegment.NULL, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_runtime_create failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            MemorySegment rt = requireNativeHandle(outPtr.get(ValueLayout.ADDRESS, 0L), "turbo_runtime_create");

            MemorySegment ctx = MemorySegment.NULL;
            MemorySegment mdl = MemorySegment.NULL;
            try {
                try {
                    code = (int) TurboLibrary.TURBO_CONTEXT_CREATE.invokeExact(rt, 0, outPtr, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_context_create failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                ctx = requireNativeHandle(outPtr.get(ValueLayout.ADDRESS, 0L), "turbo_context_create");

                MemorySegment pathText = TurboLibrary.allocateText(arena, bundleDir.toString());
                try {
                    code = (int) TurboLibrary.TURBO_MODEL_LOAD.invokeExact(ctx, pathText, outPtr, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_model_load failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                mdl = requireNativeHandle(outPtr.get(ValueLayout.ADDRESS, 0L), "turbo_model_load");

                try {
                    code = (int) TurboLibrary.TURBO_SESSION_CREATE.invokeExact(
                            mdl, MemorySegment.NULL, outPtr, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_session_create failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                MemorySegment sess = requireNativeHandle(outPtr.get(ValueLayout.ADDRESS, 0L), "turbo_session_create");
                return new TurboSession(rt, ctx, mdl, sess);
            } catch (RuntimeException e) {
                releaseSuppressed(e, TurboLibrary.TURBO_MODEL_RELEASE, mdl);
                releaseSuppressed(e, TurboLibrary.TURBO_CONTEXT_RELEASE, ctx);
                releaseSuppressed(e, TurboLibrary.TURBO_RUNTIME_RELEASE, rt);
                throw e;
            }
        }
    }

    private static MemorySegment requireNativeHandle(MemorySegment handle, String operation) {
        if (handle == null || MemorySegment.NULL.equals(handle)) {
            throw new IllegalStateException(operation + " returned NULL");
        }
        return handle;
    }

    private MemorySegment requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("TurboSession is already closed");
        }
        return session;
    }

    private static void validateStrings(List<String> values, String name) {
        Objects.requireNonNull(values, name);
        for (int i = 0; i < values.size(); i++) {
            Objects.requireNonNull(values.get(i), name + "[" + i + "]");
        }
    }

    /**
     * Embeds a batch of UTF-8 texts and returns one {@code float[dim]} vector per input text.
     *
     * @param texts non-empty list of texts to embed
     * @return {@code float[texts.size()][dim]} embedding matrix
     */
    public float[][] embed(List<String> texts) {
        validateStrings(texts, "texts");
        invocationLock.lock();
        try {
        MemorySegment s = requireOpen();
        String[] arr = texts.toArray(String[]::new);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textArray = TurboLibrary.allocateTextArray(arena, arr);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_EMBED_WRITE_TEXT.invokeExact(
                        s, textArray, arr.length, MemorySegment.NULL, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_embed_write_text failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return executeAndReadF32(arena, s, err);
        }
        } finally {
            invocationLock.unlock();
        }
    }

    /**
     * Reranks {@code passages} against {@code query} and returns raw relevance logits.
     *
     * @param query    query text
     * @param passages candidate passage texts
     * @return one relevance score per passage
     */
    public float[] rerank(String query, List<String> passages) {
        Objects.requireNonNull(query, "query");
        validateStrings(passages, "passages");
        invocationLock.lock();
        try {
        MemorySegment s = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            long pairStride = TurboLibrary.TURBO_TEXT_PAIR_LAYOUT.byteSize();
            MemorySegment pairs = passages.isEmpty()
                    ? MemorySegment.NULL
                    : arena.allocate(pairStride * passages.size(), TurboLibrary.TURBO_TEXT_PAIR_LAYOUT.byteAlignment());
            for (int i = 0; i < passages.size(); i++) {
                long base = i * pairStride;
                TurboLibrary.writeTextInPlace(arena, pairs, base, query);
                TurboLibrary.writeTextInPlace(arena, pairs, base + 16L, passages.get(i));
            }
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_RERANK_WRITE_PAIRS.invokeExact(
                        s, pairs, passages.size(), MemorySegment.NULL, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_rerank_write_pairs failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            float[][] matrix = executeAndReadF32(arena, s, err);
            if (matrix.length != passages.size()) {
                throw new IllegalStateException(
                        "native rerank returned " + matrix.length + " rows for " + passages.size() + " passages");
            }
            float[] scores = new float[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                if (matrix[i].length != 1) {
                    throw new IllegalStateException("native rerank row " + i + " has dimension " + matrix[i].length);
                }
                scores[i] = matrix[i][0];
            }
            return scores;
        }
        } finally {
            invocationLock.unlock();
        }
    }

    private static float[][] executeAndReadF32(Arena arena, MemorySegment s, MemorySegment err) {
        MemorySegment resOut = arena.allocate(ValueLayout.ADDRESS);
        int code;
        try {
            code = (int) TurboLibrary.TURBO_SESSION_RUN.invokeExact(s, resOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_session_run failed", t);
        }
        TurboLibrary.checkStatus(code, err);
        MemorySegment resPtr = resOut.get(ValueLayout.ADDRESS, 0L);
        if (resPtr == null || MemorySegment.NULL.equals(resPtr)) {
            throw new IllegalStateException("turbo_session_run returned NULL");
        }
        RuntimeException failure = null;
        try {
            MemorySegment info = arena.allocate(TurboLibrary.TURBO_RESULT_INFO_SIZE, 8L);
            info.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_RESULT_INFO_SIZE);
            try {
                code = (int) TurboLibrary.TURBO_RESULT_GET_INFO.invokeExact(resPtr, info, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_result_get_info failed", t);
            }
            TurboLibrary.checkStatus(code, err);

            long batchLong = Integer.toUnsignedLong(info.get(ValueLayout.JAVA_INT, 8L));
            long dimLong = Integer.toUnsignedLong(info.get(ValueLayout.JAVA_INT, 12L));
            if (batchLong > Integer.MAX_VALUE || dimLong > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "native result shape exceeds Java arrays: " + batchLong + " x " + dimLong);
            }
            int batch = (int) batchLong;
            int dim = (int) dimLong;
            long bytes = info.get(ValueLayout.JAVA_LONG, 32L);
            long expectedBytes = Math.multiplyExact(Math.multiplyExact(batchLong, dimLong), Float.BYTES);
            if (bytes < 0 || bytes != expectedBytes) {
                throw new IllegalStateException(
                        "native result byte length " + bytes + " does not match shape bytes " + expectedBytes);
            }

            MemorySegment dst = arena.allocate(Math.max(bytes, 4L), 4L);
            MemorySegment written = arena.allocate(ValueLayout.JAVA_LONG);
            try {
                code = (int) TurboLibrary.TURBO_RESULT_READ.invokeExact(resPtr, dst, bytes, written, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_result_read failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            long writtenBytes = written.get(ValueLayout.JAVA_LONG, 0L);
            if (writtenBytes != bytes) {
                throw new IllegalStateException(
                        "native result read wrote " + writtenBytes + " bytes, expected " + bytes);
            }

            float[][] out = new float[batch][dim];
            for (int r = 0; r < batch; r++) {
                for (int c = 0; c < dim; c++) {
                    out[r][c] = dst.get(ValueLayout.JAVA_FLOAT, ((long) r * dim + c) * 4L);
                }
            }
            return out;
        } catch (RuntimeException e) {
            failure = e;
            throw e;
        } finally {
            if (failure == null) {
                safeRelease(TurboLibrary.TURBO_RESULT_RELEASE, resPtr);
            } else {
                releaseSuppressed(failure, TurboLibrary.TURBO_RESULT_RELEASE, resPtr);
            }
        }
    }

    private static void releaseSuppressed(
            RuntimeException failure, java.lang.invoke.MethodHandle releaseFn, MemorySegment ptr) {
        try {
            safeRelease(releaseFn, ptr);
        } catch (RuntimeException releaseFailure) {
            failure.addSuppressed(releaseFailure);
        }
    }

    private static void safeRelease(java.lang.invoke.MethodHandle releaseFn, MemorySegment ptr) {
        if (ptr != null && !MemorySegment.NULL.equals(ptr)) {
            try {
                releaseFn.invokeExact(ptr);
            } catch (Throwable t) {
                throw new IllegalStateException("native release failed", t);
            }
        }
    }

    /** Returns whether this session has released its native handles. */
    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        invocationLock.lock();
        try {
            RuntimeException failure = null;
            try {
                safeRelease(TurboLibrary.TURBO_SESSION_RELEASE, session);
            } catch (RuntimeException e) {
                failure = e;
            }
            try {
                safeRelease(TurboLibrary.TURBO_MODEL_RELEASE, model);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
            try {
                safeRelease(TurboLibrary.TURBO_CONTEXT_RELEASE, context);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
            try {
                safeRelease(TurboLibrary.TURBO_RUNTIME_RELEASE, runtime);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
            if (failure != null) {
                throw failure;
            }
        } finally {
            invocationLock.unlock();
        }
    }
}
