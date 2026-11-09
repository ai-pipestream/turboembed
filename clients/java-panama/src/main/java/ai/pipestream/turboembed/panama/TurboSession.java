package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.TurboEmbedOptions;
import ai.pipestream.turboembed.TurboModelInfo;
import ai.pipestream.turboembed.TurboRerankOptions;
import ai.pipestream.turboembed.TurboSessionOptions;
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
 * using Java 22+ Panama FFM, on the device the {@link TurboSessionOptions} name (device 0, the
 * CPU, by default).
 */
public final class TurboSession implements ai.pipestream.turboembed.TurboSession {
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantLock invocationLock = new ReentrantLock(true);
    private final MemorySegment runtime;
    private final MemorySegment context;
    private final MemorySegment model;
    private final MemorySegment session;
    private final int device;

    private TurboSession(
            MemorySegment runtime,
            MemorySegment context,
            MemorySegment model,
            MemorySegment session,
            int device) {
        this.runtime = runtime;
        this.context = context;
        this.model = model;
        this.session = session;
        this.device = device;
    }

    /**
     * Opens a bundle directory on the CPU device (device 0) and creates a ready-to-run session.
     *
     * @param bundleDir path to a verified {@code libturbo} bundle directory
     * @return open {@code TurboSession}
     */
    public static TurboSession open(Path bundleDir) {
        return open(bundleDir, TurboSessionOptions.DEFAULT);
    }

    /**
     * Opens a bundle directory on the device {@code options} names (an index, or the device
     * {@code turbo_runtime_select} picks for a task) with its precision, batch, sequence length
     * and CPU threads.
     */
    public static TurboSession open(Path bundleDir, TurboSessionOptions options) {
        Objects.requireNonNull(bundleDir, "bundleDir");
        Objects.requireNonNull(options, "options");
        int[] o = options.toNative();
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
                int device = o[0];
                if (o[1] != 0) {
                    MemorySegment picked = arena.allocate(ValueLayout.JAVA_INT);
                    try {
                        code = (int) TurboLibrary.TURBO_RUNTIME_SELECT.invokeExact(
                                rt, o[1], picked, MemorySegment.NULL, 0, err);
                    } catch (Throwable t) {
                        throw new IllegalStateException("turbo_runtime_select failed", t);
                    }
                    TurboLibrary.checkStatus(code, err);
                    device = picked.get(ValueLayout.JAVA_INT, 0L);
                }
                try {
                    code = (int) TurboLibrary.TURBO_CONTEXT_CREATE.invokeExact(rt, device, outPtr, err);
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

                MemorySegment desc = arena.allocate(TurboLibrary.TURBO_SESSION_DESC_SIZE, 4L);
                desc.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_SESSION_DESC_SIZE);
                desc.set(ValueLayout.JAVA_INT, 4L, o[3]);
                desc.set(ValueLayout.JAVA_INT, 8L, o[4]);
                desc.set(ValueLayout.JAVA_INT, 12L, o[2]);
                desc.set(ValueLayout.JAVA_INT, 24L, o[5]);
                try {
                    code = (int) TurboLibrary.TURBO_SESSION_CREATE.invokeExact(mdl, desc, outPtr, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_session_create failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                MemorySegment sess = requireNativeHandle(outPtr.get(ValueLayout.ADDRESS, 0L), "turbo_session_create");
                return new TurboSession(rt, ctx, mdl, sess, device);
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

    /** Writes the texts with the options and runs; {@code read} reads the result. */
    private <T> T embedRun(List<String> texts, TurboEmbedOptions options, ResultReader<T> read) {
        validateStrings(texts, "texts");
        Objects.requireNonNull(options, "options");
        invocationLock.lock();
        try {
            MemorySegment s = requireOpen();
            String[] arr = texts.toArray(String[]::new);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment textArray = TurboLibrary.allocateTextArray(arena, arr);
                MemorySegment err = TurboLibrary.allocateError(arena);
                MemorySegment opts = embedOptions(arena, options);
                int code;
                try {
                    code = (int) TurboLibrary.TURBO_EMBED_WRITE_TEXT.invokeExact(s, textArray, arr.length, opts, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_embed_write_text failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                return executeAndRead(arena, s, err, read);
            }
        } finally {
            invocationLock.unlock();
        }
    }

    /** {@code turbo_embed_options} for {@code options}, its ranges in {@code arena}. */
    private static MemorySegment embedOptions(Arena arena, TurboEmbedOptions options) {
        int[] o = options.toNative();
        MemorySegment opts = arena.allocate(TurboLibrary.TURBO_EMBED_OPTIONS_SIZE, 8L);
        opts.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_EMBED_OPTIONS_SIZE);
        for (int i = 0; i < o.length; i++) {
            opts.set(ValueLayout.JAVA_INT, 4L * (i + 1), o[i]);
        }
        float[] ranges = options.int8Ranges();
        MemorySegment r = MemorySegment.NULL;
        if (ranges != null && ranges.length > 0) {
            r = arena.allocate((long) ranges.length * Float.BYTES, 4L);
            MemorySegment.copy(ranges, 0, r, ValueLayout.JAVA_FLOAT, 0L, ranges.length);
        }
        opts.set(ValueLayout.ADDRESS, 32L, r);
        return opts;
    }

    @Override
    public float[][] embed(List<String> texts, TurboEmbedOptions options) {
        Objects.requireNonNull(options, "options");
        if (options.quantization().codes()) {
            throw new IllegalArgumentException(options.quantization() + " makes codes: read them with embedCodes");
        }
        return embedRun(texts, options, TurboSession::readF32);
    }

    @Override
    public byte[][] embedCodes(List<String> texts, TurboEmbedOptions options) {
        Objects.requireNonNull(options, "options");
        if (!options.quantization().codes()) {
            throw new IllegalArgumentException(options.quantization() + " makes F32 vectors: read them with embed");
        }
        return embedRun(texts, options, TurboSession::readCodes);
    }

    @Override
    public TurboModelInfo modelInfo() {
        invocationLock.lock();
        try {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment err = TurboLibrary.allocateError(arena);
                MemorySegment info = arena.allocate(TurboLibrary.TURBO_MODEL_INFO_SIZE, 4L);
                info.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_MODEL_INFO_SIZE);
                int code;
                try {
                    code = (int) TurboLibrary.TURBO_MODEL_GET_INFO.invokeExact(model, info, err);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_model_get_info failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                int dims = Math.min(info.get(ValueLayout.JAVA_INT, 696L), 16);
                int[] numbers = new int[10 + dims];
                numbers[0] = device;
                for (int i = 1; i <= 7; i++) {
                    numbers[i] = info.get(ValueLayout.JAVA_INT, 4L * i);
                }
                numbers[8] = info.get(ValueLayout.JAVA_INT, 764L);
                numbers[9] = info.get(ValueLayout.JAVA_INT, 768L);
                for (int i = 0; i < dims; i++) {
                    numbers[10 + i] = info.get(ValueLayout.JAVA_INT, 700L + 4L * i);
                }
                long[][] fields = {{32, 128}, {160, 64}, {224, 72}, {296, 72}, {368, 72}, {440, 128}, {568, 128}};
                String[] text = new String[fields.length];
                for (int i = 0; i < fields.length; i++) {
                    text[i] = cString(info.asSlice(fields[i][0], fields[i][1]));
                }
                return TurboModelInfo.fromNative(numbers, text);
            }
        } finally {
            invocationLock.unlock();
        }
    }

    /** The UTF-8 string in {@code field} up to its first NUL. */
    private static String cString(MemorySegment field) {
        int n = 0;
        while (n < field.byteSize() && field.get(ValueLayout.JAVA_BYTE, n) != 0) {
            n++;
        }
        byte[] bytes = new byte[n];
        MemorySegment.copy(field, ValueLayout.JAVA_BYTE, 0L, bytes, 0, n);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    @Override
    public float[] rerank(String query, List<String> passages, TurboRerankOptions options) {
        Objects.requireNonNull(query, "query");
        validateStrings(passages, "passages");
        Objects.requireNonNull(options, "options");
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
            int[] o = options.toNative();
            MemorySegment opts = arena.allocate(TurboLibrary.TURBO_RERANK_OPTIONS_SIZE, 4L);
            opts.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_RERANK_OPTIONS_SIZE);
            for (int i = 0; i < o.length; i++) {
                opts.set(ValueLayout.JAVA_INT, 4L * (i + 1), o[i]);
            }
            int code;
            try {
                code = (int) TurboLibrary.TURBO_RERANK_WRITE_PAIRS.invokeExact(s, pairs, passages.size(), opts, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_rerank_write_pairs failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            float[][] matrix = executeAndRead(arena, s, err, TurboSession::readF32);
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

    /** Reads a run's output: its dtype, shape and the bytes read out. */
    @FunctionalInterface
    private interface ResultReader<T> {
        T read(int dtype, int batch, int dim, MemorySegment data);
    }

    private static float[][] readF32(int dtype, int batch, int dim, MemorySegment data) {
        if (dtype != TurboLibrary.TURBO_DTYPE_F32) {
            throw new IllegalStateException("the result's dtype is " + dtype + ", not F32");
        }
        float[][] out = new float[batch][dim];
        for (int r = 0; r < batch; r++) {
            MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, (long) r * dim * 4L, out[r], 0, dim);
        }
        return out;
    }

    private static byte[][] readCodes(int dtype, int batch, int dim, MemorySegment data) {
        if (dtype != TurboLibrary.TURBO_DTYPE_I8 && dtype != TurboLibrary.TURBO_DTYPE_U8) {
            throw new IllegalStateException("the result's dtype is " + dtype + ", not I8 or U8");
        }
        int width = dtype == TurboLibrary.TURBO_DTYPE_I8 ? dim : (dim + 7) / 8;
        byte[][] out = new byte[batch][width];
        for (int r = 0; r < batch; r++) {
            MemorySegment.copy(data, ValueLayout.JAVA_BYTE, (long) r * width, out[r], 0, width);
        }
        return out;
    }

    private static <T> T executeAndRead(Arena arena, MemorySegment s, MemorySegment err, ResultReader<T> read) {
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
            int dtype = info.get(ValueLayout.JAVA_INT, 16L);
            long bytes = info.get(ValueLayout.JAVA_LONG, 32L);
            long rowBytes = switch (dtype) {
                case TurboLibrary.TURBO_DTYPE_F32 -> dimLong * Float.BYTES;
                case TurboLibrary.TURBO_DTYPE_I8 -> dimLong;
                case TurboLibrary.TURBO_DTYPE_U8 -> (dimLong + 7) / 8;
                default -> throw new IllegalStateException("the result's dtype is " + dtype);
            };
            long expectedBytes = Math.multiplyExact(batchLong, rowBytes);
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

            return read.read(dtype, batch, dim, dst);
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
