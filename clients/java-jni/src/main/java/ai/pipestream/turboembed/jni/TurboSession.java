package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboEmbedOptions;
import ai.pipestream.turboembed.TurboModelInfo;
import ai.pipestream.turboembed.TurboRerankOptions;
import ai.pipestream.turboembed.TurboSessionOptions;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process model session over {@code libturbo} for dense embedding and cross-encoder reranking
 * through the Java Native Interface. The native library opens a runtime, a context on the device
 * the {@link TurboSessionOptions} name (device 0, the CPU, by default), the bundle and one session
 * over it, and releases them together on {@link #close()}.
 */
public final class TurboSession implements ai.pipestream.turboembed.TurboSession {
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final ReentrantLock invocationLock = new ReentrantLock(true);
    private final long handle;

    private TurboSession(long handle) {
        this.handle = handle;
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
     * Opens a bundle directory on the device {@code options} names, with its precision, batch,
     * sequence length and CPU threads.
     */
    public static TurboSession open(Path bundleDir, TurboSessionOptions options) {
        Objects.requireNonNull(bundleDir, "bundleDir");
        Objects.requireNonNull(options, "options");
        long handle = TurboJni.sessionOpen(Utf8.bytes(bundleDir.toString()), options.toNative());
        if (handle == 0L) {
            throw new IllegalStateException("turbo_session_create returned NULL");
        }
        return new TurboSession(handle);
    }

    private long requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("TurboSession is already closed");
        }
        return handle;
    }

    private static String[] validateStrings(List<String> values, String name) {
        Objects.requireNonNull(values, name);
        String[] array = values.toArray(new String[0]);
        for (int i = 0; i < array.length; i++) {
            Objects.requireNonNull(array[i], name + "[" + i + "]");
        }
        return array;
    }

    @Override
    public float[][] embed(List<String> texts, TurboEmbedOptions options) {
        String[] array = validateStrings(texts, "texts");
        Objects.requireNonNull(options, "options");
        if (options.quantization().codes()) {
            throw new IllegalArgumentException(options.quantization() + " makes codes: read them with embedCodes");
        }
        invocationLock.lock();
        try {
            long session = requireOpen();
            Utf8.Packed packed = Utf8.pack(array, "texts");
            float[][] matrix = TurboJni.sessionEmbed(
                    session, packed.bytes, packed.lengths, options.toNative(), options.int8Ranges());
            if (matrix == null || matrix.length != array.length) {
                throw new IllegalStateException(
                        "native embed returned " + (matrix == null ? "no" : matrix.length) + " rows for " + array.length
                                + " texts");
            }
            return matrix;
        } finally {
            invocationLock.unlock();
        }
    }

    @Override
    public byte[][] embedCodes(List<String> texts, TurboEmbedOptions options) {
        String[] array = validateStrings(texts, "texts");
        Objects.requireNonNull(options, "options");
        if (!options.quantization().codes()) {
            throw new IllegalArgumentException(options.quantization() + " makes F32 vectors: read them with embed");
        }
        invocationLock.lock();
        try {
            long session = requireOpen();
            Utf8.Packed packed = Utf8.pack(array, "texts");
            byte[][] codes = TurboJni.sessionEmbedCodes(
                    session, packed.bytes, packed.lengths, options.toNative(), options.int8Ranges());
            if (codes == null || codes.length != array.length) {
                throw new IllegalStateException(
                        "native embed returned " + (codes == null ? "no" : codes.length) + " rows for " + array.length
                                + " texts");
            }
            return codes;
        } finally {
            invocationLock.unlock();
        }
    }

    @Override
    public float[] rerank(String query, List<String> passages, TurboRerankOptions options) {
        Objects.requireNonNull(query, "query");
        String[] array = validateStrings(passages, "passages");
        Objects.requireNonNull(options, "options");
        invocationLock.lock();
        try {
            long session = requireOpen();
            Utf8.Packed packed = Utf8.pack(array, "passages");
            float[] scores = TurboJni.sessionRerank(
                    session, Utf8.bytes(query), packed.bytes, packed.lengths, options.toNative());
            if (scores == null || scores.length != array.length) {
                throw new IllegalStateException(
                        "native rerank returned " + (scores == null ? "no" : scores.length) + " scores for "
                                + array.length + " passages");
            }
            return scores;
        } finally {
            invocationLock.unlock();
        }
    }

    @Override
    public TurboModelInfo modelInfo() {
        invocationLock.lock();
        try {
            long session = requireOpen();
            int[] numbers = TurboJni.sessionInfo(session);
            byte[][] text = TurboJni.sessionInfoText(session);
            String[] strings = new String[text.length];
            for (int i = 0; i < text.length; i++) {
                strings[i] = new String(text[i], StandardCharsets.UTF_8);
            }
            return TurboModelInfo.fromNative(numbers, strings);
        } finally {
            invocationLock.unlock();
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
            TurboJni.sessionRelease(handle);
        } finally {
            invocationLock.unlock();
        }
    }
}
