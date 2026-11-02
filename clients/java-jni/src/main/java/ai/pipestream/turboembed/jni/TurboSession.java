package ai.pipestream.turboembed.jni;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process model session over {@code libturbo} for dense embedding and cross-encoder reranking
 * through the Java Native Interface. The native library opens a runtime, a context on the CPU
 * device (device 0), the bundle and one session over it, and releases them together on
 * {@link #close()}.
 */
public final class TurboSession implements AutoCloseable {
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
        Objects.requireNonNull(bundleDir, "bundleDir");
        long handle = TurboJni.sessionOpen(Utf8.bytes(bundleDir.toString()));
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

    /**
     * Embeds a batch of UTF-8 texts and returns one {@code float[dim]} vector per input text.
     *
     * @param texts non-empty list of texts to embed
     * @return {@code float[texts.size()][dim]} embedding matrix
     */
    public float[][] embed(List<String> texts) {
        String[] array = validateStrings(texts, "texts");
        invocationLock.lock();
        try {
            long session = requireOpen();
            Utf8.Packed packed = Utf8.pack(array, "texts");
            float[][] matrix = TurboJni.sessionEmbed(session, packed.bytes, packed.lengths);
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

    /**
     * Reranks {@code passages} against {@code query} and returns raw relevance logits.
     *
     * @param query    query text
     * @param passages candidate passage texts
     * @return one relevance score per passage
     */
    public float[] rerank(String query, List<String> passages) {
        Objects.requireNonNull(query, "query");
        String[] array = validateStrings(passages, "passages");
        invocationLock.lock();
        try {
            long session = requireOpen();
            Utf8.Packed packed = Utf8.pack(array, "passages");
            float[] scores = TurboJni.sessionRerank(session, Utf8.bytes(query), packed.bytes, packed.lengths);
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
