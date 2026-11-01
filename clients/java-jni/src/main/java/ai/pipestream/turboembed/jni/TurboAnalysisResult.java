package ai.pipestream.turboembed.jni;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Owner of one native {@code turbo_analysis_result} reference.
 *
 * <p>When a result is made, its span rows and string pool cross from the native library into
 * Java arrays once; the graph crosses on first use. {@link TurboSpanView}s read those arrays and
 * decode one pool slice each. The native handle is kept only for {@link #retain()} and the
 * graph, and is released by {@link #close()}, after which every view's accessors throw
 * {@link IllegalStateException}. Use {@link TurboSpanView#materialize()}, {@link #copySpans()}
 * or {@link #copyGraphBytes()} when data must outlive this result.
 *
 * <p>This class is safe for concurrent reads. {@code close()} waits for in-progress reads, then
 * releases exactly the one native reference owned by this object. {@link #retain()} creates
 * another independent owner of the same native result.
 */
public final class TurboAnalysisResult implements AutoCloseable {
    private static final int ROW_INTS = TurboLibrary.TURBO_SPAN_INTS;

    private static final int LAYER_KIND = 0;
    private static final int FLAGS = 1;
    private static final int ITEM_INDEX = 2;
    private static final int FRAGMENT_INDEX = 3;
    private static final int VIEW_ID = 4;
    private static final int CLASSIFICATION = 5;
    private static final int VIEW_BYTE_START = 6;
    private static final int SOURCE_BYTE_START = 12;
    private static final int TOKEN_START = 18;
    private static final int TOKEN_END = 19;
    private static final int LAYER_OFFSET = 20;
    private static final int LAYER_LEN = 21;
    private static final int INSTANCE_OFFSET = 22;
    private static final int INSTANCE_LEN = 23;
    private static final int LABEL_OFFSET = 24;
    private static final int LABEL_LEN = 25;
    private static final int VALUE_OFFSET = 26;
    private static final int VALUE_LEN = 27;

    private final long handle;
    private final int[] rows;
    private final double[] probabilities;
    private final byte[] pool;
    private final int spanCount;
    private final long graphLength;
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock(true);
    private final Object graphLock = new Object();
    private byte[] graph;
    private boolean closed;

    TurboAnalysisResult(long handle) {
        if (handle == 0L) {
            throw new IllegalArgumentException("native analysis result handle is NULL");
        }
        int[] rows;
        double[] probabilities;
        byte[] pool;
        long graphLength;
        try {
            rows = Objects.requireNonNull(TurboJni.resultRows(handle), "rows");
            probabilities = Objects.requireNonNull(TurboJni.resultProbabilities(handle), "probabilities");
            pool = Objects.requireNonNull(TurboJni.resultStrings(handle), "pool");
            graphLength = TurboJni.resultGraphLength(handle);
            if (rows.length % ROW_INTS != 0 || probabilities.length != rows.length / ROW_INTS) {
                throw new IllegalStateException(
                        "native rows crossed as " + rows.length + " ints and " + probabilities.length + " probabilities");
            }
            if (graphLength < 0L) {
                throw new IllegalStateException("native graph length overflows a signed Java long: " + graphLength);
            }
        } catch (RuntimeException | Error e) {
            try {
                TurboJni.resultRelease(handle);
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            }
            throw e;
        }
        this.handle = handle;
        this.rows = rows;
        this.probabilities = probabilities;
        this.pool = pool;
        this.spanCount = rows.length / ROW_INTS;
        this.graphLength = graphLength;
    }

    private TurboAnalysisResult(
            long handle, int[] rows, double[] probabilities, byte[] pool, long graphLength, byte[] graph) {
        this.handle = handle;
        this.rows = rows;
        this.probabilities = probabilities;
        this.pool = pool;
        this.spanCount = rows.length / ROW_INTS;
        this.graphLength = graphLength;
        this.graph = graph;
    }

    /** Returns the number of native span rows. Fragmented items occupy more than one row. */
    public int spanCount() {
        return spanCount;
    }

    /** Returns whether this owner has released its native reference. */
    public boolean isClosed() {
        lifecycleLock.readLock().lock();
        try {
            return closed;
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Returns a lifetime-bound view of one row.
     *
     * @param index zero-based native row index
     * @return row view owned by this result
     */
    public TurboSpanView span(int index) {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            checkIndex(index);
            return new TurboSpanView(this, index);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Returns a list facade over all rows. The facade creates lightweight views on access and
     * becomes unusable when this result closes; it is not a copied snapshot.
     */
    public List<TurboSpanView> spans() {
        return new AbstractList<>() {
            @Override
            public TurboSpanView get(int index) {
                return span(index);
            }

            @Override
            public int size() {
                return spanCount;
            }
        };
    }

    /** Returns lightweight views for rows of one layer kind. */
    public List<TurboSpanView> spans(TurboLayerKind kind) {
        Objects.requireNonNull(kind, "kind");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            List<TurboSpanView> out = new ArrayList<>();
            for (int i = 0; i < spanCount; i++) {
                if (layerKindRaw(i) == kind) {
                    out.add(new TurboSpanView(this, i));
                }
            }
            return Collections.unmodifiableList(out);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Returns lightweight views for rows of one layer kind and named instance. */
    public List<TurboSpanView> spans(TurboLayerKind kind, String instance) {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(instance, "instance");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            List<TurboSpanView> out = new ArrayList<>();
            for (int i = 0; i < spanCount; i++) {
                if (layerKindRaw(i) == kind && instanceRaw(i).equals(instance)) {
                    out.add(new TurboSpanView(this, i));
                }
            }
            return Collections.unmodifiableList(out);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Copies one row and only the strings it names. */
    public TurboSpanData copySpan(int index) {
        return span(index).materialize();
    }

    /** Explicitly copies every row. The returned data remains usable after close. */
    public List<TurboSpanData> copySpans() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            List<TurboSpanData> out = new ArrayList<>(spanCount);
            for (int i = 0; i < spanCount; i++) {
                out.add(materializeRaw(i));
            }
            return Collections.unmodifiableList(out);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Explicitly copies rows of one layer kind. */
    public List<TurboSpanData> copySpans(TurboLayerKind kind) {
        Objects.requireNonNull(kind, "kind");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            List<TurboSpanData> out = new ArrayList<>();
            for (int i = 0; i < spanCount; i++) {
                if (layerKindRaw(i) == kind) {
                    out.add(materializeRaw(i));
                }
            }
            return Collections.unmodifiableList(out);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Returns the graph byte length without copying the graph. */
    public long graphByteLength() {
        return graphLength;
    }

    /** Returns whether this result carries a non-empty graph buffer. */
    public boolean hasGraph() {
        return graphLength > 0L;
    }

    /** Checks the four-byte {@code TGRF} graph magic. */
    public boolean hasGraphMagic() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            byte[] bytes = graphBytes();
            return bytes.length >= 4
                    && bytes[0] == (byte) 'T'
                    && bytes[1] == (byte) 'G'
                    && bytes[2] == (byte) 'R'
                    && bytes[3] == (byte) 'F';
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Explicitly copies the graph bytes. The returned array remains usable after close. */
    public byte[] copyGraphBytes() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return graphBytes().clone();
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** The graph, crossed from the native result on first use while this result is open. */
    private byte[] graphBytes() {
        synchronized (graphLock) {
            if (graph == null) {
                byte[] crossed = Objects.requireNonNull(TurboJni.resultGraph(handle), "graph");
                if (crossed.length != graphLength) {
                    throw new IllegalStateException(
                            "native graph crossed as " + crossed.length + " bytes, its length is " + graphLength);
                }
                graph = crossed;
            }
            return graph;
        }
    }

    /**
     * Creates another owner retaining the same native result. Closing either owner releases
     * only its own reference and does not invalidate views belonging to the other owner.
     */
    public TurboAnalysisResult retain() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            long retained = TurboJni.resultRetain(handle);
            if (retained == 0L) {
                throw new IllegalStateException("turbo_analysis_result_retain returned NULL");
            }
            byte[] crossed;
            synchronized (graphLock) {
                crossed = graph;
            }
            return new TurboAnalysisResult(retained, rows, probabilities, pool, graphLength, crossed);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public void close() {
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            TurboJni.resultRelease(handle);
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    TurboLayerKind layerKind(int index) {
        return readLocked(() -> layerKindRaw(index));
    }

    long itemIndex(int index) {
        return readLocked(() -> u32(index, ITEM_INDEX));
    }

    long fragmentIndex(int index) {
        return readLocked(() -> u32(index, FRAGMENT_INDEX));
    }

    long viewId(int index) {
        return readLocked(() -> u32(index, VIEW_ID));
    }

    long classification(int index) {
        return readLocked(() -> u32(index, CLASSIFICATION));
    }

    Optional<TurboSpanData.CoordinateRange> viewRange(int index) {
        return readLocked(() -> viewRangeRaw(index));
    }

    Optional<TurboSpanData.CoordinateRange> sourceRange(int index) {
        return readLocked(() -> sourceRangeRaw(index));
    }

    Optional<TurboSpanData.TokenRange> tokenRange(int index) {
        return readLocked(() -> tokenRangeRaw(index));
    }

    String layer(int index) {
        return readLocked(() -> layerRaw(index));
    }

    String instance(int index) {
        return readLocked(() -> instanceRaw(index));
    }

    Optional<String> label(int index) {
        return readLocked(() -> labelRaw(index));
    }

    Optional<String> value(int index) {
        return readLocked(() -> valueRaw(index));
    }

    OptionalDouble probability(int index) {
        return readLocked(() -> probabilityRaw(index));
    }

    TurboSpanData materialize(int index) {
        return readLocked(() -> materializeRaw(index));
    }

    private interface LockedRead<T> {
        T run();
    }

    private <T> T readLocked(LockedRead<T> read) {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return read.run();
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("TurboAnalysisResult is already closed");
        }
    }

    private void checkIndex(int index) {
        if (index < 0 || index >= spanCount) {
            throw new IndexOutOfBoundsException("span index " + index + " outside 0.." + spanCount);
        }
    }

    private long u32(int index, int field) {
        checkIndex(index);
        return Integer.toUnsignedLong(rows[index * ROW_INTS + field]);
    }

    private int flagsRaw(int index) {
        long flags = u32(index, FLAGS);
        if ((flags & ~TurboLibrary.TURBO_SPAN_KNOWN_FLAGS) != 0L) {
            throw new IllegalStateException("native span has unknown flags: " + flags);
        }
        return (int) flags;
    }

    private boolean hasFlag(int index, int flag) {
        return (flagsRaw(index) & flag) != 0;
    }

    private TurboLayerKind layerKindRaw(int index) {
        return TurboLayerKind.fromNative(u32(index, LAYER_KIND));
    }

    /** A range of six fields from {@code first}: byte, UTF-16 and codepoint start and end. */
    private Optional<TurboSpanData.CoordinateRange> rangeRaw(int index, int flag, int first, String name) {
        if (!hasFlag(index, flag)) {
            for (int field = first; field < first + 6; field++) {
                if (u32(index, field) != 0L) {
                    throw new IllegalStateException("native span has nonzero absent " + name + " range fields");
                }
            }
            return Optional.empty();
        }
        return Optional.of(new TurboSpanData.CoordinateRange(
                u32(index, first),
                u32(index, first + 1),
                u32(index, first + 2),
                u32(index, first + 3),
                u32(index, first + 4),
                u32(index, first + 5)));
    }

    private Optional<TurboSpanData.CoordinateRange> viewRangeRaw(int index) {
        return rangeRaw(index, TurboLibrary.TURBO_SPAN_VIEW_RANGE, VIEW_BYTE_START, "view");
    }

    private Optional<TurboSpanData.CoordinateRange> sourceRangeRaw(int index) {
        return rangeRaw(index, TurboLibrary.TURBO_SPAN_SOURCE_RANGE, SOURCE_BYTE_START, "source");
    }

    private Optional<TurboSpanData.TokenRange> tokenRangeRaw(int index) {
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_TOKEN_RANGE)) {
            if (u32(index, TOKEN_START) != 0 || u32(index, TOKEN_END) != 0) {
                throw new IllegalStateException("native span has nonzero absent token range fields");
            }
            return Optional.empty();
        }
        return Optional.of(new TurboSpanData.TokenRange(u32(index, TOKEN_START), u32(index, TOKEN_END)));
    }

    private String poolString(long offset, long length, String what) {
        long end = Math.addExact(offset, length);
        if (end > pool.length) {
            throw new IllegalStateException(
                    "native " + what + " pool slice [" + offset + ", " + end + ") exceeds pool length " + pool.length);
        }
        if (length == 0L) {
            return "";
        }
        return Utf8.decode(pool, (int) offset, (int) length);
    }

    private String requiredPoolString(int index, int offsetField, int lengthField, String what) {
        return poolString(u32(index, offsetField), u32(index, lengthField), what);
    }

    private Optional<String> optionalPoolString(int index, int flag, int offsetField, int lengthField, String what) {
        long offset = u32(index, offsetField);
        long length = u32(index, lengthField);
        if (!hasFlag(index, flag)) {
            if (offset != 0L || length != 0L) {
                throw new IllegalStateException("native span has nonzero absent " + what + " pool fields");
            }
            return Optional.empty();
        }
        return Optional.of(poolString(offset, length, what));
    }

    private String layerRaw(int index) {
        return requiredPoolString(index, LAYER_OFFSET, LAYER_LEN, "layer");
    }

    private String instanceRaw(int index) {
        return requiredPoolString(index, INSTANCE_OFFSET, INSTANCE_LEN, "instance");
    }

    private Optional<String> labelRaw(int index) {
        return optionalPoolString(index, TurboLibrary.TURBO_SPAN_LABEL, LABEL_OFFSET, LABEL_LEN, "label");
    }

    private Optional<String> valueRaw(int index) {
        return optionalPoolString(index, TurboLibrary.TURBO_SPAN_VALUE, VALUE_OFFSET, VALUE_LEN, "value");
    }

    private OptionalDouble probabilityRaw(int index) {
        checkIndex(index);
        double prob = probabilities[index];
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_PROBABILITY)) {
            if (!Double.isNaN(prob)) {
                throw new IllegalStateException("native span has a probability value without its presence flag");
            }
            return OptionalDouble.empty();
        }
        if (!Double.isFinite(prob) || prob < 0.0 || prob > 1.0) {
            throw new IllegalStateException("native span probability is outside 0..=1: " + prob);
        }
        return OptionalDouble.of(prob);
    }

    private TurboSpanData materializeRaw(int index) {
        return new TurboSpanData(
                layerKindRaw(index),
                u32(index, ITEM_INDEX),
                u32(index, FRAGMENT_INDEX),
                u32(index, VIEW_ID),
                u32(index, CLASSIFICATION),
                viewRangeRaw(index),
                sourceRangeRaw(index),
                tokenRangeRaw(index),
                layerRaw(index),
                instanceRaw(index),
                labelRaw(index),
                valueRaw(index),
                probabilityRaw(index));
    }
}
