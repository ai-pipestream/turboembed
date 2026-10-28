package ai.pipestream.turboembed.panama;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;

/**
 * Owner of one native {@code turbo_analysis_result} reference.
 *
 * <p>{@link TurboSpanView}s obtained from this object read the native span array and string pool
 * in place. Creating a result or a view does not copy either buffer, and reading one label does
 * not copy the rest of the pool. Every view is lifetime-bound to this owner: after {@link #close()}
 * its accessors throw {@link IllegalStateException}. Use {@link TurboSpanView#materialize()},
 * {@link #copySpans()}, or {@link #copyGraphBytes()} when data must outlive this result.
 *
 * <p>This class is safe for concurrent reads. {@code close()} waits for in-progress reads and
 * graph-segment callbacks, then releases exactly the one native reference owned by this object.
 * {@link #retain()} creates another independent owner of the same native result.
 */
public final class TurboAnalysisResult implements AutoCloseable {
    private static final long SPAN_STRIDE = TurboLibrary.TURBO_SPAN_LAYOUT.byteSize();

    private static final long LAYER_KIND_OFFSET = offset("layer_kind");
    private static final long FLAGS_OFFSET = offset("flags");
    private static final long ITEM_INDEX_OFFSET = offset("item_index");
    private static final long FRAGMENT_INDEX_OFFSET = offset("fragment_index");
    private static final long VIEW_ID_OFFSET = offset("view_id");
    private static final long CLASSIFICATION_OFFSET = offset("classification");
    private static final long VIEW_BYTE_START_OFFSET = offset("view_byte_start");
    private static final long VIEW_BYTE_END_OFFSET = offset("view_byte_end");
    private static final long VIEW_UTF16_START_OFFSET = offset("view_utf16_start");
    private static final long VIEW_UTF16_END_OFFSET = offset("view_utf16_end");
    private static final long VIEW_CODEPOINT_START_OFFSET = offset("view_codepoint_start");
    private static final long VIEW_CODEPOINT_END_OFFSET = offset("view_codepoint_end");
    private static final long SOURCE_BYTE_START_OFFSET = offset("source_byte_start");
    private static final long SOURCE_BYTE_END_OFFSET = offset("source_byte_end");
    private static final long SOURCE_UTF16_START_OFFSET = offset("source_utf16_start");
    private static final long SOURCE_UTF16_END_OFFSET = offset("source_utf16_end");
    private static final long SOURCE_CODEPOINT_START_OFFSET = offset("source_codepoint_start");
    private static final long SOURCE_CODEPOINT_END_OFFSET = offset("source_codepoint_end");
    private static final long TOKEN_START_OFFSET = offset("token_start");
    private static final long TOKEN_END_OFFSET = offset("token_end");
    private static final long LAYER_OFFSET_OFFSET = offset("layer_offset");
    private static final long LAYER_LEN_OFFSET = offset("layer_len");
    private static final long INSTANCE_OFFSET_OFFSET = offset("instance_offset");
    private static final long INSTANCE_LEN_OFFSET = offset("instance_len");
    private static final long LABEL_OFFSET_OFFSET = offset("label_offset");
    private static final long LABEL_LEN_OFFSET = offset("label_len");
    private static final long VALUE_OFFSET_OFFSET = offset("value_offset");
    private static final long VALUE_LEN_OFFSET = offset("value_len");
    private static final long PROB_OFFSET = offset("prob");

    static {
        if (SPAN_STRIDE != 120L || TurboLibrary.TURBO_SPAN_LAYOUT.byteAlignment() != 8L) {
            throw new ExceptionInInitializerError(
                    "turbo_span layout must be 120 bytes with alignment 8, got "
                            + SPAN_STRIDE
                            + "/"
                            + TurboLibrary.TURBO_SPAN_LAYOUT.byteAlignment());
        }
        if (PROB_OFFSET != 112L) {
            throw new ExceptionInInitializerError("turbo_span.prob must be at offset 112, got " + PROB_OFFSET);
        }
    }

    private final MemorySegment handle;
    private final MemorySegment spansSegment;
    private final int spanCount;
    private final MemorySegment poolSegment;
    private final MemorySegment graphSegment;
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock(true);
    private boolean closed;

    TurboAnalysisResult(MemorySegment handle) {
        MemorySegment adopted = Objects.requireNonNull(handle, "handle");
        if (MemorySegment.NULL.equals(adopted)) {
            throw new IllegalArgumentException("native analysis result handle is NULL");
        }

        MemorySegment spans;
        int count;
        MemorySegment pool;
        MemorySegment graph;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = TurboLibrary.allocateError(arena);

            MemorySegment spansPtrOut = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment countOut = arena.allocate(ValueLayout.JAVA_INT);
            int code = invokeSpans(adopted, spansPtrOut, countOut, err);
            TurboLibrary.checkStatus(code, err);
            long unsignedCount = Integer.toUnsignedLong(countOut.get(ValueLayout.JAVA_INT, 0L));
            if (unsignedCount > Integer.MAX_VALUE) {
                throw new IllegalStateException("native span count exceeds Java indexing: " + unsignedCount);
            }
            count = (int) unsignedCount;
            spans = boundedSegment(
                    spansPtrOut.get(ValueLayout.ADDRESS, 0L),
                    Math.multiplyExact(unsignedCount, SPAN_STRIDE),
                    "span array");

            MemorySegment poolPtrOut = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment poolLenOut = arena.allocate(ValueLayout.JAVA_LONG);
            code = invokeStrings(adopted, poolPtrOut, poolLenOut, err);
            TurboLibrary.checkStatus(code, err);
            pool = boundedSegment(
                    poolPtrOut.get(ValueLayout.ADDRESS, 0L),
                    checkedNativeLength(poolLenOut.get(ValueLayout.JAVA_LONG, 0L), "string pool"),
                    "string pool");

            MemorySegment graphPtrOut = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment graphLenOut = arena.allocate(ValueLayout.JAVA_LONG);
            code = invokeGraph(adopted, graphPtrOut, graphLenOut, err);
            TurboLibrary.checkStatus(code, err);
            graph = boundedSegment(
                    graphPtrOut.get(ValueLayout.ADDRESS, 0L),
                    checkedNativeLength(graphLenOut.get(ValueLayout.JAVA_LONG, 0L), "graph"),
                    "graph");
        } catch (RuntimeException | Error e) {
            try {
                releaseNative(adopted);
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            }
            throw e;
        }

        this.handle = adopted;
        this.spansSegment = spans;
        this.spanCount = count;
        this.poolSegment = pool;
        this.graphSegment = graph;
    }

    private TurboAnalysisResult(
            MemorySegment handle,
            MemorySegment spansSegment,
            int spanCount,
            MemorySegment poolSegment,
            MemorySegment graphSegment) {
        this.handle = handle;
        this.spansSegment = spansSegment;
        this.spanCount = spanCount;
        this.poolSegment = poolSegment;
        this.graphSegment = graphSegment;
    }

    private static long offset(String field) {
        return TurboLibrary.TURBO_SPAN_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement(field));
    }

    private static int invokeSpans(
            MemorySegment result, MemorySegment spansPtrOut, MemorySegment countOut, MemorySegment err) {
        try {
            return (int) TurboLibrary.TURBO_ANALYSIS_RESULT_SPANS.invokeExact(result, spansPtrOut, countOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_spans failed", t);
        }
    }

    private static int invokeStrings(
            MemorySegment result, MemorySegment poolPtrOut, MemorySegment poolLenOut, MemorySegment err) {
        try {
            return (int) TurboLibrary.TURBO_ANALYSIS_RESULT_STRINGS.invokeExact(result, poolPtrOut, poolLenOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_strings failed", t);
        }
    }

    private static int invokeGraph(
            MemorySegment result, MemorySegment graphPtrOut, MemorySegment graphLenOut, MemorySegment err) {
        try {
            return (int) TurboLibrary.TURBO_ANALYSIS_RESULT_GRAPH.invokeExact(result, graphPtrOut, graphLenOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_graph failed", t);
        }
    }

    private static long checkedNativeLength(long length, String what) {
        if (length < 0) {
            throw new IllegalStateException("native " + what + " length overflows a signed Java long: " + length);
        }
        return length;
    }

    private static MemorySegment boundedSegment(MemorySegment pointer, long length, String what) {
        if (length == 0L) {
            return MemorySegment.NULL;
        }
        if (pointer == null || MemorySegment.NULL.equals(pointer)) {
            throw new IllegalStateException("native " + what + " pointer is NULL for length " + length);
        }
        return pointer.reinterpret(length);
    }

    static void releaseNative(MemorySegment result) {
        if (result == null || MemorySegment.NULL.equals(result)) {
            return;
        }
        try {
            TurboLibrary.TURBO_ANALYSIS_RESULT_RELEASE.invokeExact(result);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_release failed", t);
        }
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
     * Returns a lifetime-bound view of one row without copying the span array or string pool.
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
            return List.copyOf(out);
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
            return List.copyOf(out);
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
            return List.copyOf(out);
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
            return List.copyOf(out);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Returns the graph byte length without copying the graph. */
    public long graphByteLength() {
        return graphSegment.byteSize();
    }

    /** Returns whether this result carries a non-empty graph buffer. */
    public boolean hasGraph() {
        return graphSegment.byteSize() > 0L;
    }

    /** Checks the four-byte {@code TGRF} graph magic without copying the graph. */
    public boolean hasGraphMagic() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return graphSegment.byteSize() >= 4L
                    && graphSegment.get(ValueLayout.JAVA_BYTE, 0L) == (byte) 'T'
                    && graphSegment.get(ValueLayout.JAVA_BYTE, 1L) == (byte) 'G'
                    && graphSegment.get(ValueLayout.JAVA_BYTE, 2L) == (byte) 'R'
                    && graphSegment.get(ValueLayout.JAVA_BYTE, 3L) == (byte) 'F';
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Runs {@code action} with the borrowed, read-only native span segment while this result
     * cannot close. Its layout is {@link TurboLibrary#TURBO_SPAN_LAYOUT}. Prefer
     * {@link TurboSpanView} unless a bulk native consumer needs the column directly.
     * The segment must not be retained after the callback returns.
     */
    public <T> T withSpansSegment(Function<? super MemorySegment, ? extends T> action) {
        Objects.requireNonNull(action, "action");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return action.apply(spansSegment.asReadOnly());
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Runs {@code action} with the borrowed, read-only native string-pool segment while this
     * result cannot close. Pool offsets are meaningful only together with native span rows.
     * The segment must not be retained after the callback returns.
     */
    public <T> T withStringPoolSegment(Function<? super MemorySegment, ? extends T> action) {
        Objects.requireNonNull(action, "action");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return action.apply(poolSegment.asReadOnly());
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Runs {@code action} with the borrowed graph segment while this result cannot close.
     * The segment must not be retained after the callback returns.
     */
    public <T> T withGraphSegment(Function<? super MemorySegment, ? extends T> action) {
        Objects.requireNonNull(action, "action");
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return action.apply(graphSegment.asReadOnly());
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Explicitly copies the graph bytes. The returned array remains usable after close. */
    public byte[] copyGraphBytes() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return graphSegment.toArray(ValueLayout.JAVA_BYTE);
        } finally {
            lifecycleLock.readLock().unlock();
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
            MemorySegment retained;
            try {
                retained = (MemorySegment) TurboLibrary.TURBO_ANALYSIS_RESULT_RETAIN.invokeExact(handle);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analysis_result_retain failed", t);
            }
            if (retained == null || MemorySegment.NULL.equals(retained)) {
                throw new IllegalStateException("turbo_analysis_result_retain returned NULL");
            }
            return new TurboAnalysisResult(retained, spansSegment, spanCount, poolSegment, graphSegment);
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
            releaseNative(handle);
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    TurboLayerKind layerKind(int index) {
        return readLocked(() -> layerKindRaw(index));
    }

    long itemIndex(int index) {
        return readLocked(() -> u32(index, ITEM_INDEX_OFFSET));
    }

    long fragmentIndex(int index) {
        return readLocked(() -> u32(index, FRAGMENT_INDEX_OFFSET));
    }

    long viewId(int index) {
        return readLocked(() -> u32(index, VIEW_ID_OFFSET));
    }

    long classification(int index) {
        return readLocked(() -> u32(index, CLASSIFICATION_OFFSET));
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

    private long base(int index) {
        checkIndex(index);
        return Math.multiplyExact((long) index, SPAN_STRIDE);
    }

    private long u32(int index, long fieldOffset) {
        return Integer.toUnsignedLong(spansSegment.get(ValueLayout.JAVA_INT, base(index) + fieldOffset));
    }

    private int flagsRaw(int index) {
        long flags = u32(index, FLAGS_OFFSET);
        if ((flags & ~TurboLibrary.TURBO_SPAN_KNOWN_FLAGS) != 0L) {
            throw new IllegalStateException("native span has unknown flags: " + flags);
        }
        return (int) flags;
    }

    private boolean hasFlag(int index, int flag) {
        return (flagsRaw(index) & flag) != 0;
    }

    private TurboLayerKind layerKindRaw(int index) {
        return TurboLayerKind.fromNative(u32(index, LAYER_KIND_OFFSET));
    }

    private TurboSpanData.CoordinateRange coordinateRangeRaw(
            int index, long byteStart, long byteEnd, long utf16Start, long utf16End, long cpStart, long cpEnd) {
        return new TurboSpanData.CoordinateRange(
                u32(index, byteStart),
                u32(index, byteEnd),
                u32(index, utf16Start),
                u32(index, utf16End),
                u32(index, cpStart),
                u32(index, cpEnd));
    }

    private void requireZeroRange(
            int index, String name, long byteStart, long byteEnd, long utf16Start, long utf16End, long cpStart, long cpEnd) {
        if (u32(index, byteStart) != 0
                || u32(index, byteEnd) != 0
                || u32(index, utf16Start) != 0
                || u32(index, utf16End) != 0
                || u32(index, cpStart) != 0
                || u32(index, cpEnd) != 0) {
            throw new IllegalStateException("native span has nonzero absent " + name + " range fields");
        }
    }

    private Optional<TurboSpanData.CoordinateRange> viewRangeRaw(int index) {
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_VIEW_RANGE)) {
            requireZeroRange(
                    index,
                    "view",
                    VIEW_BYTE_START_OFFSET,
                    VIEW_BYTE_END_OFFSET,
                    VIEW_UTF16_START_OFFSET,
                    VIEW_UTF16_END_OFFSET,
                    VIEW_CODEPOINT_START_OFFSET,
                    VIEW_CODEPOINT_END_OFFSET);
            return Optional.empty();
        }
        return Optional.of(coordinateRangeRaw(
                index,
                VIEW_BYTE_START_OFFSET,
                VIEW_BYTE_END_OFFSET,
                VIEW_UTF16_START_OFFSET,
                VIEW_UTF16_END_OFFSET,
                VIEW_CODEPOINT_START_OFFSET,
                VIEW_CODEPOINT_END_OFFSET));
    }

    private Optional<TurboSpanData.CoordinateRange> sourceRangeRaw(int index) {
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_SOURCE_RANGE)) {
            requireZeroRange(
                    index,
                    "source",
                    SOURCE_BYTE_START_OFFSET,
                    SOURCE_BYTE_END_OFFSET,
                    SOURCE_UTF16_START_OFFSET,
                    SOURCE_UTF16_END_OFFSET,
                    SOURCE_CODEPOINT_START_OFFSET,
                    SOURCE_CODEPOINT_END_OFFSET);
            return Optional.empty();
        }
        return Optional.of(coordinateRangeRaw(
                index,
                SOURCE_BYTE_START_OFFSET,
                SOURCE_BYTE_END_OFFSET,
                SOURCE_UTF16_START_OFFSET,
                SOURCE_UTF16_END_OFFSET,
                SOURCE_CODEPOINT_START_OFFSET,
                SOURCE_CODEPOINT_END_OFFSET));
    }

    private Optional<TurboSpanData.TokenRange> tokenRangeRaw(int index) {
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_TOKEN_RANGE)) {
            if (u32(index, TOKEN_START_OFFSET) != 0 || u32(index, TOKEN_END_OFFSET) != 0) {
                throw new IllegalStateException("native span has nonzero absent token range fields");
            }
            return Optional.empty();
        }
        return Optional.of(new TurboSpanData.TokenRange(
                u32(index, TOKEN_START_OFFSET), u32(index, TOKEN_END_OFFSET)));
    }

    private String poolString(long offset, long length, String what) {
        long end = Math.addExact(offset, length);
        if (end > poolSegment.byteSize()) {
            throw new IllegalStateException(
                    "native " + what + " pool slice [" + offset + ", " + end + ") exceeds pool length "
                            + poolSegment.byteSize());
        }
        if (length == 0L) {
            return "";
        }
        byte[] bytes = poolSegment.asSlice(offset, length).toArray(ValueLayout.JAVA_BYTE);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private String requiredPoolString(int index, long offsetField, long lengthField, String what) {
        return poolString(u32(index, offsetField), u32(index, lengthField), what);
    }

    private Optional<String> optionalPoolString(
            int index, int flag, long offsetField, long lengthField, String what) {
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
        return requiredPoolString(index, LAYER_OFFSET_OFFSET, LAYER_LEN_OFFSET, "layer");
    }

    private String instanceRaw(int index) {
        return requiredPoolString(index, INSTANCE_OFFSET_OFFSET, INSTANCE_LEN_OFFSET, "instance");
    }

    private Optional<String> labelRaw(int index) {
        return optionalPoolString(index, TurboLibrary.TURBO_SPAN_LABEL, LABEL_OFFSET_OFFSET, LABEL_LEN_OFFSET, "label");
    }

    private Optional<String> valueRaw(int index) {
        return optionalPoolString(index, TurboLibrary.TURBO_SPAN_VALUE, VALUE_OFFSET_OFFSET, VALUE_LEN_OFFSET, "value");
    }

    private OptionalDouble probabilityRaw(int index) {
        double prob = spansSegment.get(ValueLayout.JAVA_DOUBLE, base(index) + PROB_OFFSET);
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
                u32(index, ITEM_INDEX_OFFSET),
                u32(index, FRAGMENT_INDEX_OFFSET),
                u32(index, VIEW_ID_OFFSET),
                u32(index, CLASSIFICATION_OFFSET),
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
