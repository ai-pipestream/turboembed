package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboLayerKind;
import ai.pipestream.turboembed.TurboPiiCandidate;
import ai.pipestream.turboembed.TurboPiiDetector;
import ai.pipestream.turboembed.TurboPiiDisposition;
import ai.pipestream.turboembed.TurboPiiEvidence;
import ai.pipestream.turboembed.TurboPiiScore;
import ai.pipestream.turboembed.TurboSpanData;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;
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
public final class TurboAnalysisResult implements ai.pipestream.turboembed.TurboAnalysisResult {
    private static final int ROW_INTS = TurboLibrary.TURBO_SPAN_INTS;
    private static final int PII_SPAN_INTS = TurboLibrary.TURBO_PII_SPAN_INTS;
    private static final int PII_FACT_INTS = TurboLibrary.TURBO_PII_FACT_INTS;
    private static final int PII_CANDIDATE_INTS = TurboLibrary.TURBO_PII_CANDIDATE_INTS;

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
    private final int[] piiSpans;
    private final int[] piiFacts;
    private final int[] piiCandidates;
    private final int[] piiRowBySpanIndex;
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
        int[] piiSpans;
        int[] piiFacts;
        int[] piiCandidates;
        int[] piiRowBySpanIndex;
        byte[] pool;
        long graphLength;
        try {
            rows = Objects.requireNonNull(TurboJni.resultRows(handle), "rows");
            probabilities = Objects.requireNonNull(TurboJni.resultProbabilities(handle), "probabilities");
            piiSpans = Objects.requireNonNull(TurboJni.resultPiiSpans(handle), "piiSpans");
            piiFacts = Objects.requireNonNull(TurboJni.resultPiiFacts(handle), "piiFacts");
            piiCandidates = Objects.requireNonNull(TurboJni.resultPiiCandidates(handle), "piiCandidates");
            pool = Objects.requireNonNull(TurboJni.resultStrings(handle), "pool");
            graphLength = TurboJni.resultGraphLength(handle);
            if (rows.length % ROW_INTS != 0 || probabilities.length != rows.length / ROW_INTS) {
                throw new IllegalStateException(
                        "native rows crossed as " + rows.length + " ints and " + probabilities.length + " probabilities");
            }
            if (piiSpans.length % PII_SPAN_INTS != 0
                    || piiFacts.length % PII_FACT_INTS != 0
                    || piiCandidates.length % PII_CANDIDATE_INTS != 0) {
                throw new IllegalStateException("native PII arrays have misaligned lengths");
            }
            if (graphLength < 0L) {
                throw new IllegalStateException("native graph length overflows a signed Java long: " + graphLength);
            }
            int count = rows.length / ROW_INTS;
            piiRowBySpanIndex = new int[count];
            Arrays.fill(piiRowBySpanIndex, -1);
            int piiCount = piiSpans.length / PII_SPAN_INTS;
            for (int r = 0; r < piiCount; r++) {
                long spanIdx = Integer.toUnsignedLong(piiSpans[r * PII_SPAN_INTS]);
                if (spanIdx >= count) {
                    throw new IllegalStateException("native turbo_pii_span span_index out of bounds: " + spanIdx);
                }
                piiRowBySpanIndex[(int) spanIdx] = r;
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
        this.piiSpans = piiSpans;
        this.piiFacts = piiFacts;
        this.piiCandidates = piiCandidates;
        this.piiRowBySpanIndex = piiRowBySpanIndex;
        this.pool = pool;
        this.spanCount = rows.length / ROW_INTS;
        this.graphLength = graphLength;
    }

    private TurboAnalysisResult(
            long handle,
            int[] rows,
            double[] probabilities,
            int[] piiSpans,
            int[] piiFacts,
            int[] piiCandidates,
            int[] piiRowBySpanIndex,
            byte[] pool,
            long graphLength,
            byte[] graph) {
        this.handle = handle;
        this.rows = rows;
        this.probabilities = probabilities;
        this.piiSpans = piiSpans;
        this.piiFacts = piiFacts;
        this.piiCandidates = piiCandidates;
        this.piiRowBySpanIndex = piiRowBySpanIndex;
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
            return new TurboAnalysisResult(
                    retained,
                    rows,
                    probabilities,
                    piiSpans,
                    piiFacts,
                    piiCandidates,
                    piiRowBySpanIndex,
                    pool,
                    graphLength,
                    crossed);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public List<TurboPiiCandidate> piiCandidates() {
        return readLocked(this::piiCandidatesRaw);
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

    OptionalLong piiKindCode(int index) {
        return readLocked(() -> piiKindCodeRaw(index));
    }

    Optional<TurboPiiDetector> piiDetector(int index) {
        return readLocked(() -> piiDetectorRaw(index));
    }

    List<TurboPiiEvidence> piiEvidence(int index) {
        return readLocked(() -> piiEvidenceRaw(index));
    }

    Optional<TurboPiiScore> piiScore(int index) {
        return readLocked(() -> piiScoreRaw(index));
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
        if (piiKindCodeRaw(index).isPresent()) {
            return Optional.empty();
        }
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_TOKEN_RANGE)) {
            if (u32(index, TOKEN_START) != 0 || u32(index, TOKEN_END) != 0) {
                throw new IllegalStateException("native span has nonzero absent token range fields");
            }
            return Optional.empty();
        }
        return Optional.of(new TurboSpanData.TokenRange(u32(index, TOKEN_START), u32(index, TOKEN_END)));
    }

    /** A PII row's numeric kind, read from token_start under TURBO_SPAN_PII_KIND (#1940). */
    private OptionalLong piiKindCodeRaw(int index) {
        if (!hasFlag(index, TurboLibrary.TURBO_SPAN_PII_KIND)) {
            return OptionalLong.empty();
        }
        if (hasFlag(index, TurboLibrary.TURBO_SPAN_TOKEN_RANGE) || layerKindRaw(index) != TurboLayerKind.PII) {
            throw new IllegalStateException("native span has a PII kind on a row that is not a PII row");
        }
        if (u32(index, TOKEN_END) != 0L) {
            throw new IllegalStateException("native span has a nonzero token end with its PII kind");
        }
        return OptionalLong.of(u32(index, TOKEN_START));
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
                probabilityRaw(index),
                piiKindCodeRaw(index),
                piiDetectorRaw(index),
                piiEvidenceRaw(index),
                piiScoreRaw(index));
    }

    private Optional<TurboPiiDetector> piiDetectorRaw(int index) {
        checkIndex(index);
        int r = piiRowBySpanIndex[index];
        if (r < 0) {
            return Optional.empty();
        }
        int base = r * PII_SPAN_INTS;
        return Optional.of(decodeDetector(piiSpans, base + 1, base + 2, base + 3, base + 4));
    }

    private List<TurboPiiEvidence> piiEvidenceRaw(int index) {
        checkIndex(index);
        int r = piiRowBySpanIndex[index];
        if (r < 0) {
            return List.of();
        }
        int base = r * PII_SPAN_INTS;
        return decodeFacts(
                Integer.toUnsignedLong(piiSpans[base + 5]), Integer.toUnsignedLong(piiSpans[base + 6]));
    }

    private Optional<TurboPiiScore> piiScoreRaw(int index) {
        checkIndex(index);
        int r = piiRowBySpanIndex[index];
        if (r < 0) {
            return Optional.empty();
        }
        int base = r * PII_SPAN_INTS;
        return decodeScore(piiSpans, base + 7);
    }

    private TurboPiiDetector decodeDetector(int[] words, int idOff, int idLen, int verOff, int verLen) {
        String id = poolString(
                Integer.toUnsignedLong(words[idOff]), Integer.toUnsignedLong(words[idLen]), "detector_id");
        String version = poolString(
                Integer.toUnsignedLong(words[verOff]), Integer.toUnsignedLong(words[verLen]), "detector_version");
        return new TurboPiiDetector(id, version);
    }

    private List<TurboPiiEvidence> decodeFacts(long factStart, long factEnd) {
        long factTotal = piiFacts.length / PII_FACT_INTS;
        if (factEnd < factStart || factEnd > factTotal) {
            throw new IllegalStateException("native PII fact slice [" + factStart + ", " + factEnd + ") out of bounds");
        }
        int count = Math.toIntExact(factEnd - factStart);
        if (count == 0) {
            return List.of();
        }
        List<TurboPiiEvidence> out = new ArrayList<>(count);
        for (long i = factStart; i < factEnd; i++) {
            int base = Math.toIntExact(i) * PII_FACT_INTS;
            out.add(TurboPiiEvidence.fromNative(
                    Integer.toUnsignedLong(piiFacts[base]),
                    Integer.toUnsignedLong(piiFacts[base + 1]),
                    Integer.toUnsignedLong(piiFacts[base + 2]),
                    Integer.toUnsignedLong(piiFacts[base + 3])));
        }
        return Collections.unmodifiableList(out);
    }

    private Optional<TurboPiiScore> decodeScore(int[] words, int scoreKindField) {
        long scoreKind = Integer.toUnsignedLong(words[scoreKindField]);
        long calibrated = Integer.toUnsignedLong(words[scoreKindField + 1]);
        long modelIdOff = Integer.toUnsignedLong(words[scoreKindField + 2]);
        long modelIdLen = Integer.toUnsignedLong(words[scoreKindField + 3]);
        long modelVerOff = Integer.toUnsignedLong(words[scoreKindField + 4]);
        long modelVerLen = Integer.toUnsignedLong(words[scoreKindField + 5]);
        float value = Float.intBitsToFloat(words[scoreKindField + 6]);
        if (scoreKind == 0L) {
            if (!Float.isNaN(value) || calibrated != 0L || modelIdOff != 0L || modelIdLen != 0L) {
                throw new IllegalStateException("native PII score has nonzero fields when score_kind is 0");
            }
            return Optional.empty();
        }
        if (calibrated != 0L && calibrated != 1L) {
            throw new IllegalStateException("native PII score_calibrated must be 0 or 1: " + calibrated);
        }
        TurboPiiDetector model = new TurboPiiDetector(
                poolString(modelIdOff, modelIdLen, "score_model_id"),
                poolString(modelVerOff, modelVerLen, "score_model_version"));
        return Optional.of(new TurboPiiScore(
                value, TurboPiiScore.Kind.fromNative(scoreKind), model, calibrated == 1L));
    }

    private List<TurboPiiCandidate> piiCandidatesRaw() {
        int candCount = piiCandidates.length / PII_CANDIDATE_INTS;
        if (candCount == 0) {
            return List.of();
        }
        List<TurboPiiCandidate> out = new ArrayList<>(candCount);
        for (int i = 0; i < candCount; i++) {
            int base = i * PII_CANDIDATE_INTS;
            long itemIndex = Integer.toUnsignedLong(piiCandidates[base]);
            long fragmentIndex = Integer.toUnsignedLong(piiCandidates[base + 1]);
            long viewId = Integer.toUnsignedLong(piiCandidates[base + 2]);
            long kindCode = Integer.toUnsignedLong(piiCandidates[base + 3]);
            TurboSpanData.CoordinateRange viewRange = new TurboSpanData.CoordinateRange(
                    Integer.toUnsignedLong(piiCandidates[base + 4]),
                    Integer.toUnsignedLong(piiCandidates[base + 5]),
                    Integer.toUnsignedLong(piiCandidates[base + 6]),
                    Integer.toUnsignedLong(piiCandidates[base + 7]),
                    Integer.toUnsignedLong(piiCandidates[base + 8]),
                    Integer.toUnsignedLong(piiCandidates[base + 9]));
            TurboSpanData.CoordinateRange sourceRange = new TurboSpanData.CoordinateRange(
                    Integer.toUnsignedLong(piiCandidates[base + 10]),
                    Integer.toUnsignedLong(piiCandidates[base + 11]),
                    Integer.toUnsignedLong(piiCandidates[base + 12]),
                    Integer.toUnsignedLong(piiCandidates[base + 13]),
                    Integer.toUnsignedLong(piiCandidates[base + 14]),
                    Integer.toUnsignedLong(piiCandidates[base + 15]));
            String instance = poolString(
                    Integer.toUnsignedLong(piiCandidates[base + 16]),
                    Integer.toUnsignedLong(piiCandidates[base + 17]),
                    "candidate_instance");
            String piiInstance = poolString(
                    Integer.toUnsignedLong(piiCandidates[base + 18]),
                    Integer.toUnsignedLong(piiCandidates[base + 19]),
                    "candidate_pii_instance");
            String label = poolString(
                    Integer.toUnsignedLong(piiCandidates[base + 20]),
                    Integer.toUnsignedLong(piiCandidates[base + 21]),
                    "candidate_label");
            TurboPiiDetector detector =
                    decodeDetector(piiCandidates, base + 22, base + 23, base + 24, base + 25);
            List<TurboPiiEvidence> evidence = decodeFacts(
                    Integer.toUnsignedLong(piiCandidates[base + 26]),
                    Integer.toUnsignedLong(piiCandidates[base + 27]));
            TurboPiiDisposition disposition = TurboPiiDisposition.fromNative(
                    Integer.toUnsignedLong(piiCandidates[base + 28]),
                    Integer.toUnsignedLong(piiCandidates[base + 29]),
                    Integer.toUnsignedLong(piiCandidates[base + 30]),
                    Integer.toUnsignedLong(piiCandidates[base + 31]));
            Optional<TurboPiiScore> score = decodeScore(piiCandidates, base + 32);
            out.add(new TurboPiiCandidate(
                    instance,
                    piiInstance,
                    itemIndex,
                    fragmentIndex,
                    viewId,
                    viewRange,
                    sourceRange,
                    kindCode,
                    label,
                    detector,
                    evidence,
                    score,
                    disposition));
        }
        return Collections.unmodifiableList(out);
    }
}
