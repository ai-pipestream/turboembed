package ai.pipestream.turboembed;

import java.util.List;

/** An owned result. Views expire when this owner closes; explicit copied values remain usable.
 * Panama reads native storage on demand; JNI copies rows and strings once at construction.
 * Retain creates a separate owner. No native buffer representation is part of this API. */
public interface TurboAnalysisResult extends AutoCloseable {
    /** Returns the number of native span rows. Fragmented items occupy more than one row. */
    int spanCount();

    /** Returns whether this owner has released its native reference. */
    boolean isClosed();

    /**
     * Returns a lifetime-bound view of one row.
     *
     * @param index zero-based native row index
     * @return row view owned by this result
     */
    TurboSpanView span(int index);

    /**
     * Returns a list facade over all rows. The facade creates lightweight views on access and
     * cannot provide row data after this result closes; it is not a copied snapshot.
     */
    List<? extends TurboSpanView> spans();

    /** Returns lightweight views for rows of one layer kind. */
    List<? extends TurboSpanView> spans(TurboLayerKind kind);

    /** Returns lightweight views for rows of one layer kind and named instance. */
    List<? extends TurboSpanView> spans(TurboLayerKind kind, String instance);

    /** Copies one row and only the strings it names. */
    TurboSpanData copySpan(int index);

    /** Explicitly copies every row. The returned data remains usable after close. */
    List<TurboSpanData> copySpans();

    /** Explicitly copies rows of one layer kind. */
    List<TurboSpanData> copySpans(TurboLayerKind kind);

    /** Returns the graph byte length without copying the graph. */
    long graphByteLength();

    /** Returns whether this result carries a non-empty graph buffer. */
    boolean hasGraph();

    /** Checks the four-byte {@code TGRF} graph magic. */
    boolean hasGraphMagic();

    /** Explicitly copies the graph bytes. The returned array remains usable after close. */
    byte[] copyGraphBytes();

    /**
     * Creates another owner retaining the same native result. Closing either owner releases
     * only its own reference and does not invalidate views belonging to the other owner.
     */
    TurboAnalysisResult retain();

    void close();
}
