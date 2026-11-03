package ai.pipestream.turboembed;

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/** A lifetime-bound row view, with explicit source, view and token coordinate spaces.
 * Accessors requiring row data fail after the owning result closes. Materialize for a snapshot. */
public interface TurboSpanView {
    /** Returns the zero-based native row index within the owning result. */
    int index();

    /** Returns this row's typed layer family. */
    TurboLayerKind layerKind();

    /** Returns the item index within its layer instance. Fragments share an item index. */
    long itemIndex();

    /** Returns the fragment index within its item. */
    long fragmentIndex();

    /** Returns the native view id governing {@link #viewRange()}. Source is {@code 0}. */
    long viewId();

    /**
     * Returns the raw layer-specific classification. For PII use {@link #piiConfidence()}; for a
     * dependency row use {@link #dependencyHead()}.
     */
    long classification();

    /**
     * Returns the head token index of a dependency row, or empty for the root of its sentence and
     * for another layer kind. The relation is the row's label.
     */
    OptionalLong dependencyHead();

    /** Returns the PII evidence classification, or empty for another layer kind. */
    Optional<TurboPiiConfidence> piiConfidence();

    /** Returns the range in {@link #viewId()}'s coordinate space, when present. */
    Optional<TurboSpanData.CoordinateRange> viewRange();

    /** Returns the checked projection to source coordinates, when present. */
    Optional<TurboSpanData.CoordinateRange> sourceRange();

    /** Returns the half-open token range, when present. */
    Optional<TurboSpanData.TokenRange> tokenRange();

    /** Returns the full native layer id, decoded from its pool slice. */
    String layer();

    /** Returns the named layer instance, or {@code ""} for the default instance. */
    String instance();

    /** Returns the annotation label, when the native row carries one. */
    Optional<String> label();

    /** Returns the normalized value, when present (currently PII rows). */
    Optional<String> value();

    /** Returns the model probability, when present. Rule rows do not invent one. */
    OptionalDouble probability();

    /**
     * Slices source text with this row's source UTF-16 range.
     *
     * @param sourceText document source text, not the text of a derived view
     * @return covered source text
     */
    String coveredSourceText(CharSequence sourceText);

    /**
     * Slices view text with this row's view UTF-16 range.
     *
     * @param viewText text of the view identified by {@link #viewId()}
     * @return covered view text
     */
    String coveredViewText(CharSequence viewText);

    /** Explicitly copies this row and only the strings it names. */
    TurboSpanData materialize();
}
