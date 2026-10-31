package ai.pipestream.turboembed.panama;

import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * A lightweight, lifetime-bound view of one row in a {@link TurboAnalysisResult}.
 *
 * <p>Accessors read native memory on demand. Numeric ranges are returned as small immutable
 * coordinate records, while layer, label, and value strings decode only their own pool slices.
 * The native span array, pool, and graph are not copied. All accessors fail after the owning
 * result closes. Call {@link #materialize()} for an explicit copied snapshot.
 */
public final class TurboSpanView {
    private final TurboAnalysisResult owner;
    private final int index;

    TurboSpanView(TurboAnalysisResult owner, int index) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.index = index;
    }

    /** Returns the zero-based native row index within the owning result. */
    public int index() {
        return index;
    }

    /** Returns this row's typed layer family. */
    public TurboLayerKind layerKind() {
        return owner.layerKind(index);
    }

    /** Returns the item index within its layer instance. Fragments share an item index. */
    public long itemIndex() {
        return owner.itemIndex(index);
    }

    /** Returns the fragment index within its item. */
    public long fragmentIndex() {
        return owner.fragmentIndex(index);
    }

    /** Returns the native view id governing {@link #viewRange()}. Source is {@code 0}. */
    public long viewId() {
        return owner.viewId(index);
    }

    /** Returns the raw layer-specific classification. For PII use {@link #piiConfidence()}. */
    public long classification() {
        return owner.classification(index);
    }

    /** Returns the PII evidence classification, or empty for another layer kind. */
    public Optional<TurboPiiConfidence> piiConfidence() {
        if (layerKind() != TurboLayerKind.PII) {
            return Optional.empty();
        }
        return Optional.of(TurboPiiConfidence.fromNative(classification()));
    }

    /** Returns the range in {@link #viewId()}'s coordinate space, when present. */
    public Optional<TurboSpanData.CoordinateRange> viewRange() {
        return owner.viewRange(index);
    }

    /** Returns the checked projection to source coordinates, when present. */
    public Optional<TurboSpanData.CoordinateRange> sourceRange() {
        return owner.sourceRange(index);
    }

    /** Returns the half-open token range, when present. */
    public Optional<TurboSpanData.TokenRange> tokenRange() {
        return owner.tokenRange(index);
    }

    /** Returns the full native layer id, decoded from its pool slice. */
    public String layer() {
        return owner.layer(index);
    }

    /** Returns the named layer instance, or {@code ""} for the default instance. */
    public String instance() {
        return owner.instance(index);
    }

    /** Returns the annotation label, when the native row carries one. */
    public Optional<String> label() {
        return owner.label(index);
    }

    /** Returns the normalized value, when present (currently PII rows). */
    public Optional<String> value() {
        return owner.value(index);
    }

    /** Returns the model probability, when present. Rule rows do not invent one. */
    public OptionalDouble probability() {
        return owner.probability(index);
    }

    /**
     * Slices source text with this row's source UTF-16 range.
     *
     * @param sourceText document source text, not the text of a derived view
     * @return covered source text
     */
    public String coveredSourceText(CharSequence sourceText) {
        return sourceRange()
                .orElseThrow(() -> new IllegalStateException("span has no source range"))
                .coveredText(sourceText);
    }

    /**
     * Slices view text with this row's view UTF-16 range.
     *
     * @param viewText text of the view identified by {@link #viewId()}
     * @return covered view text
     */
    public String coveredViewText(CharSequence viewText) {
        return viewRange()
                .orElseThrow(() -> new IllegalStateException("span has no view range"))
                .coveredText(viewText);
    }

    /** Explicitly copies this row and only the strings it names. */
    public TurboSpanData materialize() {
        return owner.materialize(index);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof TurboSpanView that && owner == that.owner && index == that.index;
    }

    @Override
    public int hashCode() {
        return 31 * System.identityHashCode(owner) + index;
    }

    @Override
    public String toString() {
        return "TurboSpanView[index=" + index + "]";
    }
}
