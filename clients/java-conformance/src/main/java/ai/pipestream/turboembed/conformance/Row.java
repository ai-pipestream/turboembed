package ai.pipestream.turboembed.conformance;

import java.util.Objects;

/**
 * One analysis row as every binding reports it: the native {@code turbo_span} fields a caller
 * reads, with each optional family null when its flag is clear. Two bindings conform when they
 * produce equal rows for the same input.
 *
 * @param layerKind      the native {@code TURBO_ANALYSIS_LAYER_*} code
 * @param itemIndex      the item within its layer instance
 * @param fragmentIndex  the fragment within its item
 * @param viewId         the view the view range is in; the source is 0
 * @param classification the layer-specific classification
 * @param viewRange      the range in the view's coordinates, or null
 * @param sourceRange    the checked projection to the source, or null
 * @param tokenRange     the half-open token range, or null
 * @param layer          the full layer id
 * @param instance       the layer instance, {@code ""} for the default
 * @param label          the label, or null
 * @param value          the normalized value, or null
 * @param probability    the probability, or null
 */
public record Row(
        int layerKind,
        long itemIndex,
        long fragmentIndex,
        long viewId,
        long classification,
        Range viewRange,
        Range sourceRange,
        Tokens tokenRange,
        String layer,
        String instance,
        String label,
        String value,
        Double probability) {

    /** Validates the required fields. */
    public Row {
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(instance, "instance");
    }

    /** One text range in UTF-8 bytes, UTF-16 code units and Unicode scalar values. */
    public record Range(
            long byteStart, long byteEnd, long utf16Start, long utf16End, long codepointStart, long codepointEnd) {

        /** The text this range covers, by its UTF-16 bounds. */
        public String covered(String text) {
            return text.substring(Math.toIntExact(utf16Start), Math.toIntExact(utf16End));
        }
    }

    /** A half-open token index range. */
    public record Tokens(long start, long end) {}

    /** The source text this row covers, by its source UTF-16 bounds. */
    public String coveredSourceText(String text) {
        if (sourceRange == null) {
            throw new IllegalStateException("row has no source range");
        }
        return sourceRange.covered(text);
    }
}
