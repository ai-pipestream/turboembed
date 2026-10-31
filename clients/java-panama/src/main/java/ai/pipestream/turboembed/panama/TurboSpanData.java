package ai.pipestream.turboembed.panama;

import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * A fully copied, immutable analysis span.
 *
 * <p>This is the explicit copy-out form of {@link TurboSpanView}. Creating it reads the row and
 * only the pool strings it names; it remains usable after its {@link TurboAnalysisResult} closes.
 * A field family is present only when the corresponding {@link Optional} is non-empty. In
 * particular, covered sentence and token text are not labels: read them from {@code sourceRange}
 * or {@code viewRange} so their coordinate space remains explicit.
 */
public record TurboSpanData(
        TurboLayerKind layerKind,
        long itemIndex,
        long fragmentIndex,
        long viewId,
        long classification,
        Optional<CoordinateRange> viewRange,
        Optional<CoordinateRange> sourceRange,
        Optional<TokenRange> tokenRange,
        String layer,
        String instance,
        Optional<String> label,
        Optional<String> value,
        OptionalDouble probability) {

    /** Validates copied span fields. */
    public TurboSpanData {
        Objects.requireNonNull(layerKind, "layerKind");
        Objects.requireNonNull(viewRange, "viewRange");
        Objects.requireNonNull(sourceRange, "sourceRange");
        Objects.requireNonNull(tokenRange, "tokenRange");
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(instance, "instance");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(probability, "probability");
        requireNonNegative(itemIndex, "itemIndex");
        requireNonNegative(fragmentIndex, "fragmentIndex");
        requireNonNegative(viewId, "viewId");
        requireNonNegative(classification, "classification");
        if (probability.isPresent()) {
            double p = probability.getAsDouble();
            if (!Double.isFinite(p) || p < 0.0 || p > 1.0) {
                throw new IllegalArgumentException("probability is outside 0..=1: " + p);
            }
        }
    }

    /**
     * Returns the PII evidence classification for a PII row.
     *
     * @return PII classification, or empty for another layer kind
     */
    public Optional<TurboPiiConfidence> piiConfidence() {
        return layerKind == TurboLayerKind.PII
                ? Optional.of(TurboPiiConfidence.fromNative(classification))
                : Optional.empty();
    }

    /**
     * Slices source text with this row's source UTF-16 range.
     *
     * @param sourceText the document source text, not a normalized view
     * @return covered source text
     */
    public String coveredSourceText(CharSequence sourceText) {
        return sourceRange
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
        return viewRange
                .orElseThrow(() -> new IllegalStateException("span has no view range"))
                .coveredText(viewText);
    }

    private static void requireNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " is negative: " + value);
        }
    }

    /** Half-open token-index range {@code [start, end)}. */
    public record TokenRange(long start, long end) {
        /** Validates this token range. */
        public TokenRange {
            requireNonNegative(start, "token start");
            if (end < start) {
                throw new IllegalArgumentException("token range ends before it starts: [" + start + ", " + end + ")");
            }
        }
    }

    /**
     * One text range in UTF-8 bytes, UTF-16 code units, and Unicode scalar values.
     *
     * <p>All three coordinate pairs describe the same half-open range in one named text: either
     * the source or the row's view. They must never be interchanged between texts.
     */
    public record CoordinateRange(
            long byteStart,
            long byteEnd,
            long utf16Start,
            long utf16End,
            long codepointStart,
            long codepointEnd) {

        /** Validates this coordinate range. */
        public CoordinateRange {
            requirePair(byteStart, byteEnd, "byte");
            requirePair(utf16Start, utf16End, "UTF-16");
            requirePair(codepointStart, codepointEnd, "codepoint");
        }

        /**
         * Slices {@code text} with this range's UTF-16 bounds.
         *
         * @param text text in this range's own coordinate space
         * @return covered text
         */
        public String coveredText(CharSequence text) {
            Objects.requireNonNull(text, "text");
            int start = checkedInt(utf16Start, "UTF-16 start");
            int end = checkedInt(utf16End, "UTF-16 end");
            if (end > text.length()) {
                throw new IllegalArgumentException(
                        "UTF-16 range [" + start + ", " + end + ") exceeds text length " + text.length());
            }
            return text.subSequence(start, end).toString();
        }

        private static void requirePair(long start, long end, String name) {
            requireNonNegative(start, name + " start");
            if (end < start) {
                throw new IllegalArgumentException(name + " range ends before it starts: [" + start + ", " + end + ")");
            }
        }

        private static int checkedInt(long value, String name) {
            return Math.toIntExact(value);
        }
    }
}
