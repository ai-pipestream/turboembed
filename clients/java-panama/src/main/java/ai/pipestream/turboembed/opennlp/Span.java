package ai.pipestream.turboembed.opennlp;

import java.util.Objects;

/**
 * Drop-in Apache OpenNLP {@code opennlp.tools.util.Span} equivalent backed by {@code libturbo}
 * offsets.
 *
 * <p>For character spans (such as {@link SentenceDetector#sentPosDetect(String)} and
 * {@link Tokenizer#tokenizePos(String)}), {@link #getStart()} and {@link #getEnd()} are UTF-16
 * code-unit offsets that match {@link String#substring(int, int)}, while {@link #getByteStart()},
 * {@link #getByteEnd()}, {@link #getCodepointStart()}, and {@link #getCodepointEnd()} provide the
 * exact UTF-8 byte and Unicode scalar offsets from {@code libturbo}.
 *
 * <p>For token-indexed spans (such as {@link TokenNameFinder#find(String[])}), {@link #getStart()}
 * and {@link #getEnd()} are token indices {@code [start, end)}.
 */
public final class Span implements Comparable<Span> {
    private final int start;
    private final int end;
    private final String type;
    private final double prob;
    private final int byteStart;
    private final int byteEnd;
    private final int codepointStart;
    private final int codepointEnd;

    /**
     * Constructs a span with start and end offsets.
     *
     * @param start inclusive start offset
     * @param end   exclusive end offset
     */
    public Span(int start, int end) {
        this(start, end, null, 0.0, start, end, start, end);
    }

    /**
     * Constructs a span with start, end, and type label.
     *
     * @param start inclusive start offset
     * @param end   exclusive end offset
     * @param type  span type (may be null)
     */
    public Span(int start, int end, String type) {
        this(start, end, type, 0.0, start, end, start, end);
    }

    /**
     * Constructs a span with start, end, type label, and probability.
     *
     * @param start inclusive start offset
     * @param end   exclusive end offset
     * @param type  span type (may be null)
     * @param prob  probability or confidence score
     */
    public Span(int start, int end, String type, double prob) {
        this(start, end, type, prob, start, end, start, end);
    }

    /**
     * Constructs a bi-directional span with UTF-16, UTF-8 byte, and Unicode codepoint offsets.
     *
     * @param start          inclusive UTF-16 code-unit or token start offset
     * @param end            exclusive UTF-16 code-unit or token end offset
     * @param type           span type (may be null)
     * @param prob           probability or confidence score
     * @param byteStart      inclusive UTF-8 byte start offset
     * @param byteEnd        exclusive UTF-8 byte end offset
     * @param codepointStart inclusive Unicode scalar start offset
     * @param codepointEnd   exclusive Unicode scalar end offset
     */
    public Span(
            int start,
            int end,
            String type,
            double prob,
            int byteStart,
            int byteEnd,
            int codepointStart,
            int codepointEnd) {
        if (start < 0 || end < start) {
            throw new IllegalArgumentException("Invalid span bounds: [" + start + ", " + end + ")");
        }
        this.start = start;
        this.end = end;
        this.type = type;
        this.prob = prob;
        this.byteStart = byteStart;
        this.byteEnd = byteEnd;
        this.codepointStart = codepointStart;
        this.codepointEnd = codepointEnd;
    }

    /**
     * Returns the inclusive start offset (UTF-16 code-unit index for character spans, or token
     * index for token spans).
     *
     * @return start offset
     */
    public int getStart() {
        return start;
    }

    /**
     * Returns the exclusive end offset (UTF-16 code-unit index for character spans, or token index
     * for token spans).
     *
     * @return end offset
     */
    public int getEnd() {
        return end;
    }

    /**
     * Returns the span type label, or {@code null} if untyped.
     *
     * @return span type or {@code null}
     */
    public String getType() {
        return type;
    }

    /**
     * Returns the probability or confidence score associated with this span.
     *
     * @return probability score
     */
    public double getProb() {
        return prob;
    }

    /**
     * Returns the length of this span ({@code getEnd() - getStart()}).
     *
     * @return span length
     */
    public int length() {
        return end - start;
    }

    /**
     * Returns the inclusive UTF-8 byte start offset into the source text.
     *
     * @return UTF-8 byte start offset
     */
    public int getByteStart() {
        return byteStart;
    }

    /**
     * Returns the exclusive UTF-8 byte end offset into the source text.
     *
     * @return UTF-8 byte end offset
     */
    public int getByteEnd() {
        return byteEnd;
    }

    /**
     * Returns the inclusive Unicode scalar (codepoint) start offset into the source text.
     *
     * @return Unicode codepoint start offset
     */
    public int getCodepointStart() {
        return codepointStart;
    }

    /**
     * Returns the exclusive Unicode scalar (codepoint) end offset into the source text.
     *
     * @return Unicode codepoint end offset
     */
    public int getCodepointEnd() {
        return codepointEnd;
    }

    /**
     * Returns the subsequence of {@code text} covered by {@code [getStart(), getEnd())}.
     *
     * @param text source character sequence
     * @return covered subsequence
     */
    public CharSequence getCoveredText(CharSequence text) {
        Objects.requireNonNull(text, "text");
        return text.subSequence(start, end);
    }

    @Override
    public int compareTo(Span other) {
        int c = Integer.compare(this.start, other.start);
        if (c != 0) {
            return c;
        }
        c = Integer.compare(this.end, other.end);
        if (c != 0) {
            return c;
        }
        if (this.type == null && other.type == null) {
            return 0;
        }
        if (this.type == null) {
            return -1;
        }
        if (other.type == null) {
            return 1;
        }
        return this.type.compareTo(other.type);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Span that)) {
            return false;
        }
        return start == that.start
                && end == that.end
                && Double.compare(prob, that.prob) == 0
                && Objects.equals(type, that.type);
    }

    @Override
    public int hashCode() {
        return Objects.hash(start, end, type, prob);
    }

    @Override
    public String toString() {
        if (type == null) {
            return "[" + start + ".." + end + ")";
        }
        return "[" + start + ".." + end + ") " + type;
    }
}
