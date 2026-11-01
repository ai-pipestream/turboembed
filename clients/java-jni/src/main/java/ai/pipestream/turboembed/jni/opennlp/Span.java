package ai.pipestream.turboembed.jni.opennlp;

import ai.pipestream.turboembed.jni.TurboSpanData;
import ai.pipestream.turboembed.jni.TurboSpanView;
import java.util.Arrays;
import java.util.Objects;

/**
 * An OpenNLP-style span in this project's namespace.
 *
 * <p>This is not {@code opennlp.tools.util.Span}. Unlike an ambiguous pair of integers, every
 * instance names its coordinate space. A {@link CoordinateSpace#UTF16} span's start/end are
 * source UTF-16 code-unit offsets and carries the corresponding source byte and codepoint
 * offsets. A {@link CoordinateSpace#TOKEN} span's start/end are token indices and has no
 * character offsets; its byte and codepoint accessors fail instead of reinterpreting token
 * indices as text offsets.
 */
public final class Span implements Comparable<Span> {
    /** Coordinate space governing {@link #getStart()} and {@link #getEnd()}. */
    public enum CoordinateSpace {
        /** Source UTF-16 code-unit offsets. */
        UTF16,
        /** Token indices. */
        TOKEN
    }

    private final CoordinateSpace coordinateSpace;
    private final int start;
    private final int end;
    private final String type;
    private final double prob;
    private final long byteStart;
    private final long byteEnd;
    private final long codepointStart;
    private final long codepointEnd;
    private final long tokenStart;
    private final long tokenEnd;
    private final String layer;
    private final String instance;

    private Span(
            CoordinateSpace coordinateSpace,
            int start,
            int end,
            String type,
            double prob,
            long byteStart,
            long byteEnd,
            long codepointStart,
            long codepointEnd,
            long tokenStart,
            long tokenEnd,
            String layer,
            String instance) {
        this.coordinateSpace = Objects.requireNonNull(coordinateSpace, "coordinateSpace");
        if (start < 0 || end < start) {
            throw new IllegalArgumentException("Invalid span bounds: [" + start + ", " + end + ")");
        }
        if (!Double.isNaN(prob) && (!Double.isFinite(prob) || prob < 0.0 || prob > 1.0)) {
            throw new IllegalArgumentException("probability is outside 0..=1: " + prob);
        }
        if (coordinateSpace == CoordinateSpace.UTF16) {
            requirePair(byteStart, byteEnd, "byte");
            requirePair(codepointStart, codepointEnd, "codepoint");
        } else if (byteStart != -1L || byteEnd != -1L || codepointStart != -1L || codepointEnd != -1L) {
            throw new IllegalArgumentException("a token span cannot carry character offsets");
        }
        if (tokenStart != -1L || tokenEnd != -1L) {
            requirePair(tokenStart, tokenEnd, "token");
        }
        if (coordinateSpace == CoordinateSpace.TOKEN && (tokenStart != start || tokenEnd != end)) {
            throw new IllegalArgumentException("a token span's token range must equal its start/end");
        }
        this.start = start;
        this.end = end;
        this.type = type;
        this.prob = prob;
        this.byteStart = byteStart;
        this.byteEnd = byteEnd;
        this.codepointStart = codepointStart;
        this.codepointEnd = codepointEnd;
        this.tokenStart = tokenStart;
        this.tokenEnd = tokenEnd;
        this.layer = layer;
        this.instance = instance;
    }

    /**
     * Creates a source UTF-16 span with its corresponding source byte and codepoint ranges.
     * Use {@link Double#NaN} when no probability exists; probability is never defaulted to zero.
     */
    public static Span utf16(
            int start,
            int end,
            String type,
            double prob,
            long byteStart,
            long byteEnd,
            long codepointStart,
            long codepointEnd) {
        return new Span(
                CoordinateSpace.UTF16,
                start,
                end,
                type,
                prob,
                byteStart,
                byteEnd,
                codepointStart,
                codepointEnd,
                -1L,
                -1L,
                null,
                null);
    }

    /** Creates a token-index span. It deliberately has no character offsets. */
    public static Span token(int start, int end, String type, double prob) {
        return new Span(CoordinateSpace.TOKEN, start, end, type, prob, -1L, -1L, -1L, -1L, start, end, null, null);
    }

    static Span fromSource(TurboSpanView row) {
        TurboSpanData.CoordinateRange source = row.sourceRange()
                .orElseThrow(() -> new IllegalStateException("native row has no source range for an OpenNLP-style span"));
        long tokenStart = -1L;
        long tokenEnd = -1L;
        if (row.tokenRange().isPresent()) {
            tokenStart = row.tokenRange().get().start();
            tokenEnd = row.tokenRange().get().end();
        }
        return new Span(
                CoordinateSpace.UTF16,
                Math.toIntExact(source.utf16Start()),
                Math.toIntExact(source.utf16End()),
                row.label().orElse(null),
                row.probability().orElse(Double.NaN),
                source.byteStart(),
                source.byteEnd(),
                source.codepointStart(),
                source.codepointEnd(),
                tokenStart,
                tokenEnd,
                row.layer(),
                row.instance());
    }

    static Span fromToken(TurboSpanView row) {
        TurboSpanData.TokenRange tokens = row.tokenRange()
                .orElseThrow(() -> new IllegalStateException("native row has no token range for an OpenNLP-style span"));
        return new Span(
                CoordinateSpace.TOKEN,
                Math.toIntExact(tokens.start()),
                Math.toIntExact(tokens.end()),
                row.label().orElse(null),
                row.probability().orElse(Double.NaN),
                -1L,
                -1L,
                -1L,
                -1L,
                tokens.start(),
                tokens.end(),
                row.layer(),
                row.instance());
    }

    /** Returns the coordinate space governing start and end. */
    public CoordinateSpace getCoordinateSpace() {
        return coordinateSpace;
    }

    /** Returns whether start/end are token indices. */
    public boolean isTokenSpan() {
        return coordinateSpace == CoordinateSpace.TOKEN;
    }

    /** Returns whether start/end are source UTF-16 code-unit offsets. */
    public boolean isCharacterSpan() {
        return coordinateSpace == CoordinateSpace.UTF16;
    }

    /** Returns the inclusive start in this span's named coordinate space. */
    public int getStart() {
        return start;
    }

    /** Returns the exclusive end in this span's named coordinate space. */
    public int getEnd() {
        return end;
    }

    /** Returns the span type label, or {@code null} if the native row had no label. */
    public String getType() {
        return type;
    }

    /** Returns the probability, or {@link Double#NaN} when no probability exists. */
    public double getProb() {
        return prob;
    }

    /** Returns end minus start in this span's named coordinate space. */
    public int length() {
        return end - start;
    }

    /** Returns whether corresponding source byte/codepoint offsets are present. */
    public boolean hasSourceCoordinates() {
        return coordinateSpace == CoordinateSpace.UTF16;
    }

    /** Returns the source UTF-8 byte start. Token spans have no such offset. */
    public long getByteStart() {
        requireCharacterSpan("byte start");
        return byteStart;
    }

    /** Returns the source UTF-8 byte end. Token spans have no such offset. */
    public long getByteEnd() {
        requireCharacterSpan("byte end");
        return byteEnd;
    }

    /** Returns the source Unicode scalar start. Token spans have no such offset. */
    public long getCodepointStart() {
        requireCharacterSpan("codepoint start");
        return codepointStart;
    }

    /** Returns the source Unicode scalar end. Token spans have no such offset. */
    public long getCodepointEnd() {
        requireCharacterSpan("codepoint end");
        return codepointEnd;
    }

    /** Returns whether a corresponding token range is present. */
    public boolean hasTokenRange() {
        return tokenStart != -1L;
    }

    /** Returns the corresponding token start, when present. */
    public long getTokenStart() {
        if (!hasTokenRange()) {
            throw new IllegalStateException("span has no token range");
        }
        return tokenStart;
    }

    /** Returns the corresponding token end, when present. */
    public long getTokenEnd() {
        if (!hasTokenRange()) {
            throw new IllegalStateException("span has no token range");
        }
        return tokenEnd;
    }

    /** Returns the native layer id when this span came from a native row, otherwise {@code null}. */
    public String getLayer() {
        return layer;
    }

    /** Returns the native layer instance when this span came from a native row, otherwise {@code null}. */
    public String getInstance() {
        return instance;
    }

    /**
     * Returns covered source text for a UTF-16 span. Token spans fail rather than slicing
     * text by token indices.
     */
    public CharSequence getCoveredText(CharSequence text) {
        Objects.requireNonNull(text, "text");
        requireCharacterSpan("covered text");
        if (end > text.length()) {
            throw new IllegalArgumentException(
                    "UTF-16 span [" + start + ", " + end + ") exceeds text length " + text.length());
        }
        return text.subSequence(start, end);
    }

    /** Returns covered tokens for a token span or a character span with a token range. */
    public String[] getCoveredTokens(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        if (!hasTokenRange()) {
            throw new IllegalStateException("span has no token range");
        }
        int from = Math.toIntExact(tokenStart);
        int to = Math.toIntExact(tokenEnd);
        if (from < 0 || to < from || to > tokens.length) {
            throw new IndexOutOfBoundsException("token range [" + from + ", " + to + ") outside 0.." + tokens.length);
        }
        return Arrays.copyOfRange(tokens, from, to);
    }

    private void requireCharacterSpan(String what) {
        if (!isCharacterSpan()) {
            throw new IllegalStateException("token span has no " + what);
        }
    }

    private static void requirePair(long start, long end, String name) {
        if (start < 0 || end < start) {
            throw new IllegalArgumentException("Invalid " + name + " bounds: [" + start + ", " + end + ")");
        }
    }

    @Override
    public int compareTo(Span other) {
        int c = coordinateSpace.compareTo(other.coordinateSpace);
        if (c != 0) {
            return c;
        }
        c = Integer.compare(start, other.start);
        if (c != 0) {
            return c;
        }
        c = Integer.compare(end, other.end);
        if (c != 0) {
            return c;
        }
        return Objects.compare(type, other.type, java.util.Comparator.nullsFirst(java.util.Comparator.naturalOrder()));
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Span that)) {
            return false;
        }
        return coordinateSpace == that.coordinateSpace
                && start == that.start
                && end == that.end
                && Double.compare(prob, that.prob) == 0
                && byteStart == that.byteStart
                && byteEnd == that.byteEnd
                && codepointStart == that.codepointStart
                && codepointEnd == that.codepointEnd
                && tokenStart == that.tokenStart
                && tokenEnd == that.tokenEnd
                && Objects.equals(type, that.type)
                && Objects.equals(layer, that.layer)
                && Objects.equals(instance, that.instance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                coordinateSpace,
                start,
                end,
                type,
                prob,
                byteStart,
                byteEnd,
                codepointStart,
                codepointEnd,
                tokenStart,
                tokenEnd,
                layer,
                instance);
    }

    @Override
    public String toString() {
        String prefix = coordinateSpace == CoordinateSpace.TOKEN ? "tokens[" : "utf16[";
        return type == null ? prefix + start + ".." + end + ")" : prefix + start + ".." + end + ") " + type;
    }
}
