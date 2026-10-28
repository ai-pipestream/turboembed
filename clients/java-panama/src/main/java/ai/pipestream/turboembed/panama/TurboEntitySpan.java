package ai.pipestream.turboembed.panama;

import java.util.Objects;

/**
 * Named entity or PII mention span produced by {@link TurboAnalyzer} with both token bounds and
 * character/byte bounds.
 *
 * @param kind           entity or PII type label (for example {@code "person"}, {@code "location"}, {@code "email"})
 * @param byteStart      start UTF-8 byte offset (0 when produced from token-only array input)
 * @param byteEnd        end UTF-8 byte offset
 * @param utf16Start     start UTF-16 code-unit offset in JVM {@link String}
 * @param utf16End       end UTF-16 code-unit offset in JVM {@link String}
 * @param codepointStart start Unicode scalar offset
 * @param codepointEnd   end Unicode scalar offset
 * @param tokenStart     inclusive start token index
 * @param tokenEnd       exclusive end token index
 * @param prob           entity probability or confidence score
 */
public record TurboEntitySpan(
        String kind,
        int byteStart,
        int byteEnd,
        int utf16Start,
        int utf16End,
        int codepointStart,
        int codepointEnd,
        int tokenStart,
        int tokenEnd,
        double prob) {

    /**
     * Validates non-null entity kind.
     */
    public TurboEntitySpan {
        Objects.requireNonNull(kind, "kind");
    }
}
