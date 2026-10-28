package ai.pipestream.turboembed.panama;

import java.util.Objects;

/**
 * A single token produced by {@link TurboAnalyzer#tokenize(String)} with UTF-8 byte, UTF-16
 * code-unit, and Unicode scalar (codepoint) offsets.
 *
 * @param index          0-based token index
 * @param surface        token surface text
 * @param byteStart      start byte offset in UTF-8 text
 * @param byteEnd        end byte offset in UTF-8 text
 * @param utf16Start     start UTF-16 code-unit offset in JVM {@link String}
 * @param utf16End       end UTF-16 code-unit offset in JVM {@link String}
 * @param codepointStart start Unicode scalar index
 * @param codepointEnd   end Unicode scalar index
 * @param prob           tokenizer confidence score
 */
public record TurboToken(
        int index,
        String surface,
        int byteStart,
        int byteEnd,
        int utf16Start,
        int utf16End,
        int codepointStart,
        int codepointEnd,
        double prob) {

    /**
     * Validates non-null surface text.
     */
    public TurboToken {
        Objects.requireNonNull(surface, "surface");
    }
}
