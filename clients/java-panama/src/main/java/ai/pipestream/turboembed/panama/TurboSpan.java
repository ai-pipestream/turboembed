package ai.pipestream.turboembed.panama;

import java.util.Objects;

/**
 * Bi-directional span returned by {@code libturbo} in-process analysis, carrying UTF-8 byte,
 * UTF-16 code-unit, Unicode scalar (codepoint), and token-index bounds alongside the label and
 * probability.
 *
 * @param byteStart      start byte offset in UTF-8 source text
 * @param byteEnd        end byte offset (exclusive) in UTF-8 source text
 * @param utf16Start     start UTF-16 code-unit offset (matches {@link String#substring(int, int)})
 * @param utf16End       end UTF-16 code-unit offset (exclusive)
 * @param codepointStart start Unicode scalar value index
 * @param codepointEnd   end Unicode scalar value index (exclusive)
 * @param tokenStart     start token index
 * @param tokenEnd       end token index (exclusive)
 * @param label          resolved span label (sentence text, token surface, POS tag, lemma, or entity type)
 * @param prob           model probability or confidence score
 */
public record TurboSpan(
        int byteStart,
        int byteEnd,
        int utf16Start,
        int utf16End,
        int codepointStart,
        int codepointEnd,
        int tokenStart,
        int tokenEnd,
        String label,
        double prob) {

    /**
     * Validates non-null label on construction.
     */
    public TurboSpan {
        Objects.requireNonNull(label, "label");
    }

    /**
     * Slices {@code text} using this span's UTF-16 code-unit bounds.
     *
     * @param text source character sequence
     * @return covered subsequence
     */
    public String coveredText(CharSequence text) {
        Objects.requireNonNull(text, "text");
        return text.subSequence(utf16Start, utf16End).toString();
    }
}
