package ai.pipestream.turbo;

/** One aggregated span from token classification ({@code turbo_span}). Byte offsets index the row's input text. */
public record Span(int row, long byteStart, long byteEnd, int label, float score) {}
