package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_classify_options;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/** Per-call classification options ({@code turbo_classify_options}). */
public record ClassifyOptions(Truncate truncate, int maxTokens, Aggregation aggregation, boolean rawScores) {

    public static ClassifyOptions defaults() {
        return new ClassifyOptions(Truncate.MODEL, 0, Aggregation.MODEL, false);
    }

    public ClassifyOptions withAggregation(Aggregation a) {
        return new ClassifyOptions(truncate, maxTokens, a, rawScores);
    }

    public ClassifyOptions withRawScores(boolean b) {
        return new ClassifyOptions(truncate, maxTokens, aggregation, b);
    }

    public ClassifyOptions withTruncate(Truncate t) {
        return new ClassifyOptions(t, maxTokens, aggregation, rawScores);
    }

    MemorySegment toNative(Arena arena) {
        MemorySegment o = turbo_classify_options.allocate(arena);
        turbo_classify_options.struct_size(o, (int) turbo_classify_options.sizeof());
        turbo_classify_options.truncate(o, truncate.value());
        turbo_classify_options.max_tokens(o, maxTokens);
        turbo_classify_options.aggregation(o, aggregation.value());
        turbo_classify_options.raw_scores(o, rawScores ? 1 : 0);
        return o;
    }
}
