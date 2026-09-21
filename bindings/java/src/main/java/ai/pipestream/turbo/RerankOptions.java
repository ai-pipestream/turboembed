package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_rerank_options;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/** Per-call rerank options ({@code turbo_rerank_options}). */
public record RerankOptions(Truncate truncate, int maxTokens, int topN, boolean returnSorted, boolean rawScores) {

    public static RerankOptions defaults() {
        return new RerankOptions(Truncate.MODEL, 0, 0, false, false);
    }

    public RerankOptions withTopN(int n) {
        return new RerankOptions(truncate, maxTokens, n, returnSorted, rawScores);
    }

    public RerankOptions withReturnSorted(boolean b) {
        return new RerankOptions(truncate, maxTokens, topN, b, rawScores);
    }

    public RerankOptions withRawScores(boolean b) {
        return new RerankOptions(truncate, maxTokens, topN, returnSorted, b);
    }

    public RerankOptions withTruncate(Truncate t) {
        return new RerankOptions(t, maxTokens, topN, returnSorted, rawScores);
    }

    MemorySegment toNative(Arena arena) {
        MemorySegment o = turbo_rerank_options.allocate(arena);
        turbo_rerank_options.struct_size(o, (int) turbo_rerank_options.sizeof());
        turbo_rerank_options.truncate(o, truncate.value());
        turbo_rerank_options.max_tokens(o, maxTokens);
        turbo_rerank_options.top_n(o, topN);
        turbo_rerank_options.return_sorted(o, returnSorted ? 1 : 0);
        turbo_rerank_options.raw_scores(o, rawScores ? 1 : 0);
        return o;
    }
}
