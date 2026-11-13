package ai.pipestream.turboembed;

/**
 * How {@link TurboSession#rerank(String, java.util.List, TurboRerankOptions)} cuts each pair, as
 * {@code turbo_rerank_options}: 0 in a field means the bundle's or the session's own value.
 *
 * @param queryTokens the query's token budget, special tokens excluded
 * @param maxTokens   a pair's length, special tokens included; at most the session's max sequence
 * @param stride      how far each passage window advances; 0 for whole windows
 */
public record TurboRerankOptions(int queryTokens, int maxTokens, int stride) {
    /** The bundle's and the session's own values. */
    public static final TurboRerankOptions DEFAULT = new TurboRerankOptions(0, 0, 0);

    /** Checks that no field is negative. */
    public TurboRerankOptions {
        TurboSessionOptions.nonNegative(queryTokens, "queryTokens");
        TurboSessionOptions.nonNegative(maxTokens, "maxTokens");
        TurboSessionOptions.nonNegative(stride, "stride");
    }

    /** The values a native bridge passes on: {@code [queryTokens, maxTokens, stride]}. */
    public int[] toNative() {
        return new int[] {queryTokens, maxTokens, stride};
    }
}
