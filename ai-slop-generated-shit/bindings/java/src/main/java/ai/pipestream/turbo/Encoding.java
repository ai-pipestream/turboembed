package ai.pipestream.turbo;

/**
 * Encoded rows: {@code ids} and {@code mask} are {@code rows x rowStride},
 * row-major; {@code lengths} holds each row's live token count.
 */
public record Encoding(int rows, int rowStride, int[] ids, int[] mask, int[] lengths) {
    /** The live ids of row {@code r}. */
    public int[] row(int r) {
        int[] out = new int[lengths[r]];
        System.arraycopy(ids, r * rowStride, out, 0, lengths[r]);
        return out;
    }
}
