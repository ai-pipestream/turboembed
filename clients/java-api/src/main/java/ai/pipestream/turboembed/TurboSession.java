package ai.pipestream.turboembed;

import java.nio.file.Path;
import java.util.List;

/** Shared embedding and reranking session. Implementations own and release the native session. */
public interface TurboSession extends AutoCloseable {
    /** Opens a session on device 0 (the CPU) with the model's defaults, using the selected bridge. */
    static TurboSession open(Path bundleDir) {
        return TurboRuntime.load().openSession(bundleDir);
    }

    /** Opens a session as {@code options} says, using the selected bridge. */
    static TurboSession open(Path bundleDir, TurboSessionOptions options) {
        return TurboRuntime.load().openSession(bundleDir, options);
    }

    /**
     * Embeds a batch of UTF-8 texts with the bundle's options and returns one {@code float[dim]}
     * vector per input text.
     *
     * @param texts non-empty list of texts to embed
     * @return {@code float[texts.size()][dim]} embedding matrix
     */
    default float[][] embed(List<String> texts) {
        return embed(texts, TurboEmbedOptions.DEFAULT);
    }

    /**
     * Embeds a batch of UTF-8 texts as {@code options} says and returns one F32 vector per text,
     * {@code outputDim} wide when the options cut it. Options that quantize are refused here:
     * {@link #embedCodes} returns their codes.
     */
    float[][] embed(List<String> texts, TurboEmbedOptions options);

    /**
     * Embeds a batch of UTF-8 texts with options that quantize ({@link
     * TurboEmbedOptions.Quantization#INT8} or {@link TurboEmbedOptions.Quantization#UBINARY}) and
     * returns each text's codes: a signed byte a dimension, or the sign bits eight to a byte.
     */
    byte[][] embedCodes(List<String> texts, TurboEmbedOptions options);

    /**
     * Reranks {@code passages} against {@code query} and returns raw relevance logits.
     *
     * @param query    query text
     * @param passages candidate passage texts
     * @return one relevance score per passage
     */
    default float[] rerank(String query, List<String> passages) {
        return rerank(query, passages, TurboRerankOptions.DEFAULT);
    }

    /** Reranks {@code passages} against {@code query} with each pair cut as {@code options} says. */
    float[] rerank(String query, List<String> passages, TurboRerankOptions options);

    /** What the model is and the device the session runs on. */
    TurboModelInfo modelInfo();

    /** Returns whether this session has released its native handles. */
    boolean isClosed();

    void close();
}
