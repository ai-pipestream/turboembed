package ai.pipestream.turboembed;

import java.nio.file.Path;
import java.util.List;

/** Shared embedding and reranking session. Implementations own and release the native session. */
public interface TurboSession extends AutoCloseable {
    /** Opens a CPU session using the selected runtime bridge. */
    static TurboSession open(Path bundleDir) { return TurboRuntime.load().openSession(bundleDir); }

    /**
     * Embeds a batch of UTF-8 texts and returns one {@code float[dim]} vector per input text.
     *
     * @param texts non-empty list of texts to embed
     * @return {@code float[texts.size()][dim]} embedding matrix
     */
    float[][] embed(List<String> texts);

    /**
     * Reranks {@code passages} against {@code query} and returns raw relevance logits.
     *
     * @param query    query text
     * @param passages candidate passage texts
     * @return one relevance score per passage
     */
    float[] rerank(String query, List<String> passages);

    /** Returns whether this session has released its native handles. */
    boolean isClosed();

    void close();
}
