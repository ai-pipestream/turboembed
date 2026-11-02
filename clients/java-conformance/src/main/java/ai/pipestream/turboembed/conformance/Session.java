package ai.pipestream.turboembed.conformance;

import java.util.List;

/** A binding's embedding and reranking session. */
public interface Session extends AutoCloseable {
    float[][] embed(List<String> texts);

    float[] rerank(String query, List<String> passages);

    @Override
    void close();
}
