package ai.pipestream.turboembed.conformance;

import java.util.List;

/** A binding's analyzer, every task answering copied neutral rows. */
public interface Analyzer extends AutoCloseable {
    List<Row> detectSentences(String text);

    List<Row> tokenize(String text);

    List<Row> posTag(String[] tokens);

    List<Row> lemmatize(String[] tokens, String[] tags);

    List<Row> findTokens(String[] tokens);

    List<Row> findText(String text);

    List<Row> detectPii(String text);

    Graph analyzeGraph(String text);

    List<Row> inspectGraph(byte[] graph);

    @Override
    void close();
}
