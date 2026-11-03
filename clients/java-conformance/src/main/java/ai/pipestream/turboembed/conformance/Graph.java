package ai.pipestream.turboembed.conformance;

import java.util.List;

/** The rows and the serialized graph one {@code analyzeGraph} call produced. */
public record Graph(List<Row> rows, byte[] bytes) {
    /** Takes an explicit test snapshot and closes its native owner. */
    static Graph copy(ai.pipestream.turboembed.TurboAnalysisResult result) {
        try (result) { return new Graph(Row.rows(result.copySpans()), result.copyGraphBytes()); }
    }
}
