package ai.pipestream.turboembed.conformance;

import java.util.List;

/** The rows and the serialized graph one {@code analyzeGraph} call produced. */
public record Graph(List<Row> rows, byte[] bytes) {}
