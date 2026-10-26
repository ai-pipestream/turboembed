package ai.pipestream.turboembed.examples;

import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import turbo.result.v1.AnalysisServiceGrpc;
import turbo.result.v1.AnalysisServiceGrpc.AnalysisServiceBlockingStub;
import turbo.result.v1.TurboAnalysis.AnalyzePipelineRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeProgressiveEvent;
import turbo.result.v1.TurboAnalysis.CombinedSegmentProto;
import turbo.result.v1.TurboAnalysis.DocumentWindowRequest;
import turbo.result.v1.TurboAnalysis.DocumentWindowResponse;
import turbo.result.v1.TurboAnalysis.HeatmapBucketProto;
import turbo.result.v1.TurboAnalysis.HeatmapLaneProto;
import turbo.result.v1.TurboAnalysis.ProjectChunksRequest;
import turbo.result.v1.TurboAnalysis.ProjectChunksResponse;
import turbo.result.v1.TurboAnalysis.ProjectedPointProto;
import turbo.result.v1.TurboAnalysis.SegmentAnnotationProto;
import turbo.result.v1.TurboTypedAnnotations.OffsetMode;

/**
 * The docling example from Java, against a running {@code turbo-kserve}.
 *
 * <p>Sends the docling structure ({@code data/report.seed.graph}, which
 * {@code docling-client seed} writes) as AnalyzeDocumentProgressive's
 * initial_graph, lets the server analyze, chunk and embed it, then asks
 * InspectDocumentWindow for the heat and ProjectChunks for the chunks.
 * Offsets are asked for in UTF-16 units, which Java strings index. Prints
 * the summary and, given the expected file, fails unless it matches.
 *
 * <pre>java DoclingClient HOST:PORT SEED [EXPECTED]</pre>
 *
 * <p>Needs only the {@code turboembed-grpc} stubs (clients/java) and a
 * gRPC transport.
 */
public final class DoclingClient {
    static final String MODEL = "tiny-bert-bundle";
    static final List<String> STEPS = List.of(
            "turbo:normalize:whitespace",
            "turbo-sentdetect:rules",
            "turbo-tokenize:uax29-untyped",
            "turbo:subwords",
            "turbo-geo:names",
            "turbo:structure-boundaries",
            "turbo:chunker",
            "turbo:levels",
            "turbo:chunk-vectors");
    static final int HEAT_BUCKETS = 16;
    static final String ENTITIES = "opennlp:entities";

    private DoclingClient() {}

    static String quote(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    }

    static String summary(int completed, DocumentWindowResponse window, ProjectChunksResponse chunks) {
        String text = window.getWindow().getText();
        StringBuilder out = new StringBuilder();
        out.append("steps ").append(completed).append('\n');
        for (ProjectedPointProto p : chunks.getProjection().getPointsList()) {
            out.append("chunk ").append(p.getItem()).append(" level ").append(p.getLevel())
                    .append(" vector ").append(p.getEmpty() ? "no" : "yes").append(' ')
                    .append(quote(text.substring(p.getStart(), p.getEnd()))).append('\n');
        }
        Map<Integer, Object[]> entities = new TreeMap<>();
        for (CombinedSegmentProto seg : window.getSegmentsList()) {
            for (SegmentAnnotationProto a : seg.getAnnotationsList()) {
                if (!a.getLayer().equals(ENTITIES)) {
                    continue;
                }
                Object[] e = entities.computeIfAbsent(a.getItem(), k -> new Object[] {a.getLabel(), seg.getStart(), seg.getEnd()});
                e[1] = Math.min((Integer) e[1], seg.getStart());
                e[2] = Math.max((Integer) e[2], seg.getEnd());
            }
        }
        for (Map.Entry<Integer, Object[]> e : entities.entrySet()) {
            Object[] v = e.getValue();
            out.append("entity ").append(e.getKey()).append(' ').append(v[0]).append(' ')
                    .append(quote(text.substring((Integer) v[1], (Integer) v[2]))).append('\n');
        }
        for (HeatmapLaneProto lane : window.getHeatmapLanesList()) {
            out.append("heat ").append(lane.getId()).append(' ').append(lane.getKind());
            for (HeatmapBucketProto b : lane.getBucketsList()) {
                out.append(' ').append(b.getCount());
            }
            out.append('\n');
        }
        return out.toString();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2 && args.length != 3) {
            throw new IllegalArgumentException("usage: DoclingClient HOST:PORT SEED [EXPECTED]");
        }
        ByteString seed = ByteString.copyFrom(Files.readAllBytes(Path.of(args[1])));
        ManagedChannel channel = ManagedChannelBuilder.forTarget(args[0]).usePlaintext().build();
        try {
            AnalysisServiceBlockingStub stub = AnalysisServiceGrpc.newBlockingStub(channel);
            AnalyzePipelineRequest request = AnalyzePipelineRequest.newBuilder()
                    .setDocId("report")
                    .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                    .addAllSteps(STEPS)
                    .setModel(MODEL)
                    .setInitialGraph(seed)
                    .build();
            int completed = 0;
            ByteString graph = null;
            Iterator<AnalyzeProgressiveEvent> events =
                    stub.withDeadlineAfter(60, TimeUnit.SECONDS).analyzeDocumentProgressive(request);
            while (events.hasNext()) {
                AnalyzeProgressiveEvent event = events.next();
                switch (event.getEventCase()) {
                    case LAYERS_READY -> completed++;
                    case STEP_FAILED -> throw new IllegalStateException("step " + event.getStepFailed().getStepId()
                            + " failed: " + event.getStepFailed().getFailure().getMessage());
                    case COMPLETE -> graph = event.getComplete().getGraph();
                    default -> {}
                }
            }
            if (graph == null) {
                throw new IllegalStateException("the stream ended without the document");
            }
            DocumentWindowResponse window = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                    .inspectDocumentWindow(DocumentWindowRequest.newBuilder()
                            .setGraph(graph)
                            .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                            .setHeatmapBuckets(HEAT_BUCKETS)
                            .build());
            ProjectChunksResponse chunks = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                    .projectChunks(ProjectChunksRequest.newBuilder()
                            .setGraph(graph)
                            .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                            .build());
            String got = summary(completed, window, chunks);
            System.out.print(got);
            if (args.length == 3) {
                String want = Files.readString(Path.of(args[2]), StandardCharsets.UTF_8);
                if (!want.equals(got)) {
                    throw new AssertionError("java docling client: the summary differs from " + args[2]);
                }
                System.out.println("java docling client ok");
            }
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
