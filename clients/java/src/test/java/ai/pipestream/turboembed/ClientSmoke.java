package ai.pipestream.turboembed;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.util.Iterator;
import turbo.result.v1.AnalysisServiceGrpc;
import turbo.result.v1.TurboAnalysis.AnalyzeDocument;
import turbo.result.v1.TurboAnalysis.AnalyzePipelineRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeProgressiveEvent;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamHeader;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamResponse;
import turbo.result.v1.TurboAnalysis.BundleCapabilities;
import turbo.result.v1.TurboAnalysis.CapabilitiesRequest;
import turbo.result.v1.TurboAnalysis.CapabilitiesResponse;
import turbo.result.v1.TurboAnalysis.DocumentGraphResponse;
import turbo.result.v1.TurboAnalysis.LayerInfo;
import turbo.result.v1.TurboAnalysis.StructureBlock;
import turbo.result.v1.TurboAnalysis.StructureBlockKind;
import turbo.result.v1.TurboAnalysis.StructureSeed;
import turbo.result.v1.TurboTypedAnnotations.OffsetMode;

/**
 * Minimal Java client compiled only against the published {@code turboembed-grpc}
 * stub artifact (issue #1297). Calls {@code Capabilities} and {@code AnalyzeStream}
 * against a running {@code turbo-kserve} instance, then sends a heading and a list
 * as a {@code StructureSeed} to {@code AnalyzeDocumentProgressive}, its offsets the
 * UTF-16 indices of a Java string.
 */
public final class ClientSmoke {
    private ClientSmoke() {}

    public static void main(String[] args) throws Exception {
        String target = args.length > 0 ? args[0] : System.getenv("TURBO_KSERVE_SMOKE_ADDR");
        if (target == null || target.isEmpty()) {
            throw new IllegalArgumentException("Expected server address as argv[0] or TURBO_KSERVE_SMOKE_ADDR");
        }

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        try {
            AnalysisServiceGrpc.AnalysisServiceBlockingStub blocking =
                    AnalysisServiceGrpc.newBlockingStub(channel).withDeadlineAfter(15, TimeUnit.SECONDS);
            CapabilitiesResponse caps = blocking.capabilities(CapabilitiesRequest.getDefaultInstance());
            if (caps.getBundlesCount() < 1) {
                throw new AssertionError("Expected at least 1 loaded bundle in CapabilitiesResponse");
            }
            BundleCapabilities bundle = caps.getBundles(0);
            if (!"tiny-bert-bundle".equals(bundle.getName())
                    || bundle.getWidthsCount() < 1
                    || bundle.getWidths(0) != 32) {
                throw new AssertionError(
                        "Unexpected bundle capabilities: name="
                                + bundle.getName()
                                + " widths="
                                + bundle.getWidthsList());
            }

            AnalysisServiceGrpc.AnalysisServiceStub asyncStub =
                    AnalysisServiceGrpc.newStub(channel).withDeadlineAfter(15, TimeUnit.SECONDS);
            CountDownLatch done = new CountDownLatch(1);
            AtomicReference<Throwable> streamError = new AtomicReference<>();
            List<AnalyzeStreamResponse> responses = new ArrayList<>();

            StreamObserver<AnalyzeStreamRequest> reqObserver = asyncStub.analyzeStream(
                    new StreamObserver<AnalyzeStreamResponse>() {
                        @Override
                        public void onNext(AnalyzeStreamResponse value) {
                            responses.add(value);
                        }

                        @Override
                        public void onError(Throwable t) {
                            streamError.set(t);
                            done.countDown();
                        }

                        @Override
                        public void onCompleted() {
                            done.countDown();
                        }
                    });

            reqObserver.onNext(AnalyzeStreamRequest.newBuilder()
                    .setHeader(AnalyzeStreamHeader.newBuilder().setOrderValue(0).setMaxInflight(4).setBatchLingerMs(0))
                    .build());
            reqObserver.onNext(AnalyzeStreamRequest.newBuilder()
                    .setDocument(AnalyzeDocument.newBuilder()
                            .setId("java-doc-1")
                            .setModel("tiny-bert-bundle")
                            .setText(ByteString.copyFrom("hello from java client stub", StandardCharsets.UTF_8)))
                    .build());
            reqObserver.onCompleted();

            if (!done.await(15, TimeUnit.SECONDS)) {
                throw new AssertionError("Timed out waiting for AnalyzeStream response");
            }
            if (streamError.get() != null) {
                throw new AssertionError("AnalyzeStream failed", streamError.get());
            }
            if (responses.size() != 1) {
                throw new AssertionError("Expected 1 AnalyzeStreamResponse, got " + responses.size());
            }
            AnalyzeStreamResponse resp = responses.get(0);
            if (!"java-doc-1".equals(resp.getId()) || !resp.hasVectors()) {
                throw new AssertionError("Unexpected AnalyzeStreamResponse: " + resp);
            }
            if (resp.getVectors().getDim() != 32 || resp.getVectors().getData().size() != 32 * 4) {
                throw new AssertionError(
                        "Unexpected vector shape: dim="
                                + resp.getVectors().getDim()
                                + " bytes="
                                + resp.getVectors().getData().size());
            }
            System.out.println("java stub client ok: bundle=" + bundle.getName() + " dim=" + resp.getVectors().getDim());
            structureSeed(channel);
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    static StructureBlock block(String text, StructureBlockKind kind, String covers, int from, int level, int parent) {
        int start = text.indexOf(covers, from);
        StructureBlock.Builder b = StructureBlock.newBuilder()
                .setKind(kind)
                .setStart(start)
                .setEnd(start + covers.length())
                .setLevel(level);
        if (parent >= 0) {
            b.setParent(parent);
        }
        return b.build();
    }

    static AnalyzePipelineRequest seedRequest(StructureSeed seed) {
        return AnalyzePipelineRequest.newBuilder()
                .setDocId("java-seed")
                .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                .addSteps("turbo-sentdetect:rules")
                .addSteps("turbo:structure-boundaries")
                .setStructureSeed(seed)
                .build();
    }

    /** A heading and a two-item list as a seed; then the same list overlapping, refused. */
    static void structureSeed(ManagedChannel channel) {
        String text = "# Caf\u00e9\n\n- \ud83d\ude00 one\n- two\n";
        String list = "- \ud83d\ude00 one\n- two";
        StructureSeed seed = StructureSeed.newBuilder()
                .setText(text)
                .addBlocks(block(text, StructureBlockKind.STRUCTURE_BLOCK_KIND_HEADING, "# Caf\u00e9", 0, 1, -1))
                .addBlocks(block(text, StructureBlockKind.STRUCTURE_BLOCK_KIND_LIST, list, 0, 1, -1))
                .addBlocks(block(text, StructureBlockKind.STRUCTURE_BLOCK_KIND_LIST_ITEM, "- \ud83d\ude00 one", 0, 1, 1))
                .addBlocks(block(text, StructureBlockKind.STRUCTURE_BLOCK_KIND_LIST_ITEM, "- two", 0, 1, 1))
                .build();
        AnalysisServiceGrpc.AnalysisServiceBlockingStub stub =
                AnalysisServiceGrpc.newBlockingStub(channel).withDeadlineAfter(15, TimeUnit.SECONDS);
        DocumentGraphResponse complete = null;
        Iterator<AnalyzeProgressiveEvent> events = stub.analyzeDocumentProgressive(seedRequest(seed));
        while (events.hasNext()) {
            AnalyzeProgressiveEvent event = events.next();
            if (event.hasStepFailed()) {
                throw new AssertionError("step failed: " + event.getStepFailed());
            }
            if (event.hasComplete()) {
                complete = event.getComplete();
            }
        }
        if (complete == null || !text.equals(complete.getSourceText())) {
            throw new AssertionError("the seeded stream ended without its document");
        }
        LayerInfo structure = complete.getLayersList().stream()
                .filter(l -> l.getLayer().equals("turbo:structure"))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no turbo:structure layer"));
        if (structure.getItems() != 4) {
            throw new AssertionError("expected 4 structure blocks, got " + structure.getItems());
        }

        StructureBlock second = seed.getBlocks(3).toBuilder().setStart(seed.getBlocks(2).getStart() + 1).build();
        StructureSeed overlapping = seed.toBuilder().setBlocks(3, second).build();
        try {
            Iterator<AnalyzeProgressiveEvent> refused = AnalysisServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(15, TimeUnit.SECONDS)
                    .analyzeDocumentProgressive(seedRequest(overlapping));
            refused.hasNext();
            throw new AssertionError("an overlapping seed was accepted");
        } catch (StatusRuntimeException e) {
            if (e.getStatus().getCode() != Status.Code.INVALID_ARGUMENT
                    || !e.getStatus().getDescription().contains("block 3 overlaps")) {
                throw new AssertionError("unexpected refusal of an overlapping seed", e);
            }
        }
        System.out.println("java structure seed ok: blocks=" + structure.getItems());
    }
}
