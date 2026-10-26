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
import turbo.result.v1.AnalysisServiceGrpc;
import turbo.result.v1.TurboAnalysis.AnalyzeDocument;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamHeader;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeStreamResponse;
import turbo.result.v1.TurboAnalysis.BundleCapabilities;
import turbo.result.v1.TurboAnalysis.CapabilitiesRequest;
import turbo.result.v1.TurboAnalysis.CapabilitiesResponse;

/**
 * Minimal Java client compiled only against the published {@code turboembed-grpc}
 * stub artifact (issue #1297). Calls {@code Capabilities} and {@code AnalyzeStream}
 * against a running {@code turbo-kserve} instance.
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
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
