package ai.pipestream.turboembed;

import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import turbo.result.v1.AnalysisServiceGrpc;
import turbo.result.v1.AnalysisServiceGrpc.AnalysisServiceBlockingStub;
import turbo.result.v1.TurboAnalysis.AnalyzePipelineRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeProgressiveEvent;
import turbo.result.v1.TurboAnalysis.CapabilitiesRequest;
import turbo.result.v1.TurboAnalysis.CapabilitiesResponse;
import turbo.result.v1.TurboAnalysis.ProfileCapabilities;
import turbo.result.v1.TurboTypedAnnotations.InspectDocumentGraphRequest;
import turbo.result.v1.TurboTypedAnnotations.LayerWireStatus;
import turbo.result.v1.TurboTypedAnnotations.OffsetMode;
import turbo.result.v1.TurboTypedAnnotations.SpanFragment;
import turbo.result.v1.TurboTypedAnnotations.TypedAnnotation;
import turbo.result.v1.TurboTypedAnnotations.TypedDocumentResult;
import turbo.result.v1.TurboTypedAnnotations.TypedLayer;

/**
 * One document through {@code AnalyzeDocumentProgressive} with the
 * {@code default-v1} profile, read back with {@code InspectDocumentGraph} in
 * UTF-16 units, which is how a Java {@code String} is indexed. Every span must
 * slice {@code raw_text} on character boundaries, every token and sentence
 * slice must equal the text the annotation carries, and the profile's layers
 * must come back under the ids {@code Capabilities} lists.
 *
 * <pre>java DocumentSmoke HOST:PORT</pre>
 */
public final class DocumentSmoke {
    static final String MODEL = "tiny-bert-bundle";
    static final String PROFILE = "default-v1";
    static final String PLACE = "Zürich";
    // Latin with diacritics, an emoji outside the BMP (two UTF-16 units, four
    // UTF-8 bytes), CJK (one unit, three bytes) and a gazetteer place name
    // after both, so a span past them is wrong in every unit but the right one.
    static final String TEXT = "Grüße 😀 aus dem Büro.\n\n東京の会議は 🙂 午後です。\n\nThe team meets in " + PLACE + " next week.";

    private DocumentSmoke() {}

    static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError(what);
        }
    }

    static boolean onBoundary(String text, int at) {
        return at == 0 || at == text.length() || !Character.isLowSurrogate(text.charAt(at));
    }

    static String slice(String text, TypedLayer layer, TypedAnnotation a) {
        check(a.getSpansCount() > 0, layer.getId() + ": an annotation without a span");
        SpanFragment first = a.getSpans(0);
        SpanFragment last = a.getSpans(a.getSpansCount() - 1);
        for (SpanFragment s : a.getSpansList()) {
            check(s.getStart() <= s.getEnd() && s.getEnd() <= text.length(),
                    layer.getId() + ": span " + s.getStart() + ".." + s.getEnd() + " outside " + text.length() + " units");
            check(onBoundary(text, s.getStart()) && onBoundary(text, s.getEnd()),
                    layer.getId() + ": span " + s.getStart() + ".." + s.getEnd() + " splits a surrogate pair");
        }
        return text.substring(first.getStart(), last.getEnd());
    }

    static boolean hasLayer(TypedDocumentResult doc, String id) {
        // A keyed id such as "turbo:embeddings<turbo:chunks>" names the layer
        // and the layer it hangs on.
        int open = id.indexOf('<');
        String base = open < 0 ? id : id.substring(0, open);
        String anchor = open < 0 ? null : id.substring(open + 1, id.length() - 1);
        for (TypedLayer layer : doc.getLayersList()) {
            if (layer.getId().equals(id)
                    || (layer.getId().equals(base) && anchor != null && anchor.equals(layer.getAnchorLayer()))) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) throws Exception {
        String target = args.length > 0 ? args[0] : System.getenv("TURBO_KSERVE_SMOKE_ADDR");
        if (target == null || target.isEmpty()) {
            throw new IllegalArgumentException("Expected server address as argv[0] or TURBO_KSERVE_SMOKE_ADDR");
        }
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        try {
            AnalysisServiceBlockingStub stub = AnalysisServiceGrpc.newBlockingStub(channel);
            CapabilitiesResponse caps =
                    stub.withDeadlineAfter(15, TimeUnit.SECONDS).capabilities(CapabilitiesRequest.getDefaultInstance());
            ProfileCapabilities profile = caps.getProfilesList().stream()
                    .filter(p -> p.getName().equals(PROFILE))
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Capabilities lists no " + PROFILE + " profile"));

            Iterator<AnalyzeProgressiveEvent> events = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                    .analyzeDocumentProgressive(AnalyzePipelineRequest.newBuilder()
                            .setDocId("java-document")
                            .setText(TEXT)
                            .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                            .setProfile(PROFILE)
                            .setModel(MODEL)
                            .build());
            ByteString graph = null;
            while (events.hasNext()) {
                AnalyzeProgressiveEvent event = events.next();
                switch (event.getEventCase()) {
                    case STEP_FAILED -> throw new AssertionError("step " + event.getStepFailed().getStepId()
                            + " failed: " + event.getStepFailed().getFailure().getMessage());
                    case COMPLETE -> graph = event.getComplete().getGraph();
                    default -> {}
                }
            }
            check(graph != null, "the stream ended without the document");

            TypedDocumentResult doc = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                    .inspectDocumentGraph(InspectDocumentGraphRequest.newBuilder()
                            .setDocId("java-document")
                            .setGraph(graph)
                            .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
                            .build());
            String text = doc.getRawText();
            check(text.equals(TEXT), "raw_text is not the text sent");

            for (String id : profile.getLayersList()) {
                check(hasLayer(doc, id), PROFILE + " lists " + id + " but the document has no such layer");
            }

            int spans = 0;
            int tokens = 0;
            int sentences = 0;
            List<String> entities = new ArrayList<>();
            for (TypedLayer layer : doc.getLayersList()) {
                check(layer.getStatus() == LayerWireStatus.LAYER_WIRE_STATUS_OK,
                        layer.getId() + ": status " + layer.getStatus() + " " + layer.getStatusDetail());
                check(!layer.getId().startsWith("opennlp:"), "a layer id in the opennlp: namespace: " + layer.getId());
                for (TypedAnnotation a : layer.getAnnotationsList()) {
                    if (a.getSpansCount() == 0) {
                        continue;
                    }
                    String got = slice(text, layer, a);
                    spans++;
                    switch (a.getValueCase()) {
                        case TOKEN -> {
                            tokens++;
                            check(got.equals(a.getToken().getSurface()),
                                    "token slice " + got + " is not its surface " + a.getToken().getSurface());
                        }
                        case SENTENCE -> {
                            sentences++;
                            check(got.equals(a.getSentence().getText()),
                                    "sentence slice " + got + " is not its text " + a.getSentence().getText());
                        }
                        case ENTITY -> entities.add(got);
                        default -> {}
                    }
                }
            }
            check(tokens > 0 && sentences > 0, "no tokens or no sentences: " + tokens + ", " + sentences);
            check(entities.contains(PLACE), "no entity slices to " + PLACE + "; entities " + entities);
            System.out.println("java document client ok: layers=" + doc.getLayersCount() + " spans=" + spans
                    + " tokens=" + tokens + " sentences=" + sentences + " entities=" + entities);
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
