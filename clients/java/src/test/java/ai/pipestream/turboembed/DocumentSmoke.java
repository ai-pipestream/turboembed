package ai.pipestream.turboembed;

import com.google.protobuf.ByteString;
import com.google.protobuf.UnknownFieldSet;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.UnaryOperator;
import turbo.result.v1.AnalysisServiceGrpc;
import turbo.result.v1.AnalysisServiceGrpc.AnalysisServiceBlockingStub;
import turbo.result.v1.TurboAnalysis.AnalyzePipelineRequest;
import turbo.result.v1.TurboAnalysis.AnalyzeProgressiveEvent;
import turbo.result.v1.TurboAnalysis.CapabilitiesRequest;
import turbo.result.v1.TurboAnalysis.CapabilitiesResponse;
import turbo.result.v1.TurboAnalysis.ProfileCapabilities;
import turbo.result.v1.TurboTypedAnnotations.AnchorKind;
import turbo.result.v1.TurboTypedAnnotations.EditRun;
import turbo.result.v1.TurboTypedAnnotations.InlineMapping;
import turbo.result.v1.TurboTypedAnnotations.InspectDocumentGraphRequest;
import turbo.result.v1.TurboTypedAnnotations.LayerProvenance;
import turbo.result.v1.TurboTypedAnnotations.LayerReceipt;
import turbo.result.v1.TurboTypedAnnotations.LayerWireStatus;
import turbo.result.v1.TurboTypedAnnotations.MappingPayload;
import turbo.result.v1.TurboTypedAnnotations.OffsetMode;
import turbo.result.v1.TurboTypedAnnotations.ProcessingReceipt;
import turbo.result.v1.TurboTypedAnnotations.ProcessingStatus;
import turbo.result.v1.TurboTypedAnnotations.ScopedRef;
import turbo.result.v1.TurboTypedAnnotations.SpanFragment;
import turbo.result.v1.TurboTypedAnnotations.TypedAnnotation;
import turbo.result.v1.TurboTypedAnnotations.TypedDocumentResult;
import turbo.result.v1.TurboTypedAnnotations.TypedLayer;
import turbo.result.v1.TurboTypedAnnotations.ViewChain;
import turbo.result.v1.TurboTypedAnnotations.ViewInfo;

/**
 * One document through {@code AnalyzeDocumentProgressive} with the
 * {@code default-v1} profile, read back with {@code InspectDocumentGraph} in
 * UTF-16 units, which is how a Java {@code String} is indexed. Every span must
 * slice {@code raw_text} on character boundaries, every token and sentence
 * slice must equal the text the annotation carries, and the profile's layers
 * must come back under the ids {@code Capabilities} lists. Then the Rust to
 * generated Java to Rust conformance cases of {@link Conformance} run on
 * the same server (issue #1910).
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
                        layer.getId() + ": status " + layer.getStatus() + " " + layer.getError().getKind());
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
            Conformance.run(stub);
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}

/**
 * Rust to generated Java to Rust conformance (issue #1910), run after the
 * document smoke on the same live server. Rust produces a typed result
 * with its envelope, receipt and a PII layer on a source and a masked
 * view; generated Java parses it, re-encodes it and reads it back through
 * {@code RoundTripTypedDocument}, which validates. The cases cover the
 * mapping chain (a Java port of the server's stepwise {@code map_span},
 * held to Rust on the #1907 chained-view cases in
 * {@code clients/java/src/test/resources/mapping-cases.txt}), presence,
 * scoped references, the receipt, offsets above 2^31 read unsigned,
 * unknown fields and enum numbers on both sides, and where a planted PII
 * value appears in the bytes. Test-side code only: none of it is a Java
 * API.
 */
final class Conformance {
    static final String VALUE = "jane.doe@example.com";
    static final String TEXT = "Contact " + VALUE + " by Friday. Call +1 415 555 0123 or write to 10 Main St.";
    static final String DOC = "java-conformance";
    static final String OPERATION = "java-conformance-1";
    static final long BIG = 3_000_000_000L;
    /** Field 999, varint 7: the tag {@code (999 << 3) | 0} as a varint, then the value. */
    static final byte[] UNKNOWN_FIELD = {(byte) 0xb8, 0x3e, 0x07};
    static final int UNKNOWN_ENUM = 99;
    static final Path CASES = Path.of("clients", "java", "src", "test", "resources", "mapping-cases.txt");

    private Conformance() {}

    static void check(boolean ok, String what) {
        DocumentSmoke.check(ok, what);
    }

    static void run(AnalysisServiceBlockingStub stub) throws Exception {
        ByteString graph = analyze(stub, TEXT, List.of("turbo:pii", "turbo:pii-masked"));
        TypedDocumentResult source = inspect(stub, graph, 0);
        TypedDocumentResult masked = inspect(stub, graph, 1);
        envelope(source, masked);
        TypedLayer pii = layer(source, "turbo:pii");
        roundTrip(stub, source);
        roundTrip(stub, masked);
        presence(stub, source, pii);
        scopedRefs(stub, source, masked, pii);
        receipt(stub, source, pii);
        largeOffsets(stub, source);
        unknownFields(stub, source);
        unknownEnums(stub, source);
        disclosure(source, masked, pii);
        int cases = mappingCases(stub);
        System.out.println("java conformance ok: mentions=" + pii.getAnnotationsCount() + " views="
                + source.getViewsCount() + " mapping cases=" + cases);
    }

    static ByteString analyze(AnalysisServiceBlockingStub stub, String text, List<String> steps) {
        Iterator<AnalyzeProgressiveEvent> events = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                .analyzeDocumentProgressive(AnalyzePipelineRequest.newBuilder()
                        .setDocId(DOC)
                        .setText(text)
                        .setOffsets(OffsetMode.OFFSET_MODE_UTF8)
                        .addAllSteps(steps)
                        .setOperationKey(OPERATION)
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
        return graph;
    }

    static TypedDocumentResult inspect(AnalysisServiceBlockingStub stub, ByteString graph, int view) {
        return stub.withDeadlineAfter(60, TimeUnit.SECONDS).inspectDocumentGraph(InspectDocumentGraphRequest.newBuilder()
                .setDocId(DOC)
                .setGraph(graph)
                .setView(view)
                .setOffsets(OffsetMode.OFFSET_MODE_UTF8)
                .setOperationKey(OPERATION)
                .build());
    }

    static TypedDocumentResult roundTrip(AnalysisServiceBlockingStub stub, TypedDocumentResult m) {
        TypedDocumentResult back = stub.withDeadlineAfter(60, TimeUnit.SECONDS).roundTripTypedDocument(m);
        check(back.equals(m), "the round trip answers another message");
        check(Arrays.equals(back.toByteArray(), m.toByteArray()), "the round trip answers other bytes");
        return back;
    }

    /** The server's refusal of {@code m}: INVALID_ARGUMENT, with its message. */
    static String refused(AnalysisServiceBlockingStub stub, TypedDocumentResult m, String what) {
        try {
            stub.withDeadlineAfter(60, TimeUnit.SECONDS).roundTripTypedDocument(m);
        } catch (StatusRuntimeException e) {
            check(e.getStatus().getCode() == Status.Code.INVALID_ARGUMENT,
                    what + ": refused with " + e.getStatus().getCode() + ", not INVALID_ARGUMENT");
            return String.valueOf(e.getStatus().getDescription());
        }
        throw new AssertionError(what + ": accepted, not refused");
    }

    static TypedLayer layer(TypedDocumentResult m, String id) {
        for (TypedLayer l : m.getLayersList()) {
            if (l.getId().equals(id)) {
                return l;
            }
        }
        throw new AssertionError("no " + id + " layer");
    }

    static int layerIndex(TypedDocumentResult m, String id) {
        for (int i = 0; i < m.getLayersCount(); i++) {
            if (m.getLayers(i).getId().equals(id)) {
                return i;
            }
        }
        throw new AssertionError("no " + id + " layer");
    }

    static TypedDocumentResult withAnnotation(TypedDocumentResult m, String layerId, int index,
            UnaryOperator<TypedAnnotation.Builder> edit) {
        int li = layerIndex(m, layerId);
        TypedLayer.Builder layer = m.getLayers(li).toBuilder();
        layer.setAnnotations(index, edit.apply(layer.getAnnotations(index).toBuilder()).build());
        return m.toBuilder().setLayers(li, layer.build()).build();
    }

    static String hex(ByteString bytes) {
        StringBuilder out = new StringBuilder();
        for (byte b : bytes.toByteArray()) {
            out.append(String.format("%02x", b));
        }
        return out.toString();
    }

    static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** {@code text[start..end)} in UTF-8 bytes, the unit of the message. */
    static String slice(String text, long start, long end) {
        byte[] bytes = utf8(text);
        check(0 <= start && start <= end && end <= bytes.length, "span " + start + ".." + end + " outside " + bytes.length);
        return new String(bytes, (int) start, (int) (end - start), StandardCharsets.UTF_8);
    }

    static int count(byte[] haystack, byte[] needle) {
        int n = 0;
        for (int i = 0; i + needle.length <= haystack.length; i++) {
            if (Arrays.equals(Arrays.copyOfRange(haystack, i, i + needle.length), needle)) {
                n++;
            }
        }
        return n;
    }

    static InlineMapping mapping(TypedDocumentResult m) {
        check(m.hasMapping() && m.getMapping().hasInline(), "no inline mapping");
        return m.getMapping().getInline();
    }

    static ViewChain chainView(TypedDocumentResult m, int view) {
        for (ViewChain v : mapping(m).getViewsList()) {
            if (v.getView() == view) {
                return v;
            }
        }
        throw new AssertionError("view " + view + " is not in the mapping");
    }

    // Envelope: identities, views, provenance and the PII layer as produced.
    static void envelope(TypedDocumentResult source, TypedDocumentResult masked) {
        check(source.getRawText().equals(TEXT), "source raw_text is not the text sent");
        check(!masked.getRawText().equals(TEXT) && !masked.getRawText().contains(VALUE),
                "the masked view still shows the value");
        check(source.hasInput() && source.hasResult() && source.hasReceipt(), "no envelope on the source view");
        check(source.getInput().getRawBytesDigest().size() == 32 && source.getInput().getSourceTextDigest().size() == 32,
                "input digests are not 32 bytes");
        check(source.getResult().getContentDigest().size() == 32, "content_digest is not 32 bytes");
        check(source.getResult().getResultId().equals(hex(source.getResult().getContentDigest())),
                "result_id is not the hexadecimal of content_digest");
        check(masked.getResult().equals(source.getResult()) && masked.getInput().equals(source.getInput()),
                "the masked view has another identity");
        check(source.getView() == 0 && masked.getView() == 1, "views shown are not 0 and 1");
        InlineMapping mapping = mapping(source);
        check(mapping.equals(mapping(masked)), "the mapping differs between views of one result");
        check(mapping.getMappingSchemaVersion() == 1 && mapping.getViewsCount() == 2, "mapping is not two views");
        ViewChain maskedChain = mapping.getViews(1);
        check(maskedChain.getInstance().equals("masked") && maskedChain.hasParent() && maskedChain.getParent() == 0,
                "view 1 is not the masked child of the source");
        check(maskedChain.getTextLen() == utf8(masked.getRawText()).length, "masked text_len is not its raw_text");
        check(source.getViewsCount() == 2 && source.getViews(1).getInstance().equals("masked"), "views list");
        for (ViewInfo v : source.getViewsList()) {
            check(v.getStatus() == LayerWireStatus.LAYER_WIRE_STATUS_OK, "view " + v.getInstance() + " not OK");
            check(hasProvenance(source, "turbo:view", v.getInstance()), "no provenance for view " + v.getInstance());
        }
        TypedLayer pii = layer(source, "turbo:pii");
        check(pii.getStatus() == LayerWireStatus.LAYER_WIRE_STATUS_OK, "pii layer not OK");
        check(hasProvenance(source, pii.getId(), pii.getInstance()), "no provenance for the pii layer");
        check(pii.hasNativeView() && pii.getNativeView() == 0, "the pii layer is not native to the source view");
        check(pii.getAnnotationsCount() >= 2, "fewer than two mentions: " + pii.getAnnotationsCount());
        TypedAnnotation first = pii.getAnnotations(0);
        check(first.getValueCase() == TypedAnnotation.ValueCase.PII, "pii[0] is not a PiiMention");
        check(first.getSpansCount() == 1, "pii[0] is fragmented");
        SpanFragment span = first.getSpans(0);
        String got = slice(TEXT, Integer.toUnsignedLong(span.getStart()), Integer.toUnsignedLong(span.getEnd()));
        check(got.equals(VALUE), "pii[0] slices to " + got.length() + " bytes, not the planted value");
        check(first.getPii().getKind().equals("email"), "pii[0] kind " + first.getPii().getKind());
        check(!first.hasProbability(), "pii[0] carries a probability");
        check(first.hasModelRef(), "pii[0] carries no model_ref");
        boolean known = source.getModelsList().stream().anyMatch(m -> m.getModelRef().equals(first.getModelRef()));
        check(known, "pii[0] model_ref " + first.getModelRef() + " is not in models");
        TypedAnnotation onMasked = layer(masked, "turbo:pii").getAnnotations(0);
        check(onMasked.getSpansList().equals(first.getSpansList()), "an equal-width mask moved the mention");
        check(slice(masked.getRawText(), span.getStart(), span.getEnd()).chars().allMatch(c -> c == '*' || c == '.' || c == '@'),
                "the masked view shows letters of the value");
    }

    static boolean hasProvenance(TypedDocumentResult m, String layer, String instance) {
        return m.getProvenanceList().stream().anyMatch(p -> p.getLayer().equals(layer) && p.getInstance().equals(instance));
    }

    // Presence: absent is not zero, absent is not empty, on both sides.
    static void presence(AnalysisServiceBlockingStub stub, TypedDocumentResult source, TypedLayer pii) throws Exception {
        TypedDocumentResult zero = withAnnotation(source, "turbo:pii", 0, b -> b.setProbability(0.0));
        check(zero.toByteArray().length == source.toByteArray().length + 9, "a present zero probability costs 9 bytes");
        TypedAnnotation back = layer(roundTrip(stub, zero), "turbo:pii").getAnnotations(0);
        check(back.hasProbability() && back.getProbability() == 0.0, "Rust lost the present zero probability");
        check(!layer(source, "turbo:pii").getAnnotations(0).hasProbability(), "the original gained a probability");

        TypedDocumentResult absent = withAnnotation(source, "turbo:pii", 0, TypedAnnotation.Builder::clearModelRef);
        check(!layer(roundTrip(stub, absent), "turbo:pii").getAnnotations(0).hasModelRef(), "an absent model_ref came back");
        TypedDocumentResult empty = withAnnotation(source, "turbo:pii", 0, b -> b.setModelRef(""));
        check(TypedAnnotation.parseFrom(empty.getLayers(layerIndex(source, "turbo:pii")).getAnnotations(0).toByteString())
                .hasModelRef(), "Java lost the present empty model_ref");
        String why = refused(stub, empty, "empty model_ref");
        check(why.contains("empty model_ref"), "empty model_ref refused for another reason: " + why);
    }

    // Scoped references: made on one message, resolved on another of the
    // same result; another result or another view text refused.
    static ScopedRef scopedRef(TypedDocumentResult m, TypedLayer layer, int index) {
        check(!m.getResult().getResultId().isEmpty(), "no result identity");
        check(index < layer.getAnnotationsCount(), "index outside the layer");
        ScopedRef.Builder r = ScopedRef.newBuilder()
                .setResultId(m.getResult().getResultId())
                .setLayer(layer.getId())
                .setInstance(layer.getInstance())
                .setIndex(index);
        if (layer.hasNativeView()) {
            ViewChain v = chainView(m, layer.getNativeView());
            r.setViewInstance(v.getInstance()).setViewTextDigest(v.getTextDigest());
        }
        return r.build();
    }

    static TypedAnnotation resolve(TypedDocumentResult m, ScopedRef r) {
        if (!r.getResultId().equals(m.getResult().getResultId())) {
            throw new AssertionError("the reference is into another result");
        }
        TypedLayer layer = null;
        for (TypedLayer l : m.getLayersList()) {
            if (l.getId().equals(r.getLayer()) && l.getInstance().equals(r.getInstance())) {
                layer = l;
            }
        }
        if (layer == null || r.getIndex() >= layer.getAnnotationsCount()) {
            throw new AssertionError("no such layer or item");
        }
        if (layer.hasNativeView()) {
            ViewChain v = chainView(m, layer.getNativeView());
            if (!v.getInstance().equals(r.getViewInstance())) {
                throw new AssertionError("the reference is on another view");
            }
            if (!v.getTextDigest().equals(r.getViewTextDigest())) {
                throw new AssertionError("the view text is not the text the reference was made on");
            }
        } else if (!r.getViewInstance().isEmpty() || !r.getViewTextDigest().isEmpty()) {
            throw new AssertionError("the reference names a view and the layer has no text spans");
        }
        return layer.getAnnotations(r.getIndex());
    }

    static void scopedRefs(AnalysisServiceBlockingStub stub, TypedDocumentResult source, TypedDocumentResult masked,
            TypedLayer pii) throws Exception {
        ScopedRef r = scopedRef(source, pii, 0);
        check(r.getViewInstance().isEmpty() && r.getViewTextDigest().size() == 32, "the reference names the source view");
        check(ScopedRef.parseFrom(r.toByteArray()).equals(r), "ScopedRef does not survive its bytes");
        TypedAnnotation onSource = resolve(roundTrip(stub, source), r);
        TypedAnnotation onMasked = resolve(masked, r);
        check(onSource.equals(pii.getAnnotations(0)), "the reference resolves to another annotation");
        check(onMasked.getPii().getKind().equals(onSource.getPii().getKind()), "another mention on the masked view");
        expectRefused(() -> resolve(masked, r.toBuilder().setResultId("0" + r.getResultId().substring(1)).build()),
                "another result");
        expectRefused(() -> resolve(masked, r.toBuilder().setViewInstance("masked").build()), "another view");
        expectRefused(() -> resolve(masked, r.toBuilder().setIndex(pii.getAnnotationsCount()).build()), "no such");
    }

    static void expectRefused(Runnable call, String why) {
        try {
            call.run();
        } catch (AssertionError e) {
            check(String.valueOf(e.getMessage()).contains(why), "refused for another reason: " + e.getMessage());
            return;
        }
        throw new AssertionError("not refused: " + why);
    }

    // Receipt: bound to the envelope, every layer named once with its
    // provenance digest; a changed receipt refused.
    static void receipt(AnalysisServiceBlockingStub stub, TypedDocumentResult source, TypedLayer pii) {
        ProcessingReceipt r = source.getReceipt();
        check(r.getStatus() == ProcessingStatus.PROCESSING_STATUS_COMPLETE, "receipt status " + r.getStatus());
        check(r.getOperationKey().equals(OPERATION), "receipt operation_key " + r.getOperationKey());
        check(r.getInput().equals(source.getInput()) && r.getResult().equals(source.getResult()),
                "the receipt names another input or result");
        check(r.getLayersCount() == source.getProvenanceCount(), "receipt layers and provenance differ in count");
        int piiIndex = -1;
        for (int i = 0; i < r.getLayersCount(); i++) {
            LayerReceipt l = r.getLayers(i);
            LayerProvenance p = source.getProvenanceList().stream()
                    .filter(x -> x.getLayer().equals(l.getLayer()) && x.getInstance().equals(l.getInstance()))
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("receipt layer " + l.getLayer() + " has no provenance"));
            check(l.hasProvenanceDigest() && l.getProvenanceDigest().equals(p.getProvenanceDigest()),
                    "receipt layer " + l.getLayer() + " has another provenance digest");
            if (l.getLayer().equals(pii.getId()) && l.getInstance().equals(pii.getInstance())) {
                piiIndex = i;
            }
        }
        check(piiIndex >= 0, "the receipt does not name the pii layer");
        LayerReceipt piiReceipt = r.getLayers(piiIndex);
        check(piiReceipt.getItemCount() == pii.getAnnotationsCount(), "the receipt counts other items");
        check(piiReceipt.getAssessedCount() == 1 && piiReceipt.getAssessed(0).getStart() == 0
                && Integer.toUnsignedLong(piiReceipt.getAssessed(0).getEnd()) == utf8(TEXT).length,
                "the pii layer was not assessed over the whole text");
        ProcessingReceipt counted = r.toBuilder()
                .setLayers(piiIndex, piiReceipt.toBuilder().setItemCount(piiReceipt.getItemCount() + 1))
                .build();
        String why = refused(stub, source.toBuilder().setReceipt(counted).build(), "receipt item count");
        check(why.contains("item_count"), "item count refused for another reason: " + why);
        why = refused(stub, source.toBuilder().clearReceipt().build(), "no receipt");
        check(why.contains("no processing receipt"), "missing receipt refused for another reason: " + why);
        ProcessingReceipt other = r.toBuilder()
                .setResult(r.getResult().toBuilder().setResultId("0" + r.getResult().getResultId().substring(1)))
                .build();
        why = refused(stub, source.toBuilder().setReceipt(other).build(), "receipt of another result");
        check(why.contains("receipt.result"), "another result refused for another reason: " + why);
    }

    // Offsets above 2^31: a signed int in Java, read unsigned; the bytes
    // are the five-byte varint of the unsigned value, and Rust reads the
    // same number.
    static void largeOffsets(AnalysisServiceBlockingStub stub, TypedDocumentResult source) throws Exception {
        SpanFragment big = SpanFragment.newBuilder().setStart((int) BIG).setEnd((int) (BIG + 5)).build();
        check(big.getStart() < 0, "the signed view of 3000000000 is not negative");
        check(Integer.toUnsignedLong(big.getStart()) == BIG && Integer.toUnsignedLong(big.getEnd()) == BIG + 5,
                "unsigned access does not give the value back");
        check(big.getSerializedSize() == 12, "two uint32 above 2^31 are not two five-byte varints: " + big.getSerializedSize());
        check(SpanFragment.parseFrom(big.toByteArray()).equals(big), "the large span does not survive its bytes");

        TypedDocumentResult span = withAnnotation(source, "turbo:pii", 0,
                b -> b.setSpans(0, b.getSpans(0).toBuilder().setEnd((int) BIG)));
        String why = refused(stub, span, "span end above 2^31");
        check(why.contains(Long.toString(BIG)), "Rust did not read the span end as " + BIG + ": " + why);

        InlineMapping.Builder mapping = mapping(source).toBuilder();
        ViewChain.Builder view = mapping.getViews(1).toBuilder();
        int last = view.getEditsCount() - 1;
        view.setEdits(last, view.getEdits(last).toBuilder().setParentStart((int) BIG));
        mapping.setViews(1, view);
        TypedDocumentResult run = source.toBuilder()
                .setMapping(MappingPayload.newBuilder().setInline(mapping))
                .build();
        EditRun moved = mapping(run).getViews(1).getEdits(last);
        check(Integer.toUnsignedLong(moved.getParentStart()) == BIG, "edit run start");
        long runEnd = BIG + Integer.toUnsignedLong(moved.getParentWidth()) * Integer.toUnsignedLong(moved.getCount());
        why = refused(stub, run, "edit run above 2^31");
        check(why.contains(Long.toString(runEnd)), "Rust did not read the run as ending at " + runEnd + ": " + why);
    }

    // Unknown fields: Java keeps and re-encodes them; Rust drops them on
    // decode and never forwards them.
    static void unknownFields(AnalysisServiceBlockingStub stub, TypedDocumentResult source) throws Exception {
        byte[] bytes = source.toByteArray();
        byte[] withUnknown = Arrays.copyOf(bytes, bytes.length + UNKNOWN_FIELD.length);
        System.arraycopy(UNKNOWN_FIELD, 0, withUnknown, bytes.length, UNKNOWN_FIELD.length);
        TypedDocumentResult parsed = TypedDocumentResult.parseFrom(withUnknown);
        check(parsed.getUnknownFields().hasField(999), "Java dropped unknown field 999");
        check(Arrays.equals(parsed.toByteArray(), withUnknown), "Java did not re-encode unknown field 999");
        TypedDocumentResult back = stub.withDeadlineAfter(60, TimeUnit.SECONDS).roundTripTypedDocument(parsed);
        check(back.getUnknownFields().asMap().isEmpty(), "Rust forwarded an unknown field");
        check(Arrays.equals(back.toByteArray(), bytes), "Rust's answer is not the message without field 999");

        UnknownFieldSet nested = UnknownFieldSet.newBuilder()
                .addField(999, UnknownFieldSet.Field.newBuilder().addVarint(7).build())
                .build();
        TypedDocumentResult inMention = withAnnotation(source, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setUnknownFields(nested)));
        check(count(inMention.toByteArray(), UNKNOWN_FIELD) == 1, "Java did not encode the nested unknown field");
        back = stub.withDeadlineAfter(60, TimeUnit.SECONDS).roundTripTypedDocument(inMention);
        check(Arrays.equals(back.toByteArray(), bytes), "Rust forwarded a nested unknown field");

        // A value arm this build does not know (field 200) is an unknown
        // field to Java and no value to Rust, which refuses the annotation
        // rather than reading it as nothing to decide.
        TypedAnnotation known = layer(source, "turbo:pii").getAnnotations(0);
        byte[] arm = known.toBuilder().clearValue().build().toByteArray();
        byte[] unknownArm = {(byte) 0xc2, 0x0c, 0x02, 0x08, 0x01};
        byte[] annotation = Arrays.copyOf(arm, arm.length + unknownArm.length);
        System.arraycopy(unknownArm, 0, annotation, arm.length, unknownArm.length);
        TypedAnnotation parsedArm = TypedAnnotation.parseFrom(annotation);
        check(parsedArm.getValueCase() == TypedAnnotation.ValueCase.VALUE_NOT_SET && parsedArm.getUnknownFields().hasField(200),
                "Java read the unknown arm as a value");
        String why = refused(stub, withAnnotation(source, "turbo:pii", 0, b -> parsedArm.toBuilder()), "unknown value arm");
        check(why.contains("missing its typed annotation value"), "unknown arm refused for another reason: " + why);
    }

    // Unknown enum numbers: Java keeps the number as UNRECOGNIZED and
    // re-encodes it; Rust refuses it wherever a decision reads it, and
    // never downgrades it to a known value.
    static void unknownEnums(AnalysisServiceBlockingStub stub, TypedDocumentResult source) throws Exception {
        int li = layerIndex(source, "turbo:pii");
        TypedLayer scoped = source.getLayers(li).toBuilder().setScopeValue(UNKNOWN_ENUM).build();
        check(scoped.getScope() == AnchorKind.UNRECOGNIZED && scoped.getScopeValue() == UNKNOWN_ENUM, "scope 99 in Java");
        check(TypedLayer.parseFrom(scoped.toByteArray()).getScopeValue() == UNKNOWN_ENUM, "Java re-encodes scope 99");
        String why = refused(stub, source.toBuilder().setLayers(li, scoped).build(), "unknown scope");
        check(why.contains("99"), "unknown scope refused without naming it: " + why);

        TypedLayer status = source.getLayers(li).toBuilder().setStatusValue(UNKNOWN_ENUM).build();
        check(status.getStatus() == LayerWireStatus.UNRECOGNIZED, "status 99 in Java");
        why = refused(stub, source.toBuilder().setLayers(li, status).build(), "unknown layer status");
        check(why.contains("99"), "unknown status refused without naming it: " + why);

        TypedDocumentResult offsets = source.toBuilder().setOffsetsValue(UNKNOWN_ENUM).build();
        check(offsets.getOffsets() == OffsetMode.UNRECOGNIZED, "offsets 99 in Java");
        why = refused(stub, offsets, "unknown offsets");
        check(why.contains("99"), "unknown offsets refused without naming it: " + why);

        ProcessingReceipt receipt = source.getReceipt().toBuilder().setStatusValue(UNKNOWN_ENUM).build();
        check(receipt.getStatus() == ProcessingStatus.UNRECOGNIZED, "receipt status 99 in Java");
        why = refused(stub, source.toBuilder().setReceipt(receipt).build(), "unknown receipt status");
        check(why.contains("99"), "unknown receipt status refused without naming it: " + why);
    }

    // Disclosure: where the planted value is in the bytes. The source view
    // shows it in raw_text and in PiiMention.normalized; the masked view's
    // raw_text does not show it, and until values are withheld by default
    // (#1904) PiiMention.normalized is the one place it may appear, once
    // per mention of it. Searched in the bytes, not in toString().
    static void disclosure(TypedDocumentResult source, TypedDocumentResult masked, TypedLayer pii) {
        byte[] value = utf8(VALUE);
        long mentionsOfValue = pii.getAnnotationsList().stream().filter(a -> a.getPii().getNormalized().equals(VALUE)).count();
        check(mentionsOfValue == 1, "the value is normalized " + mentionsOfValue + " times");
        check(count(source.toByteArray(), value) == 2, "the source view bytes hold the value other than in raw_text and once normalized");
        check(count(utf8(masked.getRawText()), value) == 0, "the masked raw_text holds the value");
        check(count(masked.toByteArray(), value) == mentionsOfValue,
                "the masked view bytes hold the value outside PiiMention.normalized");
        check(count(utf8(masked.getReceipt().toString()), value) == 0, "the receipt prints the value");
        check(count(masked.getInput().toByteArray(), value) == 0 && count(masked.getResult().toByteArray(), value) == 0,
                "an identity holds the value");
    }

    // The mapping port against Rust on the #1907 chained-view cases.
    static int mappingCases(AnalysisServiceBlockingStub stub) throws Exception {
        check(Files.isRegularFile(CASES), "no " + CASES + " under the working directory (run from the repository root)");
        List<String> lines = Files.readAllLines(CASES, StandardCharsets.UTF_8);
        int cases = 0;
        int projected = 0;
        int i = 0;
        while (i < lines.size()) {
            String line = lines.get(i++);
            if (!line.startsWith("case ")) {
                continue;
            }
            String[] head = line.split(" ");
            String name = head[1];
            int views = Integer.parseInt(head[3]);
            String graphLine = lines.get(i++);
            check(graphLine.startsWith("graph "), name + ": no graph line");
            ByteString graph = ByteString.copyFrom(fromHex(graphLine.substring("graph ".length())));
            TypedDocumentResult[] shown = new TypedDocumentResult[views];
            for (int v = 0; v < views; v++) {
                shown[v] = inspect(stub, graph, v);
                check(mapping(shown[v]).equals(mapping(shown[0])), name + ": the mapping differs in view " + v);
            }
            WireChainPort port = WireChainPort.of(mapping(shown[0]));
            check(port.size() == views, name + ": the chain has " + port.size() + " views, not " + views);
            for (int v = 0; v < views; v++) {
                check(port.textLen(v) == utf8(shown[v].getRawText()).length, name + ": view " + v + " text_len");
            }
            int pairs = 0;
            while (!lines.get(i).equals("end")) {
                String[] map = lines.get(i++).split(" ");
                check(map[0].equals("map"), name + ": unexpected line " + lines.get(i - 1));
                int from = Integer.parseInt(map[1]);
                int to = Integer.parseInt(map[2]);
                List<long[]> spans = allSpans(shown[from].getRawText());
                check(spans.size() == map.length - 3, name + ": " + spans.size() + " spans and " + (map.length - 3) + " expected");
                for (int s = 0; s < spans.size(); s++) {
                    long[] want = parseSpan(map[3 + s]);
                    long[] got = port.mapSpan(from, to, spans.get(s)[0], spans.get(s)[1]);
                    check(Arrays.equals(got, want), name + ": span " + spans.get(s)[0] + ".." + spans.get(s)[1]
                            + " from view " + from + " to view " + to + " maps to " + got[0] + ".." + got[1]
                            + ", Rust says " + want[0] + ".." + want[1]);
                }
                pairs++;
            }
            i++;
            check(pairs == views * views, name + ": " + pairs + " view pairs");
            projected += projections(name, shown, port);
            cases++;
        }
        check(projected > 0, "no layer projection was held to the port");
        return cases;
    }

    /** The server's own projections of every layer into every view are the port's; how many were checked. */
    static int projections(String name, TypedDocumentResult[] shown, WireChainPort port) {
        int checked = 0;
        for (TypedLayer layer : shown[0].getLayersList()) {
            if (!layer.hasNativeView() || layer.getAnnotationsCount() == 0) {
                continue;
            }
            int home = layer.getNativeView();
            TypedLayer nativeLayer = layer(shown[home], layer.getId());
            for (int v = 0; v < shown.length; v++) {
                TypedLayer there = layer(shown[v], layer.getId());
                for (int a = 0; a < nativeLayer.getAnnotationsCount(); a++) {
                    SpanFragment n = nativeLayer.getAnnotations(a).getSpans(0);
                    long[] got = port.mapSpan(home, v, Integer.toUnsignedLong(n.getStart()), Integer.toUnsignedLong(n.getEnd()));
                    SpanFragment s = there.getAnnotations(a).getSpans(0);
                    check(there.getAnnotations(a).getSpansCount() == 1 && Integer.toUnsignedLong(s.getStart()) == got[0]
                            && Integer.toUnsignedLong(s.getEnd()) == got[1], name + ": " + layer.getId() + "[" + a
                            + "] in view " + v + " is " + s.getStart() + ".." + s.getEnd() + ", the port maps it to "
                            + got[0] + ".." + got[1]);
                    checked++;
                }
            }
        }
        return checked;
    }

    /** Every span on code point boundaries of {@code text} in UTF-8 bytes, by start then end. */
    static List<long[]> allSpans(String text) {
        List<Long> bounds = new ArrayList<>();
        long at = 0;
        bounds.add(at);
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            at += new String(Character.toChars(cp)).getBytes(StandardCharsets.UTF_8).length;
            bounds.add(at);
            i += Character.charCount(cp);
        }
        List<long[]> out = new ArrayList<>();
        for (int s = 0; s < bounds.size(); s++) {
            for (int e = s; e < bounds.size(); e++) {
                out.add(new long[] {bounds.get(s), bounds.get(e)});
            }
        }
        return out;
    }

    static long[] parseSpan(String s) {
        int colon = s.indexOf(':');
        return new long[] {Long.parseLong(s.substring(0, colon)), Long.parseLong(s.substring(colon + 1))};
    }

    static byte[] fromHex(String hex) {
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(2 * i, 2 * i + 2), 16);
        }
        return out;
    }
}

/**
 * A port of the server's wire chain consumer ({@code WireChain} in
 * {@code server/src/results/typed/chain.rs}): the views rebuilt from an
 * {@code InlineMapping}, spans mapped between any two of them stepwise
 * along the lowest-common-ancestor path, up one {@code spanToParent} at
 * a time and down one {@code spanToChild} at a time, never through a
 * composed alignment. A start widens to the start of an edit it falls
 * inside and an end to that edit's end; an empty span at inserted or
 * deleted bytes stays empty and lands before them. Bounds only: the
 * chain holds byte lengths, not texts. Offsets are {@code long} because
 * a protobuf {@code uint32} is a signed {@code int} in Java.
 */
final class WireChainPort {
    /** One edit: {@code parent[ps..pe)} became {@code child[cs..ce)}. */
    record Edit(long ps, long pe, long cs, long ce) {}

    record View(int id, int parent, long textLen, List<Edit> edits) {}

    private final List<View> views;

    private WireChainPort(List<View> views) {
        this.views = views;
    }

    static WireChainPort of(InlineMapping mapping) {
        Conformance.check(mapping.getMappingSchemaVersion() == 1, "mapping schema " + mapping.getMappingSchemaVersion());
        List<View> views = new ArrayList<>();
        for (int i = 0; i < mapping.getViewsCount(); i++) {
            ViewChain v = mapping.getViews(i);
            Conformance.check(v.getView() == i, "mapping view " + v.getView() + " at position " + i);
            int parent = v.hasParent() ? v.getParent() : -1;
            Conformance.check(parent < i, "view " + i + " names parent " + parent + ", which is not before it");
            List<Edit> edits = new ArrayList<>();
            for (EditRun run : v.getEditsList()) {
                long ps = Integer.toUnsignedLong(run.getParentStart());
                long cs = Integer.toUnsignedLong(run.getChildStart());
                long pw = Integer.toUnsignedLong(run.getParentWidth());
                long cw = Integer.toUnsignedLong(run.getChildWidth());
                long count = Integer.toUnsignedLong(run.getCount());
                Conformance.check(count > 0 && (pw > 0 || cw > 0), "a run with no edits or that changes nothing");
                for (long k = 0; k < count; k++) {
                    edits.add(new Edit(ps, ps + pw, cs, cs + cw));
                    ps += pw;
                    cs += cw;
                }
            }
            Conformance.check(edits.size() == Integer.toUnsignedLong(v.getEditCount()), "view " + i + " edit_count");
            views.add(new View(i, parent, Integer.toUnsignedLong(v.getTextLen()), edits));
        }
        return new WireChainPort(views);
    }

    int size() {
        return views.size();
    }

    long textLen(int view) {
        return views.get(view).textLen();
    }

    /** Maps byte span {@code [start, end)} of view {@code from} into view {@code to}. */
    long[] mapSpan(int from, int to, long start, long end) {
        Conformance.check(start <= end && end <= views.get(from).textLen(), "span " + start + ".." + end + " outside view " + from);
        List<View> up = new ArrayList<>();
        for (View v = views.get(from); v != null; v = v.parent() < 0 ? null : views.get(v.parent())) {
            up.add(v);
        }
        List<View> down = new ArrayList<>();
        View cur = views.get(to);
        while (true) {
            int pos = -1;
            for (int i = 0; i < up.size(); i++) {
                if (up.get(i).id() == cur.id()) {
                    pos = i;
                }
            }
            if (pos >= 0) {
                up = up.subList(0, pos);
                break;
            }
            Conformance.check(cur.parent() >= 0, "views " + from + " and " + to + " share no ancestor");
            down.add(cur);
            cur = views.get(cur.parent());
        }
        java.util.Collections.reverse(down);
        long s = start;
        long e = end;
        for (View v : up) {
            long ns = map(v.edits(), s, true, false);
            long ne = map(v.edits(), e, false, false);
            s = Math.min(ns, ne);
            e = ne;
        }
        for (View v : down) {
            long ns = map(v.edits(), s, true, true);
            long ne = map(v.edits(), e, false, true);
            s = Math.min(ns, ne);
            e = ne;
        }
        return new long[] {s, e};
    }

    /**
     * One offset across one alignment: {@code x} in the child mapped to the
     * parent, or with {@code toChild} the other way. {@code isStart} picks the
     * side rule: a start belongs to the edit it is inside and widens to its
     * start, an end to the edit it ends inside and widens to its end.
     */
    static long map(List<Edit> edits, long x, boolean isStart, boolean toChild) {
        int k = 0;
        while (k < edits.size()) {
            long fromEnd = toChild ? edits.get(k).pe() : edits.get(k).ce();
            if (isStart ? fromEnd <= x : fromEnd < x) {
                k++;
            } else {
                break;
            }
        }
        if (k < edits.size()) {
            Edit e = edits.get(k);
            long fs = toChild ? e.ps() : e.cs();
            long fe = toChild ? e.pe() : e.ce();
            long ts = toChild ? e.cs() : e.ps();
            long te = toChild ? e.ce() : e.pe();
            if (fs < x && x <= fe && !(isStart && x == fe)) {
                return isStart ? ts : te;
            }
        }
        if (k == 0) {
            return x;
        }
        Edit prev = edits.get(k - 1);
        long fromEnd = toChild ? prev.pe() : prev.ce();
        long toEnd = toChild ? prev.ce() : prev.pe();
        return toEnd + (x - fromEnd);
    }
}
