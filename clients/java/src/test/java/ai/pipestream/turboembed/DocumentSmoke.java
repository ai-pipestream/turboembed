package ai.pipestream.turboembed;

import com.google.protobuf.ByteString;
import com.google.protobuf.UnknownFieldSet;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.io.File;
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
import turbo.result.v1.TurboPiiPolicy.PiiAbbreviateStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiAgeSafeHarborStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiDateStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiDateYear;
import turbo.result.v1.TurboPiiPolicy.PiiDefaultStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiFormatStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiKeepSide;
import turbo.result.v1.TurboPiiPolicy.PiiKeyedTokenStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiKindRule;
import turbo.result.v1.TurboPiiPolicy.PiiLastNStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiMapEntries;
import turbo.result.v1.TurboPiiPolicy.PiiMapStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiMapTarget;
import turbo.result.v1.TurboPiiPolicy.PiiMaskStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiPolicyKeyRef;
import turbo.result.v1.TurboPiiPolicy.PiiPseudonymStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiRedactionSelection;
import turbo.result.v1.TurboPiiPolicy.PiiRewriteStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiRuleState;
import turbo.result.v1.TurboPiiPolicy.PiiSha256Strategy;
import turbo.result.v1.TurboPiiPolicy.PiiStaticStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiStrategyEntry;
import turbo.result.v1.TurboPiiPolicy.PiiTruncateStrategy;
import turbo.result.v1.TurboPiiPolicy.PiiZipSafeHarborStrategy;
import turbo.result.v1.TurboPiiPolicy.RedactionPolicy;
import turbo.result.v1.TurboTypedAnnotations.AnchorKind;
import turbo.result.v1.TurboTypedAnnotations.EditRun;
import turbo.result.v1.TurboTypedAnnotations.FragmentMember;
import turbo.result.v1.TurboTypedAnnotations.InlineMapping;
import turbo.result.v1.TurboTypedAnnotations.InspectDocumentGraphRequest;
import turbo.result.v1.TurboTypedAnnotations.LayerProvenance;
import turbo.result.v1.TurboTypedAnnotations.LayerReceipt;
import turbo.result.v1.TurboTypedAnnotations.LayerWireStatus;
import turbo.result.v1.TurboTypedAnnotations.MappingPayload;
import turbo.result.v1.TurboTypedAnnotations.OffsetMode;
import turbo.result.v1.TurboTypedAnnotations.PiiCandidateAnnotation;
import turbo.result.v1.TurboTypedAnnotations.PiiConfidenceTier;
import turbo.result.v1.TurboTypedAnnotations.PiiDisposition;
import turbo.result.v1.TurboTypedAnnotations.PiiEditAnnotation;
import turbo.result.v1.TurboTypedAnnotations.PiiEvidence;
import turbo.result.v1.TurboTypedAnnotations.PiiKindId;
import turbo.result.v1.TurboTypedAnnotations.PiiKindRef;
import turbo.result.v1.TurboTypedAnnotations.PiiMention;
import turbo.result.v1.TurboTypedAnnotations.PiiOutputOptions;
import turbo.result.v1.TurboTypedAnnotations.PiiPlannedRewrite;
import turbo.result.v1.TurboTypedAnnotations.PiiStrategy;
import turbo.result.v1.TurboTypedAnnotations.PiiValue;
import turbo.result.v1.TurboTypedAnnotations.PiiValueCiphertext;
import turbo.result.v1.TurboTypedAnnotations.PiiValueForm;
import turbo.result.v1.TurboTypedAnnotations.PiiValueToken;
import turbo.result.v1.TurboTypedAnnotations.PiiValueWithheld;
import turbo.result.v1.TurboTypedAnnotations.PiiWithheldReason;
import turbo.result.v1.TurboTypedAnnotations.ProcessingReceipt;
import turbo.result.v1.TurboTypedAnnotations.ProcessingStatus;
import turbo.result.v1.TurboTypedAnnotations.ScopedRef;
import turbo.result.v1.TurboTypedAnnotations.SpanFragment;
import turbo.result.v1.TurboTypedAnnotations.SpanOutcome;
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
 * the same server (issue #1910), then the redaction policy cases of
 * {@link Redaction} (issue #1882 part C).
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
        boolean mtls = args.length == 4;
        ManagedChannel channel = mtls
                ? NettyChannelBuilder.forTarget(target)
                        .sslContext(GrpcSslContexts.forClient().trustManager(new File(args[1]))
                                .keyManager(new File(args[2]), new File(args[3])).build())
                        .build()
                : ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        try {
            AnalysisServiceBlockingStub stub = AnalysisServiceGrpc.newBlockingStub(channel);
            if (mtls) {
                Conformance.run(stub, true);
                return;
            }
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
            Conformance.run(stub, false);
            Redaction.run(stub);
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
    /** The document holds PII: its text and graph only come back when asked for (#1904). */
    static final PiiOutputOptions FULL_TEXT = PiiOutputOptions.newBuilder().setFullText(true).build();
    static final PiiOutputOptions FINDINGS_ONLY = PiiOutputOptions.getDefaultInstance();
    static final String CASES_PROPERTY = "turbo.mapping.cases";
    static final Path CASES = Path.of("clients", "java", "src", "test", "resources", "mapping-cases.txt");
    /** Projected fragments seen per {@link SpanOutcome} number over the mapping cases. */
    static final int[] OUTCOMES = new int[4];
    /** Coalesced {@link FragmentMember} outcomes seen per {@link SpanOutcome} number over the mapping cases. */
    static final int[] MEMBER_OUTCOMES = new int[4];

    private Conformance() {}

    static void check(boolean ok, String what) {
        DocumentSmoke.check(ok, what);
    }

    static void run(AnalysisServiceBlockingStub stub, boolean requireRelease) throws Exception {
        ByteString graph = analyze(stub, TEXT, List.of("turbo:pii", "turbo:pii-masked"));
        TypedDocumentResult source = inspect(stub, graph, 0, FULL_TEXT);
        TypedDocumentResult masked = inspect(stub, graph, 1, FULL_TEXT);
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
        disclosure(stub, graph, source, masked, pii);
        releasedValues(stub, graph, source, pii, requireRelease);
        editsAndCandidates(stub);
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
                        .setPiiOutput(FULL_TEXT)
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

    static TypedDocumentResult inspect(AnalysisServiceBlockingStub stub, ByteString graph, int view,
            PiiOutputOptions piiOutput) {
        return stub.withDeadlineAfter(60, TimeUnit.SECONDS).inspectDocumentGraph(InspectDocumentGraphRequest.newBuilder()
                .setDocId(DOC)
                .setGraph(graph)
                .setView(view)
                .setOffsets(OffsetMode.OFFSET_MODE_UTF8)
                .setOperationKey(OPERATION)
                .setPiiOutput(piiOutput)
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
    @SuppressWarnings("deprecation")
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
        PiiMention mention = first.getPii();
        check(mention.getKindRef().getKindCase() == PiiKindRef.KindCase.BUILTIN
                && mention.getKindRef().getBuiltin() == PiiKindId.PII_KIND_ID_EMAIL,
                "pii[0] kind_ref " + mention.getKindRef());
        check(mention.getConfidence() == PiiConfidenceTier.PII_CONFIDENCE_TIER_FORMAT,
                "pii[0] confidence " + mention.getConfidence());
        check(mention.getKind().isEmpty() && mention.getConfidenceTier().isEmpty(),
                "pii[0] populates a retired string field");
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
        check(onMasked.getPii().getKindRef().equals(onSource.getPii().getKindRef()), "another mention on the masked view");
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

        // A kind id or confidence tier this build does not know is what a
        // redaction decision would need: refused, never read as a built-in.
        TypedDocumentResult kind = withAnnotation(source, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setKindRef(PiiKindRef.newBuilder().setBuiltinValue(UNKNOWN_ENUM))));
        PiiKindRef unknownKind = layer(kind, "turbo:pii").getAnnotations(0).getPii().getKindRef();
        check(unknownKind.getBuiltin() == PiiKindId.UNRECOGNIZED && unknownKind.getBuiltinValue() == UNKNOWN_ENUM,
                "kind id 99 in Java");
        why = refused(stub, kind, "unknown pii kind id");
        check(why.contains("not a built-in kind"), "unknown kind id refused for another reason: " + why);
        TypedDocumentResult tier = withAnnotation(source, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setConfidenceValue(UNKNOWN_ENUM)));
        check(layer(tier, "turbo:pii").getAnnotations(0).getPii().getConfidence() == PiiConfidenceTier.UNRECOGNIZED,
                "confidence 99 in Java");
        why = refused(stub, tier, "unknown pii confidence tier");
        check(why.contains("not a known tier"), "unknown tier refused for another reason: " + why);

        ProcessingReceipt receipt = source.getReceipt().toBuilder().setStatusValue(UNKNOWN_ENUM).build();
        check(receipt.getStatus() == ProcessingStatus.UNRECOGNIZED, "receipt status 99 in Java");
        why = refused(stub, source.toBuilder().setReceipt(receipt).build(), "unknown receipt status");
        check(why.contains("99"), "unknown receipt status refused without naming it: " + why);
    }

    // Disclosure (#1904): where the planted value is in the bytes. The
    // value of every mention is withheld and the retired normalized string
    // is never populated, whatever is asked for. By default (findings only)
    // no view carries its text and no answer holds the value at all; with
    // full_text the source view holds it once, in raw_text; with
    // rewritten_text the masked view shows its own text and not the value.
    // Searched in the bytes, not in toString().
    @SuppressWarnings("deprecation")
    static void disclosure(AnalysisServiceBlockingStub stub, ByteString graph, TypedDocumentResult source,
            TypedDocumentResult masked, TypedLayer pii) {
        byte[] value = utf8(VALUE);
        for (TypedAnnotation a : pii.getAnnotationsList()) {
            check(a.getPii().getValue().getFormCase() == PiiValue.FormCase.WITHHELD, "a mention value is not withheld");
            check(a.getPii().getNormalized().isEmpty(), "a mention populates the retired normalized string");
        }
        check(count(source.toByteArray(), value) == 1, "the full source view holds the value other than in raw_text");
        check(count(utf8(masked.getRawText()), value) == 0, "the masked raw_text holds the value");
        check(count(masked.toByteArray(), value) == 0, "the masked view bytes hold the value");
        check(count(utf8(masked.getReceipt().toString()), value) == 0, "the receipt prints the value");
        check(count(masked.getInput().toByteArray(), value) == 0 && count(masked.getResult().toByteArray(), value) == 0,
                "an identity holds the value");
        for (int view = 0; view < 2; view++) {
            TypedDocumentResult findings = inspect(stub, graph, view, FINDINGS_ONLY);
            check(findings.getRawText().isEmpty() && findings.hasWithheldTextLen(), "view " + view + " carries its text");
            check(count(findings.toByteArray(), value) == 0, "the findings-only view " + view + " holds the value");
            TypedLayer found = layer(findings, "turbo:pii");
            check(found.getAnnotationsList().equals(layer(view == 0 ? source : masked, "turbo:pii").getAnnotationsList()),
                    "findings-only view " + view + " lists other mentions");
            roundTrip(stub, findings);
        }
        check(inspect(stub, graph, 0, FINDINGS_ONLY).getWithheldTextLen() == utf8(TEXT).length,
                "withheld_text_len is not the source text length");
        PiiOutputOptions rewritten = PiiOutputOptions.newBuilder().setRewrittenText(true).build();
        TypedDocumentResult maskedOnly = inspect(stub, graph, 1, rewritten);
        check(maskedOnly.getRawText().equals(masked.getRawText()), "rewritten_text shows another masked text");
        check(count(maskedOnly.toByteArray(), value) == 0, "the rewritten view holds the value");
        check(inspect(stub, graph, 0, rewritten).getRawText().isEmpty(), "rewritten_text shows the source text");
    }

    /**
     * Released value forms (#1909, #1938): a {@link PiiValue} other than
     * {@code withheld} is produced by {@code InspectDocumentGraph} only under a
     * grant.
     *
     * <ul>
     *   <li>The default answer keeps {@code withheld} with
     *       {@code PII_WITHHELD_REASON_NOT_REQUESTED}; missing, unspecified or
     *       unknown withheld reasons are refused by {@code RoundTripTypedDocument}.
     *   <li>An unknown {@link PiiValue} oneof arm is an unknown field to Java
     *       ({@code FORM_NOT_SET}, {@code getClear().isEmpty()}, never shown as
     *       clear text) and is refused by Rust as a value with no form.
     *   <li>A 32-byte {@link PiiValueToken} naming its key id and version
     *       round-trips through Java and {@code RoundTripTypedDocument}, and its
     *       bytes do not contain the planted value; malformed tokens and
     *       ciphertext values (waiting for #1885) are refused without echoing
     *       their contents.
     *   <li>Live {@code InspectDocumentGraph} release of {@code PII_VALUE_FORM_TOKEN}
     *       requires mTLS ({@code --tls-cert}, {@code --tls-key},
     *       {@code --tls-client-ca}) with a verified client certificate granted
     *       {@code token} in {@code --pii-release-config} and an audit service
     *       acknowledgment. Where the test server has that grant configured, the
     *       live token response is checked to carry 32-byte tokens and never the
     *       planted value; on the plaintext {@code proto} lane server started by
     *       {@code scripts/package-proto-stubs.sh} without mTLS or release
     *       config, the request is refused with {@code UNIMPLEMENTED} and no
     *       planted value in the status.
     * </ul>
     */
    static void releasedValues(AnalysisServiceBlockingStub stub, ByteString graph, TypedDocumentResult source,
            TypedLayer pii, boolean requireRelease) throws Exception {
        byte[] value = utf8(VALUE);
        for (TypedAnnotation a : pii.getAnnotationsList()) {
            PiiValue v = a.getPii().getValue();
            check(v.getFormCase() == PiiValue.FormCase.WITHHELD, "default answer form " + v.getFormCase());
            check(v.getWithheld().getReason() == PiiWithheldReason.PII_WITHHELD_REASON_NOT_REQUESTED,
                    "default withheld reason " + v.getWithheld().getReason());
        }
        TypedDocumentResult findings = inspect(stub, graph, 0, FINDINGS_ONLY);
        for (TypedAnnotation a : layer(findings, "turbo:pii").getAnnotationsList()) {
            PiiValue v = a.getPii().getValue();
            check(v.getFormCase() == PiiValue.FormCase.WITHHELD
                    && v.getWithheld().getReason() == PiiWithheldReason.PII_WITHHELD_REASON_NOT_REQUESTED,
                    "findings-only mention value is not withheld(NOT_REQUESTED)");
        }

        // An unknown PiiValue oneof arm (field 999) is an unknown field to Java,
        // never shown as clear text, and refused by Rust on RoundTripTypedDocument.
        PiiValue unknownArm = PiiValue.parseFrom(UNKNOWN_FIELD);
        check(unknownArm.getFormCase() == PiiValue.FormCase.FORM_NOT_SET
                && unknownArm.getUnknownFields().hasField(999)
                && unknownArm.getClear().isEmpty(),
                "Java read the unknown PiiValue arm as a known form or clear text");
        TypedDocumentResult withUnknownValue = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(unknownArm)));
        check(layer(TypedDocumentResult.parseFrom(withUnknownValue.toByteArray()), "turbo:pii")
                .getAnnotations(0).getPii().getValue().getUnknownFields().hasField(999),
                "Java dropped the unknown PiiValue field on re-encode");
        String why = refused(stub, withUnknownValue, "unknown PiiValue arm");
        check(why.contains("pii mention value has no form"), "unknown PiiValue arm refused for another reason: " + why);

        // Missing or unrecognized withheld reasons and ciphertext (#1885) are refused.
        TypedDocumentResult unspecifiedReason = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(PiiValue.newBuilder()
                        .setWithheld(PiiValueWithheld.newBuilder()
                                .setReason(PiiWithheldReason.PII_WITHHELD_REASON_UNSPECIFIED)))));
        why = refused(stub, unspecifiedReason, "unspecified withheld reason");
        check(why.contains("without a known reason"), "unspecified withheld reason: " + why);

        TypedDocumentResult unknownReason = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(PiiValue.newBuilder()
                        .setWithheld(PiiValueWithheld.newBuilder().setReasonValue(UNKNOWN_ENUM)))));
        check(layer(unknownReason, "turbo:pii").getAnnotations(0).getPii().getValue().getWithheld().getReason()
                == PiiWithheldReason.UNRECOGNIZED, "withheld reason 99 in Java");
        why = refused(stub, unknownReason, "unknown withheld reason");
        check(why.contains("without a known reason"), "unknown withheld reason: " + why);

        TypedDocumentResult ciphertext = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(PiiValue.newBuilder()
                        .setCiphertext(PiiValueCiphertext.newBuilder()
                                .setCiphertext(ByteString.copyFromUtf8("planted-cipher"))
                                .setKeyId("planted-key")
                                .setKeyVersion(1)
                                .setScheme("aes-gcm")))));
        why = refused(stub, ciphertext, "ciphertext PiiValue");
        check(why.contains("ciphertext") && !why.contains("planted-cipher") && !why.contains("planted-key"),
                "ciphertext refusal: " + why);

        // A 32-byte keyed token carries no planted value and survives the round trip.
        byte[] hmacBytes = new byte[32];
        Arrays.fill(hmacBytes, (byte) 0x5a);
        PiiValueToken token = PiiValueToken.newBuilder()
                .setToken(ByteString.copyFrom(hmacBytes))
                .setKeyId("conformance-hmac")
                .setKeyVersion(3)
                .build();
        check(count(token.getToken().toByteArray(), value) == 0 && count(token.toByteArray(), value) == 0,
                "token bytes hold the planted value");
        TypedDocumentResult withToken = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(PiiValue.newBuilder().setToken(token))));
        check(count(withToken.toByteArray(), value) == 0, "the token-bearing findings message holds the planted value");
        TypedDocumentResult tokenBack = roundTrip(stub, withToken);
        PiiValue tokenValueBack = layer(tokenBack, "turbo:pii").getAnnotations(0).getPii().getValue();
        check(tokenValueBack.getFormCase() == PiiValue.FormCase.TOKEN
                && tokenValueBack.getToken().equals(token), "token did not survive RoundTripTypedDocument");

        TypedDocumentResult shortToken = withAnnotation(findings, "turbo:pii", 0,
                b -> b.setPii(b.getPii().toBuilder().setValue(PiiValue.newBuilder()
                        .setToken(token.toBuilder().setToken(ByteString.copyFrom(new byte[16]))))));
        why = refused(stub, shortToken, "short token");
        check(why.contains("not 32 bytes"), "short token refusal: " + why);

        // Live token grant over InspectDocumentGraph: succeeds only when the
        // server has mTLS and --pii-release-config granting token access;
        // refused with UNIMPLEMENTED on the plaintext proto lane server.
        PiiOutputOptions askToken = PiiOutputOptions.newBuilder()
                .setValues(PiiValueForm.PII_VALUE_FORM_TOKEN)
                .build();
        try {
            TypedDocumentResult granted = inspect(stub, graph, 0, askToken);
            check(count(granted.toByteArray(), value) == 0, "granted token response holds the planted value");
            int tokensSeen = 0;
            for (TypedAnnotation a : layer(granted, "turbo:pii").getAnnotationsList()) {
                PiiValue gv = a.getPii().getValue();
                if (gv.getFormCase() == PiiValue.FormCase.TOKEN) {
                    PiiValueToken gt = gv.getToken();
                    check(gt.getToken().size() == 32 && !gt.getKeyId().isEmpty() && gt.getKeyVersion() > 0,
                            "granted token shape");
                    check(count(gt.getToken().toByteArray(), value) == 0, "granted token bytes hold the planted value");
                    tokensSeen++;
                }
            }
            check(tokensSeen > 0, "token request succeeded without a granted token");
            roundTrip(stub, granted);
        } catch (StatusRuntimeException e) {
            if (requireRelease) {
                throw e;
            }
            check(e.getStatus().getCode() == Status.Code.UNIMPLEMENTED,
                    "plaintext server without release config answered " + e.getStatus().getCode());
            check(count(utf8(String.valueOf(e.getStatus().getDescription())), value) == 0,
                    "refusal status holds the planted value");
        }
    }

    /**
     * Edits ({@code turbo:pii-edits}, arm 98) and candidates
     * ({@code turbo:pii-candidates}, arm 99) layers through the round trip on a
     * document with two derived views ({@code masked} and {@code redacted},
     * #1902, #1903, #1921, #1938).
     */
    static void editsAndCandidates(AnalysisServiceBlockingStub stub) throws Exception {
        String labelText = "social insurance number";
        String sin = "046-454-286";
        String sinDigits = "046454286";
        String unlabelledImei = "490154203237518";
        String text = "Contact " + VALUE + " or " + labelText + " " + sin
                + " (device " + unlabelledImei + ") by Friday.";
        byte[] emailBytes = utf8(VALUE);
        byte[] sinBytes = utf8(sin);
        byte[] sinDigitBytes = utf8(sinDigits);
        byte[] imeiBytes = utf8(unlabelledImei);
        byte[] labelBytes = utf8(labelText);

        RedactionPolicy policy = Redaction.policy(7, Redaction.everyArm().get(7));
        Iterator<AnalyzeProgressiveEvent> events = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                .analyzeDocumentProgressive(AnalyzePipelineRequest.newBuilder()
                        .setDocId(DOC)
                        .setText(text)
                        .setOffsets(OffsetMode.OFFSET_MODE_UTF8)
                        .addAllSteps(List.of("turbo:pii", "turbo:pii-masked", Redaction.REDACT))
                        .setOperationKey(OPERATION)
                        .setPiiOutput(Redaction.REWRITTEN)
                        .setPiiRedaction(PiiRedactionSelection.newBuilder().setInline(policy))
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
        check(graph != null, "two-view stream ended without the document");

        // 1. With detail requested across all three views (source=0, masked=1, redacted=2).
        for (int v = 0; v < 3; v++) {
            TypedDocumentResult m = inspect(stub, graph, v, Redaction.REWRITTEN);
            check(m.getViewsCount() == 3, "expected source, masked and redacted views, got " + m.getViewsCount());
            check(chainView(m, 1).getInstance().equals("masked") && chainView(m, 2).getInstance().equals("redacted"),
                    "views 1 and 2 are not masked and redacted");

            TypedLayer piiLayer = layer(m, "turbo:pii");
            TypedLayer editsLayer = layer(m, "turbo:pii-edits");
            TypedLayer candidatesLayer = layer(m, "turbo:pii-candidates");
            // The edits' view resolves through TypedLayer.native_view (#1921),
            // which is the redacted view (2), not the first derived view (1).
            check(editsLayer.hasNativeView() && editsLayer.getNativeView() == 2,
                    "turbo:pii-edits native_view is not the redacted view (2): " + editsLayer.getNativeView());
            check(chainView(m, editsLayer.getNativeView()).getInstance().equals("redacted"),
                    "turbo:pii-edits native_view does not resolve to redacted");
            check(editsLayer.getAnnotationsCount() >= 2, "expected at least two edits");
            for (TypedAnnotation a : editsLayer.getAnnotationsList()) {
                check(a.getValueCase() == TypedAnnotation.ValueCase.PII_EDIT
                        && a.getValueCase().getNumber() == 98, "edit annotation arm is not 98");
                PiiEditAnnotation e = a.getPiiEdit();
                check(e.hasSource()
                        && e.getSource().getOutcome() == SpanOutcome.SPAN_OUTCOME_EXACT
                        && e.getSource().getMembersCount() == 0,
                        "edit source is not EXACT with no members on view " + v);
                if (v == editsLayer.getNativeView()) {
                    check(a.getSpans(0).getOutcome() == SpanOutcome.SPAN_OUTCOME_EXACT,
                            "edit span on its own redacted view is not EXACT");
                }
            }

            check(candidatesLayer.getAnnotationsCount() >= 3,
                    "expected kept and rejected candidates, got " + candidatesLayer.getAnnotationsCount());
            boolean hasKept = false;
            boolean hasRejected = false;
            boolean hasLabelFact = false;
            for (TypedAnnotation a : candidatesLayer.getAnnotationsList()) {
                check(a.getValueCase() == TypedAnnotation.ValueCase.PII_CANDIDATE
                        && a.getValueCase().getNumber() == 99, "candidate annotation arm is not 99");
                PiiCandidateAnnotation c = a.getPiiCandidate();
                if (c.getDisposition().getDispositionCase() == PiiDisposition.DispositionCase.KEPT) {
                    hasKept = true;
                } else if (c.getDisposition().getDispositionCase() == PiiDisposition.DispositionCase.REJECTED_BY_RULE) {
                    hasRejected = true;
                }
                for (PiiEvidence ev : c.getEvidenceList()) {
                    if (ev.getFactCase() == PiiEvidence.FactCase.LABEL) {
                        hasLabelFact = true;
                        SpanFragment ls = ev.getLabel().getSpan();
                        if (v == 0) {
                            String sliced = slice(m.getRawText(), Integer.toUnsignedLong(ls.getStart()),
                                    Integer.toUnsignedLong(ls.getEnd()));
                            check(sliced.equals(labelText), "label evidence span slices to " + sliced);
                        }
                    }
                }
            }
            check(hasKept && hasRejected && hasLabelFact,
                    "candidates missing kept, rejected or label fact on view " + v);

            // Neither planted values nor evidence label text appear in the PII layers' bytes.
            for (TypedLayer l : List.of(piiLayer, editsLayer, candidatesLayer)) {
                byte[] lb = l.toByteArray();
                for (byte[] secret : List.of(emailBytes, sinBytes, sinDigitBytes, imeiBytes, labelBytes)) {
                    check(count(lb, secret) == 0, l.getId() + " on view " + v + " carries secret or evidence text");
                }
            }

            // Re-encode in Java and validate through RoundTripTypedDocument.
            TypedDocumentResult reencoded = TypedDocumentResult.parseFrom(m.toByteArray());
            roundTrip(stub, reencoded);

            // Tampering an edit's source span is refused against the chain on both
            // the source view and the redacted native_view (#1921).
            if (v == 0 || v == editsLayer.getNativeView()) {
                TypedDocumentResult tampered = withAnnotation(m, "turbo:pii-edits", 0, b -> {
                    PiiEditAnnotation.Builder eb = b.getPiiEdit().toBuilder();
                    eb.setSource(eb.getSource().toBuilder().setStart(eb.getSource().getStart() + 1));
                    return b.setPiiEdit(eb);
                });
                String why = refused(stub, tampered, "tampered edit source on view " + v);
                check(why.contains("disagrees with"), "tampered edit source on view " + v + ": " + why);
            }
        }

        // Detail without full_text: candidates and edits are present, and no
        // planted value or evidence label text appears anywhere in the bytes.
        PiiOutputOptions detailOnly = PiiOutputOptions.newBuilder().setDetail(true).build();
        for (int v = 0; v < 3; v++) {
            TypedDocumentResult det = inspect(stub, graph, v, detailOnly);
            check(layer(det, "turbo:pii-edits").getAnnotationsCount() >= 2, "detail-only edits on view " + v);
            check(layer(det, "turbo:pii-candidates").getAnnotationsCount() >= 3, "detail-only candidates on view " + v);
            byte[] bytes = det.toByteArray();
            for (byte[] secret : List.of(emailBytes, sinBytes, sinDigitBytes, imeiBytes, labelBytes)) {
                check(count(bytes, secret) == 0, "detail-only view " + v + " holds a planted value or evidence text");
            }
            roundTrip(stub, TypedDocumentResult.parseFrom(bytes));
        }

        // 2. Without detail: the candidates layer has no annotations, and no
        // planted value or evidence text appears in the bytes.
        for (int v = 0; v < 3; v++) {
            TypedDocumentResult bare = inspect(stub, graph, v, FINDINGS_ONLY);
            check(layer(bare, "turbo:pii-candidates").getAnnotationsCount() == 0,
                    "candidates present without detail on view " + v);
            check(layer(bare, "turbo:pii-edits").getAnnotationsCount() >= 2,
                    "edits absent on findings-only view " + v);
            byte[] bytes = bare.toByteArray();
            for (byte[] secret : List.of(emailBytes, sinBytes, sinDigitBytes, imeiBytes, labelBytes)) {
                check(count(bytes, secret) == 0, "findings-only view " + v + " holds a planted value or evidence text");
            }
            roundTrip(stub, TypedDocumentResult.parseFrom(bytes));
        }
        PiiOutputOptions rewrittenOnly = PiiOutputOptions.newBuilder().setRewrittenText(true).build();
        for (int v : List.of(1, 2)) {
            TypedDocumentResult rw = inspect(stub, graph, v, rewrittenOnly);
            check(layer(rw, "turbo:pii-candidates").getAnnotationsCount() == 0,
                    "candidates present on rewritten_text without detail on view " + v);
            byte[] bytes = rw.toByteArray();
            for (byte[] secret : List.of(emailBytes, sinBytes, sinDigitBytes)) {
                check(count(bytes, secret) == 0, "rewritten view " + v + " holds a planted value");
            }
            for (String lid : List.of("turbo:pii", "turbo:pii-edits", "turbo:pii-candidates")) {
                check(count(layer(rw, lid).toByteArray(), labelBytes) == 0,
                        lid + " on rewritten view " + v + " holds evidence label text");
            }
            roundTrip(stub, TypedDocumentResult.parseFrom(bytes));
        }
    }

    static Path casesPath() {
        String override = System.getProperty(CASES_PROPERTY);
        if (override != null && !override.isBlank()) {
            return Path.of(override.trim());
        }
        return CASES;
    }

    static Path requireCasesFile(Path path) {
        check(Files.isRegularFile(path), "no mapping cases file at " + path + " (resolved to "
                + path.toAbsolutePath() + "; run from the repository root or set -D" + CASES_PROPERTY + "=<path>)");
        return path;
    }

    // The mapping port against Rust on the #1907 chained-view cases.
    static int mappingCases(AnalysisServiceBlockingStub stub) throws Exception {
        String previousProp = System.getProperty(CASES_PROPERTY);
        try {
            Path missing = Path.of("nonexistent", "mapping-cases.txt");
            System.setProperty(CASES_PROPERTY, missing.toString());
            check(casesPath().equals(missing), "system property override " + CASES_PROPERTY + " was not read");
            expectRefused(() -> requireCasesFile(casesPath()), missing.toString());
        } finally {
            if (previousProp == null) {
                System.clearProperty(CASES_PROPERTY);
            } else {
                System.setProperty(CASES_PROPERTY, previousProp);
            }
        }
        Path casesFile = requireCasesFile(casesPath());
        List<String> lines = Files.readAllLines(casesFile, StandardCharsets.UTF_8);
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
                shown[v] = inspect(stub, graph, v, FULL_TEXT);
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
            projected += projections(stub, name, shown, port);
            cases++;
        }
        check(projected > 0, "no layer projection was held to the port");
        check(OUTCOMES[SpanOutcome.SPAN_OUTCOME_EXACT_VALUE] > 0 && OUTCOMES[SpanOutcome.SPAN_OUTCOME_WIDENED_VALUE] > 0
                && OUTCOMES[SpanOutcome.SPAN_OUTCOME_DELETED_VALUE] > 0, "the cases show " + Arrays.toString(OUTCOMES)
                + " fragments by outcome number; exact, widened and deleted must all occur");
        check(MEMBER_OUTCOMES[SpanOutcome.SPAN_OUTCOME_EXACT_VALUE] > 0
                && MEMBER_OUTCOMES[SpanOutcome.SPAN_OUTCOME_WIDENED_VALUE] > 0
                && MEMBER_OUTCOMES[SpanOutcome.SPAN_OUTCOME_DELETED_VALUE] > 0,
                "coalesced members show " + Arrays.toString(MEMBER_OUTCOMES)
                        + " by outcome number; exact, widened and deleted must all occur");
        return cases;
    }

    static SpanOutcome outcomeFromChain(WireChainPort port, int from, int to, long start, long end) {
        long[] mapped = port.mapSpan(from, to, start, end);
        long[] back = port.mapSpan(to, from, mapped[0], mapped[1]);
        return start != end && mapped[0] == mapped[1] ? SpanOutcome.SPAN_OUTCOME_DELETED
                : back[0] == start && back[1] == end ? SpanOutcome.SPAN_OUTCOME_EXACT
                : SpanOutcome.SPAN_OUTCOME_WIDENED;
    }

    static SpanOutcome mergeOutcomes(List<SpanOutcome> outcomes) {
        if (outcomes.contains(SpanOutcome.SPAN_OUTCOME_WIDENED)) {
            return SpanOutcome.SPAN_OUTCOME_WIDENED;
        }
        if (outcomes.contains(SpanOutcome.SPAN_OUTCOME_EXACT)) {
            return SpanOutcome.SPAN_OUTCOME_EXACT;
        }
        return SpanOutcome.SPAN_OUTCOME_DELETED;
    }

    /** The server's own projections of every layer into every view are the port's; how many were checked. */
    static int projections(AnalysisServiceBlockingStub stub, String name, TypedDocumentResult[] shown,
            WireChainPort port) {
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
                    List<SpanFragment> nativeFrags = nativeLayer.getAnnotations(a).getSpansList();
                    List<SpanFragment> targetFrags = there.getAnnotations(a).getSpansList();
                    check(!nativeFrags.isEmpty() && !targetFrags.isEmpty(),
                            name + ": " + layer.getId() + "[" + a + "] has no spans");
                    if (v == home) {
                        check(targetFrags.equals(nativeFrags),
                                name + ": " + layer.getId() + "[" + a + "] differs on its own view");
                    }
                    for (int f = 0; f < targetFrags.size(); f++) {
                        SpanFragment s = targetFrags.get(f);
                        long sStart = Integer.toUnsignedLong(s.getStart());
                        long sEnd = Integer.toUnsignedLong(s.getEnd());
                        String where = name + ": " + layer.getId() + "[" + a + "] fragment " + f + " in view " + v;
                        if (s.getMembersCount() == 0) {
                            if (nativeFrags.size() == 1) {
                                check(targetFrags.size() == 1, where + " split one native span");
                            }
                            boolean matched = false;
                            for (SpanFragment n : nativeFrags) {
                                long nStart = Integer.toUnsignedLong(n.getStart());
                                long nEnd = Integer.toUnsignedLong(n.getEnd());
                                long[] got = port.mapSpan(home, v, nStart, nEnd);
                                if (got[0] == sStart && got[1] == sEnd
                                        && outcomeFromChain(port, home, v, nStart, nEnd) == s.getOutcome()) {
                                    matched = true;
                                    break;
                                }
                            }
                            check(matched, where + " is " + sStart + ".." + sEnd + " (" + s.getOutcome()
                                    + "), which no native fragment maps to");
                        } else {
                            // Coalesced fragment (#1921, #1938): re-derive each FragmentMember's
                            // position and outcome from the chain member by member, as
                            // server/tests/typed_outcomes.rs does.
                            check(s.getMembersCount() > 1, where + " lists one member");
                            check(v != home, where + " coalesced on its own view");
                            long unionStart = Long.MAX_VALUE;
                            long unionEnd = Long.MIN_VALUE;
                            int prevIndex = -1;
                            List<SpanOutcome> memberOutcomes = new ArrayList<>();
                            for (FragmentMember m : s.getMembersList()) {
                                int idx = m.getIndex();
                                check(idx > prevIndex && idx < nativeFrags.size(),
                                        where + " member index " + idx + " out of order or bounds");
                                prevIndex = idx;
                                SpanFragment n = nativeFrags.get(idx);
                                long nStart = Integer.toUnsignedLong(n.getStart());
                                long nEnd = Integer.toUnsignedLong(n.getEnd());
                                long[] mapped = port.mapSpan(home, v, nStart, nEnd);
                                SpanOutcome memberWant = outcomeFromChain(port, home, v, nStart, nEnd);
                                check(m.getOutcome() == memberWant,
                                        where + " member " + idx + " says " + m.getOutcome()
                                                + ", the chain says " + memberWant);
                                unionStart = Math.min(unionStart, mapped[0]);
                                unionEnd = Math.max(unionEnd, mapped[1]);
                                memberOutcomes.add(memberWant);
                                MEMBER_OUTCOMES[memberWant.getNumber()]++;
                            }
                            check(sStart == unionStart && sEnd == unionEnd,
                                    where + " coalesced span is " + sStart + ".." + sEnd
                                            + ", members unite to " + unionStart + ".." + unionEnd);
                            SpanOutcome mergedWant = mergeOutcomes(memberOutcomes);
                            check(s.getOutcome() == mergedWant,
                                    where + " coalesced outcome is " + s.getOutcome()
                                            + ", members merge to " + mergedWant);
                            final int fragIndex = f;
                            TypedDocumentResult reversed = withAnnotation(shown[v], layer.getId(), a, b -> {
                                SpanFragment.Builder fb = b.getSpans(fragIndex).toBuilder().clearMembers();
                                for (int k = s.getMembersCount() - 1; k >= 0; k--) {
                                    fb.addMembers(s.getMembers(k));
                                }
                                return b.setSpans(fragIndex, fb);
                            });
                            String why = refused(stub, reversed, where + " reversed members");
                            check(why.contains("ascending native order"), where + " reversed members: " + why);
                        }
                        check(v != home || (s.getOutcome() == SpanOutcome.SPAN_OUTCOME_EXACT && s.getMembersCount() == 0),
                                where + " is not exact on its own view");
                        OUTCOMES[s.getOutcome().getNumber()]++;
                        checked++;
                    }
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

/**
 * The redaction policy from generated Java (issue #1882 part C), on the same
 * live server: a {@code RedactionPolicy} built with every strategy arm runs
 * the {@code turbo-pii:redact} step and a dry run; each redacted view maps
 * back to the source through the exported chain, the result re-encodes and
 * the server accepts it, and an unknown strategy arm or enum number sent by
 * Java is refused, never downgraded. {@code uint32} fields are read
 * unsigned. Test-side code only.
 */
final class Redaction {
    static final String REDACT = "turbo-pii:redact";
    static final List<String> STEPS = List.of("turbo-pii:pii", REDACT);
    static final PiiOutputOptions REWRITTEN = PiiOutputOptions.newBuilder()
            .setFullText(true).setRewrittenText(true).setDetail(true).build();

    private Redaction() {}

    static void check(boolean ok, String what) {
        DocumentSmoke.check(ok, what);
    }

    static PiiRewriteStrategy arm(UnaryOperator<PiiRewriteStrategy.Builder> set) {
        return set.apply(PiiRewriteStrategy.newBuilder()).build();
    }

    /** Every arm the server knows but the keyed token, which needs a registered key. */
    static List<PiiRewriteStrategy> everyArm() {
        return List.of(
                arm(b -> b.setMask(PiiMaskStrategy.newBuilder().setMaskChar("#"))),
                arm(b -> b.setFormat(PiiFormatStrategy.newBuilder().setTemplate("[%t]"))),
                arm(b -> b.setMap(PiiMapStrategy.newBuilder()
                        .setEntries(PiiMapEntries.newBuilder().putEntries(Conformance.VALUE, "J. Doe"))
                        .setTarget(PiiMapTarget.PII_MAP_TARGET_NORMALIZED_OR_SURFACE))),
                arm(b -> b.setTruncate(PiiTruncateStrategy.newBuilder()
                        .setSide(PiiKeepSide.PII_KEEP_SIDE_START).setCount(2))),
                arm(b -> b.setLastN(PiiLastNStrategy.newBuilder().setCount(4).setKeepFormat(true))),
                arm(b -> b.setAbbreviate(PiiAbbreviateStrategy.newBuilder().setSeparator(".").setTrailing("."))),
                arm(b -> b.setSha256(PiiSha256Strategy.newBuilder().setKindPrefix(true))),
                arm(b -> b.setStaticText(PiiStaticStrategy.newBuilder().setReplacement("[E]"))),
                arm(b -> b.setPseudonym(PiiPseudonymStrategy.newBuilder().setPrefix("[").setSuffix("]"))),
                arm(b -> b.setZipSafeHarbor(PiiZipSafeHarborStrategy.newBuilder().setMaskChar("#"))),
                arm(b -> b.setAgeSafeHarbor(PiiAgeSafeHarborStrategy.newBuilder().setMaskChar("#"))),
                arm(b -> b.setDate(PiiDateStrategy.newBuilder().setYear(PiiDateYear.getDefaultInstance()))));
    }

    static RedactionPolicy policy(long revision, PiiRewriteStrategy email) {
        return RedactionPolicy.newBuilder()
                .setId("java-redaction")
                .setRevision(revision)
                .addKindRules(PiiKindRule.newBuilder()
                        .setRuleId("email")
                        .setKind(PiiKindRef.newBuilder().setBuiltin(PiiKindId.PII_KIND_ID_EMAIL))
                        .setState(PiiRuleState.PII_RULE_STATE_ENABLED)
                        .addStrategies(PiiStrategyEntry.newBuilder().setStrategy(email)))
                .setDefaultStrategy(PiiDefaultStrategy.newBuilder().setMask(PiiMaskStrategy.getDefaultInstance()))
                .build();
    }

    /** The graph of {@code TEXT} redacted by {@code policy}. */
    static ByteString redact(AnalysisServiceBlockingStub stub, RedactionPolicy policy, boolean dryRun) {
        Iterator<AnalyzeProgressiveEvent> events = stub.withDeadlineAfter(60, TimeUnit.SECONDS)
                .analyzeDocumentProgressive(AnalyzePipelineRequest.newBuilder()
                        .setDocId(Conformance.DOC)
                        .setText(Conformance.TEXT)
                        .setOffsets(OffsetMode.OFFSET_MODE_UTF8)
                        .addAllSteps(STEPS)
                        .setPiiOutput(REWRITTEN)
                        .setPiiRedaction(PiiRedactionSelection.newBuilder().setInline(policy).setDryRun(dryRun))
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
        check(graph != null, "the redact stream ended without the document");
        return graph;
    }

    /** The refusal of a redaction by {@code policy}: INVALID_ARGUMENT and its message. */
    static String refusedPolicy(AnalysisServiceBlockingStub stub, RedactionPolicy policy, String what) {
        try {
            redact(stub, policy, false);
        } catch (StatusRuntimeException e) {
            check(e.getStatus().getCode() == Status.Code.INVALID_ARGUMENT,
                    what + ": refused with " + e.getStatus().getCode() + ", not INVALID_ARGUMENT");
            return String.valueOf(e.getStatus().getDescription());
        }
        throw new AssertionError(what + ": accepted, not refused");
    }

    static int viewOf(TypedDocumentResult m, String instance) {
        for (ViewChain v : Conformance.mapping(m).getViewsList()) {
            if (v.getInstance().equals(instance)) {
                return v.getView();
            }
        }
        return -1;
    }

    static void run(AnalysisServiceBlockingStub stub) throws Exception {
        byte[] value = Conformance.utf8(Conformance.VALUE);
        int arms = 0;
        for (PiiRewriteStrategy s : everyArm()) {
            String name = s.getStrategyCase().name();
            ByteString graph = redact(stub, policy(1, s), false);
            TypedDocumentResult source = Conformance.inspect(stub, graph, 0, REWRITTEN);
            int view = viewOf(source, "redacted");
            check(view > 0, name + ": no redacted view");
            TypedDocumentResult redacted = Conformance.inspect(stub, graph, view, REWRITTEN);
            check(Conformance.count(Conformance.utf8(redacted.getRawText()), value) == 0,
                    name + ": the redacted text holds the value");
            WireChainPort port = WireChainPort.of(Conformance.mapping(redacted));
            TypedLayer edits = Conformance.layer(redacted, "turbo:pii-edits");
            check(edits.getAnnotationsCount() >= 1, name + ": no edits");
            ByteString digest = null;
            boolean named = false;
            for (TypedAnnotation a : edits.getAnnotationsList()) {
                PiiEditAnnotation e = a.getPiiEdit();
                check(e.getPolicyDigest().size() == 32, name + ": a digest of " + e.getPolicyDigest().size() + " bytes");
                check(digest == null || digest.equals(e.getPolicyDigest()), name + ": two digests in one layer");
                digest = e.getPolicyDigest();
                check(e.getPolicyRevision() == 1, name + ": revision " + e.getPolicyRevision());
                check(e.getStrategy() != PiiStrategy.PII_STRATEGY_UNSPECIFIED, name + ": an unspecified strategy");
                named |= e.getRuleId().equals("email");
                long mentions = Conformance.layer(redacted, "turbo:pii").getAnnotationsCount();
                for (int m : e.getMembersList()) {
                    check(Integer.toUnsignedLong(m) < mentions, name + ": member " + Integer.toUnsignedLong(m));
                }
                SpanFragment span = a.getSpans(0);
                long[] back = port.mapSpan(view, 0, Integer.toUnsignedLong(span.getStart()),
                        Integer.toUnsignedLong(span.getEnd()));
                long ss = Integer.toUnsignedLong(e.getSource().getStart());
                long se = Integer.toUnsignedLong(e.getSource().getEnd());
                check(back[0] == ss && back[1] == se, name + ": the edit maps back to " + back[0] + ".." + back[1]
                        + ", not its source " + ss + ".." + se);
            }
            check(named, name + ": no edit names the email rule");
            // Java re-encodes the result and the server accepts it unchanged.
            TypedDocumentResult again = TypedDocumentResult.parseFrom(redacted.toByteArray());
            Conformance.roundTrip(stub, again);
            Conformance.roundTrip(stub, source);
            arms++;
        }

        // A dry run: no redacted view, each kept candidate planned.
        RedactionPolicy dry = policy(2, everyArm().get(7));
        TypedDocumentResult planned = Conformance.inspect(stub, redact(stub, dry, true), 0, REWRITTEN);
        check(viewOf(planned, "redacted") < 0, "a dry run wrote the redacted view");
        int plans = 0;
        for (TypedAnnotation a : Conformance.layer(planned, "turbo:pii-candidates").getAnnotationsList()) {
            PiiCandidateAnnotation c = a.getPiiCandidate();
            if (c.hasPlanned()) {
                PiiPlannedRewrite p = c.getPlanned();
                check(p.getPolicyDigest().size() == 32 && p.getPolicyRevision() == 2, "a plan without its policy");
                check(p.getStrategy() != PiiStrategy.PII_STRATEGY_UNSPECIFIED, "a plan with no strategy");
                plans++;
            }
        }
        check(plans >= 2, "a dry run planned " + plans + " candidates");
        Conformance.roundTrip(stub, planned);

        // Refused, never downgraded: a keyed token with no registered key,
        // a strategy arm this build does not know (field 999 on the oneof's
        // message), and enum numbers it does not know.
        PiiRewriteStrategy keyed = arm(b -> b.setKeyedToken(PiiKeyedTokenStrategy.newBuilder()
                .setKey(PiiPolicyKeyRef.newBuilder().setId("java-key").setVersion(1))));
        String why = refusedPolicy(stub, policy(1, keyed), "an unregistered key");
        check(why.contains("not registered"), "an unregistered key refused for another reason: " + why);
        PiiRewriteStrategy unknownArm = PiiRewriteStrategy.parseFrom(Conformance.UNKNOWN_FIELD);
        check(unknownArm.getStrategyCase() == PiiRewriteStrategy.StrategyCase.STRATEGY_NOT_SET
                && unknownArm.getUnknownFields().hasField(999), "Java keeps the unknown arm as field 999");
        why = refusedPolicy(stub, policy(1, unknownArm), "an unknown strategy arm");
        check(why.contains("strategy"), "an unknown arm refused for another reason: " + why);
        PiiRewriteStrategy target = arm(b -> b.setMap(PiiMapStrategy.newBuilder()
                .setEntries(PiiMapEntries.newBuilder().putEntries("a", "b")).setTargetValue(Conformance.UNKNOWN_ENUM)));
        check(target.getMap().getTarget() == PiiMapTarget.UNRECOGNIZED, "map target 99 in Java");
        why = refusedPolicy(stub, policy(1, target), "an unknown map target");
        check(why.contains("map.target") && why.contains("99"), "an unknown map target refused for another reason: " + why);
        PiiRewriteStrategy side = arm(b -> b.setTruncate(PiiTruncateStrategy.newBuilder()
                .setSideValue(Conformance.UNKNOWN_ENUM).setCount(2)));
        why = refusedPolicy(stub, policy(1, side), "an unknown keep side");
        check(why.contains("99"), "an unknown keep side refused for another reason: " + why);
        RedactionPolicy state = policy(1, everyArm().get(0)).toBuilder()
                .setKindRules(0, policy(1, everyArm().get(0)).getKindRules(0).toBuilder()
                        .setStateValue(Conformance.UNKNOWN_ENUM))
                .build();
        why = refusedPolicy(stub, state, "an unknown rule state");
        check(why.contains("state") && why.contains("99"), "an unknown rule state refused for another reason: " + why);
        System.out.println("java redaction ok: arms=" + arms + " planned=" + plans);
    }
}
