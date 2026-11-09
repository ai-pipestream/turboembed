package ai.pipestream.turboembed.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.TurboAnalysisResult;
import ai.pipestream.turboembed.TurboAnalyzer;
import ai.pipestream.turboembed.TurboException;
import ai.pipestream.turboembed.TurboLayerKind;
import ai.pipestream.turboembed.TurboPiiCandidate;
import ai.pipestream.turboembed.TurboPiiDetector;
import ai.pipestream.turboembed.TurboPiiDisposition;
import ai.pipestream.turboembed.TurboPiiEvidence;
import ai.pipestream.turboembed.TurboPiiKind;
import ai.pipestream.turboembed.TurboPiiScore;
import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.TurboSession;
import ai.pipestream.turboembed.TurboSpanData;
import ai.pipestream.turboembed.TurboSpanView;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/**
 * The analyzer behaviour every binding shares, over the rule splitter and tokenizer, the PII
 * task, the graph, and the heads of the sealed small BERT bundle in {@code testdata}. The
 * expected labels and probabilities are the seeded heads' real output on these texts, not
 * linguistically right ones; a binding must report them as they are.
 *
 * <p>A binding runs the suite by extending this class and returning itself from
 * {@link #binding()}.
 */
public abstract class AnalyzerConformance {
    /** Native {@code TURBO_ANALYSIS_LAYER_*} codes. */
    protected static final int SENTENCES = 1;

    protected static final int TOKENS_LAYER = 2;
    protected static final int POS_LAYER = 3;
    protected static final int LEMMAS_LAYER = 4;
    protected static final int ENTITIES = 5;
    protected static final int PII = 6;
    protected static final int DEPENDENCIES = 7;
    protected static final int MORPHOLOGY = 8;

    /** Native status codes. */
    protected static final int INVALID_ARGUMENT = 256;

    protected static final int UNSUPPORTED_TASK = 514;
    protected static final int BUNDLE_NOT_FOUND = 1280;

    /** Native {@code TURBO_PII_CONFIDENCE_*} codes. */
    protected static final int PII_FORMAT = 1;

    protected static final int PII_CHECKSUM = 2;
    protected static final int PII_LABELLED_CHECKSUM = 3;
    protected static final int PII_LABELLED = 4;

    static final String TEXT =
            "Ann met Bob in Paris, and the quiet old dog slept by the river. Bob walked the old dog.";
    static final String[] TOKENS = {
        "Ann", "met", "Bob", "in", "Paris", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river", "."
    };
    static final List<String> POS = List.of(
            "PUNCT", "DET", "ADJ", "DET", "ADJ", "ADJ", "PUNCT", "PUNCT", "VERB", "PUNCT", "PUNCT", "PUNCT", "DET",
            "ADJ", "PUNCT", "PUNCT");
    static final Set<String> FEATS =
            Set.of("_", "Number=Sing", "Number=Plur", "Tense=Past|VerbForm=Fin");
    static final List<String> LEMMAS = List.of(
            "Ann", "meet", "Bob", "in", "Pari", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river",
            ".");
    static final double FIRST_POS_PROBABILITY = 0.7409762693307227;
    static final double BOB_PROBABILITY = 0.23651067881995255;
    /**
     * The BERT encoder forward pass computes activations in IEEE-754 {@code f32} before head
     * probabilities are promoted to {@code f64}; {@code glibc} and Android {@code bionic}
     * {@code libm} differ by up to ~5.2e-9 on these inputs, well below one {@code f32} ULP.
     */
    static final double HEAD_PROBABILITY_DELTA = 1.0e-7;

    /** The binding under test. */
    protected abstract TurboRuntime binding();

    private TurboAnalyzer rules() {
        return binding().openAnalyzer("en", null);
    }

    private TurboAnalyzer heads() {
        return binding().openAnalyzer("en", Testdata.headsBundle());
    }

    @Test
    public void ruleSentencesAndTokensCarrySourceRangesWithoutLabelsOrProbabilities() {
        String text = "Hello 🚀 Café 東京! Dr. John Smith works.";
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        try (TurboAnalyzer analyzer = rules()) {
            List<Row> sentences = Row.copy(analyzer.detectSentences(text));
            assertEquals(2, sentences.size());
            for (Row sentence : sentences) {
                assertEquals(SENTENCES, sentence.layerKind());
                assertTrue(sentence.layer().endsWith(":sentences"), sentence.layer());
                assertEquals("", sentence.instance());
                assertNull(sentence.label(), "covered sentence text must not masquerade as a label");
                assertNull(sentence.value());
                assertNull(sentence.probability(), "rule sentences must not invent a probability");
                assertNull(sentence.tokenRange());
                assertEquals(sentence.viewRange(), sentence.sourceRange());
                assertFalse(sentence.coveredSourceText(text).isEmpty());
            }
            assertEquals("Hello 🚀 Café 東京!", sentences.get(0).coveredSourceText(text));

            List<Row> tokens = Row.copy(analyzer.tokenize(text));
            assertFalse(tokens.isEmpty());
            List<String> surfaces = new ArrayList<>();
            for (int i = 0; i < tokens.size(); i++) {
                Row token = tokens.get(i);
                assertEquals(TOKENS_LAYER, token.layerKind());
                assertEquals(i, token.itemIndex());
                assertEquals(0, token.fragmentIndex());
                assertEquals(0, token.viewId());
                assertNull(token.label());
                assertNull(token.probability());
                assertEquals(new Row.Tokens(i, i + 1L), token.tokenRange());
                Row.Range source = token.sourceRange();
                assertNotNull(source);
                String byUtf16 = token.coveredSourceText(text);
                String byBytes = new String(
                        Arrays.copyOfRange(utf8, Math.toIntExact(source.byteStart()), Math.toIntExact(source.byteEnd())),
                        StandardCharsets.UTF_8);
                assertEquals(byUtf16, byBytes);
                assertEquals(
                        text.codePointCount(Math.toIntExact(source.utf16Start()), Math.toIntExact(source.utf16End())),
                        source.codepointEnd() - source.codepointStart());
                surfaces.add(byUtf16);
            }
            assertEquals(List.of("Hello", "🚀", "Café", "東京", "!", "Dr", ".", "John", "Smith", "works", "."), surfaces);
        }
    }

    @Test
    public void morphologyRowsComeFromTheMorphologyHead() {
        try (TurboAnalyzer analyzer = heads()) {
            List<Row> rows = Row.copy(analyzer.morphology(TOKENS));
            assertEquals(TOKENS.length, rows.size());
            for (int i = 0; i < rows.size(); i++) {
                Row row = rows.get(i);
                assertEquals(MORPHOLOGY, row.layerKind());
                assertNull(row.viewRange(), "pre-tokenized morphology rows have no text view range");
                assertNull(row.sourceRange(), "pre-tokenized morphology rows have no source range");
                assertEquals(new Row.Tokens(i, i + 1L), row.tokenRange());
                assertTrue(FEATS.contains(row.label()), row.label());
                assertNull(row.probability(), "morphology rows carry no probability");
            }
        }
    }

    @Test
    public void posRowsComeFromThePartOfSpeechHeadWithProbabilities() {
        try (TurboAnalyzer analyzer = heads()) {
            List<Row> rows = Row.copy(analyzer.posTag(TOKENS));
            assertEquals(POS, labels(rows));
            for (int i = 0; i < rows.size(); i++) {
                Row row = rows.get(i);
                assertEquals(POS_LAYER, row.layerKind());
                assertNull(row.viewRange(), "pre-tokenized POS rows have no text view range");
                assertNull(row.sourceRange(), "pre-tokenized POS rows have no source range");
                assertEquals(new Row.Tokens(i, i + 1L), row.tokenRange());
                assertNotNull(row.probability());
                assertTrue(row.probability() > 0.0 && row.probability() <= 1.0);
            }
            assertEquals(FIRST_POS_PROBABILITY, rows.get(0).probability(), HEAD_PROBABILITY_DELTA);
        }
    }

    @Test
    public void lemmaRowsComeFromTheLemmatizerHeadReadingTheGivenTags() {
        try (TurboAnalyzer analyzer = heads()) {
            List<Row> rows = Row.copy(analyzer.lemmatize(TOKENS, POS.toArray(new String[0])));
            assertEquals(LEMMAS, labels(rows));
            assertTrue(rows.stream().allMatch(row -> row.probability() == null));
            RuntimeException failure = assertThrows(
                    RuntimeException.class, () -> Row.copy(analyzer.lemmatize(new String[] {"Ann"}, new String[] {"XYZ"})));
            assertStatus(failure, INVALID_ARGUMENT, "TURBO_E_INVALID_ARGUMENT");
            assertTrue(failure.getMessage().contains("not a label of the tag head"), failure.getMessage());
        }
    }

    @Test
    public void entityRowsAreFreshPerCallAndFoundInTextWithSourceRanges() {
        try (TurboAnalyzer analyzer = heads()) {
            List<Row> first = Row.copy(analyzer.findTokens(TOKENS));
            assertEquals(1, first.size());
            Row bob = first.get(0);
            assertEquals(ENTITIES, bob.layerKind());
            assertEquals(new Row.Tokens(2, 3), bob.tokenRange());
            assertEquals("PER", bob.label());
            assertEquals(BOB_PROBABILITY, bob.probability(), HEAD_PROBABILITY_DELTA);
            assertNull(bob.viewRange());
            assertNull(bob.sourceRange());

            Row.copy(analyzer.findTokens(new String[] {"Bo", "gave", "Ann", "a", "map"}));
            assertEquals(first, Row.copy(analyzer.findTokens(TOKENS)), "an intervening call must not change a later one");

            String sentence = TEXT.substring(0, TEXT.indexOf(". ") + 1);
            List<Row> inText = Row.copy(analyzer.findText(sentence));
            assertEquals(1, inText.size());
            assertEquals("Bob", inText.get(0).coveredSourceText(sentence));
            assertEquals(new Row.Tokens(2, 3), inText.get(0).tokenRange());
            assertEquals("PER", inText.get(0).label());
        }
    }

    @Test
    public void theGraphCarriesEveryHeadLayerFromOneEncoderPass() {
        try (TurboAnalyzer analyzer = heads()) {
            Graph graph = Graph.copy(analyzer.analyzeGraph(TEXT));
            assertTrue(graph.bytes().length > 4);
            assertEquals("TGRF", new String(graph.bytes(), 0, 4, StandardCharsets.US_ASCII));
            TreeSet<Integer> kinds = new TreeSet<>();
            for (Row row : graph.rows()) {
                kinds.add(row.layerKind());
            }
            assertEquals(
                    new TreeSet<>(List.of(
                            SENTENCES, TOKENS_LAYER, POS_LAYER, LEMMAS_LAYER, ENTITIES, DEPENDENCIES, MORPHOLOGY)),
                    kinds,
                    "the text holds no PII");

            List<Row> tokens = ofKind(graph.rows(), TOKENS_LAYER);
            List<Row> pos = ofKind(graph.rows(), POS_LAYER);
            List<Row> morph = ofKind(graph.rows(), MORPHOLOGY);
            List<Row> lemmas = ofKind(graph.rows(), LEMMAS_LAYER);
            List<Row> deps = ofKind(graph.rows(), DEPENDENCIES);
            assertEquals(22, tokens.size());
            assertEquals(tokens.size(), pos.size());
            assertEquals(tokens.size(), morph.size());
            assertEquals(tokens.size(), lemmas.size());
            assertEquals(tokens.size(), deps.size());
            assertEquals(POS, labels(pos.subList(0, TOKENS.length)));
            assertEquals(labels(Row.copy(analyzer.morphology(TOKENS))), labels(morph.subList(0, TOKENS.length)));
            assertEquals(LEMMAS, labels(lemmas.subList(0, TOKENS.length)));
            for (int i = 0; i < tokens.size(); i++) {
                assertEquals(tokens.get(i).sourceRange(), pos.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), morph.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), lemmas.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), deps.get(i).sourceRange());
                assertEquals(new Row.Tokens(i, i + 1L), morph.get(i).tokenRange());
                assertTrue(FEATS.contains(morph.get(i).label()));
                assertNull(morph.get(i).probability());
                assertEquals(new Row.Tokens(i, i + 1L), deps.get(i).tokenRange());
            }

            // A dependency row: the relation is the label, the head token index plus one is the
            // classification, zero under the root of its sentence.
            long roots = 0;
            for (Row dep : deps) {
                assertNotNull(dep.label());
                assertNull(dep.probability());
                if (dep.classification() == 0L) {
                    roots++;
                    assertEquals("root", dep.label());
                } else {
                    long head = dep.classification() - 1L;
                    assertTrue(head < tokens.size());
                    assertTrue(head != dep.tokenRange().start(), "a token is not its own head");
                }
            }
            assertEquals(2, roots, "one root a sentence");
            assertEquals(1L, deps.get(4).classification(), "the head of Paris is Ann");
            assertEquals(0L, deps.get(1).classification(), "met is the first sentence's root");

            List<Row> entities = ofKind(graph.rows(), ENTITIES);
            assertEquals(
                    List.of("Bob", "dog"),
                    entities.stream().map(row -> row.coveredSourceText(TEXT)).collect(Collectors.toList()));
            assertEquals(new Row.Tokens(20, 21), entities.get(1).tokenRange());

            assertEquals(graph.rows(), Row.copy(analyzer.inspectGraph(graph.bytes())));
        }
    }

    @Test
    public void aBundleWithoutHeadsLeavesTheLearnedTasksUnsupportedByName() {
        try (TurboAnalyzer analyzer = binding().openAnalyzer("en", Testdata.tinyBertBundle())) {
            RuntimeException pos = assertThrows(RuntimeException.class, () -> Row.copy(analyzer.posTag(new String[] {"The"})));
            assertStatus(pos, UNSUPPORTED_TASK, "TURBO_E_UNSUPPORTED_TASK");
            assertTrue(pos.getMessage().contains("no head writing pos"), pos.getMessage());
            RuntimeException morph =
                    assertThrows(RuntimeException.class, () -> Row.copy(analyzer.morphology(new String[] {"The"})));
            assertStatus(morph, UNSUPPORTED_TASK, "TURBO_E_UNSUPPORTED_TASK");
            assertTrue(morph.getMessage().contains("no head writing morphology"), morph.getMessage());
            Graph graph = Graph.copy(analyzer.analyzeGraph("The river."));
            assertFalse(ofKind(graph.rows(), TOKENS_LAYER).isEmpty());
            assertTrue(ofKind(graph.rows(), POS_LAYER).isEmpty());
            assertTrue(ofKind(graph.rows(), MORPHOLOGY).isEmpty());
            assertTrue(ofKind(graph.rows(), DEPENDENCIES).isEmpty());
        }
    }

    @Test
    public void aMissingBundleIsRefusedNamingTheDescriptorField() {
        RuntimeException failure = assertThrows(
                RuntimeException.class, () -> binding().openAnalyzer("en", Testdata.missingBundle()).close());
        assertStatus(failure, BUNDLE_NOT_FOUND, "TURBO_E_BUNDLE_NOT_FOUND");
        assertEquals(7, org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).field(), "the field number of bundle_path");
        assertTrue(failure.getMessage().contains("does-not-exist"), failure.getMessage());
    }

    @Test
    public void unconfiguredLearnedTasksAreUnsupportedInsteadOfFallingBack() {
        try (TurboAnalyzer analyzer = rules()) {
            assertStatus(
                    assertThrows(RuntimeException.class, () -> Row.copy(analyzer.posTag(new String[] {"The", "rivers", "went"}))),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class, () -> Row.copy(analyzer.morphology(new String[] {"The", "rivers", "went"}))),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> Row.copy(analyzer.lemmatize(new String[] {"went"}, new String[] {"VERB"}))),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> Row.copy(analyzer.findTokens(new String[] {"John", "Smith", "visited", "New", "York"}))),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> Row.copy(analyzer.findText("John Smith visited New York and emailed alice@example.com."))),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
        }
    }

    @Test
    public void emptyInputsProduceEmptyResultsWithoutABundle() {
        try (TurboAnalyzer analyzer = rules()) {
            assertEquals(0, Row.copy(analyzer.detectSentences("")).size());
            assertEquals(0, Row.copy(analyzer.tokenize("")).size());
            assertEquals(0, Row.copy(analyzer.posTag(new String[0])).size());
            assertEquals(0, Row.copy(analyzer.morphology(new String[0])).size());
            assertEquals(0, Row.copy(analyzer.lemmatize(new String[0], new String[0])).size());
            assertEquals(0, Row.copy(analyzer.findTokens(new String[0])).size());
            assertEquals(0, Row.copy(analyzer.detectPii("")).size());
        }
    }

    @Test
    public void piiRowsWithholdTheirValuesByDefault() {
        String text = "Email alice@example.com from 192.168.1.42. SIN: 046 454 286.";
        try (TurboAnalyzer analyzer = rules()) {
            List<Row> rows = Row.copy(analyzer.detectPii(text));
            assertTrue(rows.stream().anyMatch(row -> "email".equals(row.label())), "the mention is found");
            assertTrue(rows.stream().allMatch(row -> row.value() == null), "a PII row carries its value (#1904)");
            assertTrue(Row.copy(analyzer.analyzeGraph(text)).stream()
                    .filter(row -> row.layerKind() == PII)
                    .allMatch(row -> row.value() == null), "an analyzed PII row carries its value");
        }
    }

    @Test
    public void piiIsASeparateTypedLayerWithANormalizedValueWhenAskedFor() {
        String text = "Email alice@example.com from 192.168.1.42. SIN: 046 454 286.";
        try (TurboAnalyzer analyzer = binding().createAnalyzer(new TurboAnalyzer.Config("en", null).withPiiValues(true))) {
            List<Row> rows = Row.copy(analyzer.detectPii(text));
            assertFalse(rows.isEmpty());
            assertTrue(rows.stream().allMatch(row -> row.layerKind() == PII));
            Row email = rows.stream()
                    .filter(row -> "email".equals(row.label()))
                    .findFirst()
                    .orElseThrow(AssertionError::new);
            assertEquals("alice@example.com", email.coveredSourceText(text));
            assertEquals("alice@example.com", email.value());
            assertEquals(PII_FORMAT, email.classification());
            assertNull(email.probability(), "PII evidence classification is not a probability");
            Row sin = rows.stream()
                    .filter(row -> "ca-sin".equals(row.label()))
                    .findFirst()
                    .orElseThrow(AssertionError::new);
            assertEquals("046 454 286", sin.coveredSourceText(text));
            assertEquals("046454286", sin.value());
            assertEquals(PII_LABELLED_CHECKSUM, sin.classification());
        }
    }

    @Test
    public void piiRowsCarryTheirNumericKindWithNoLabelParse() {
        String text = "Email alice@example.com from 192.168.1.42. SIN: 046 454 286.";
        try (TurboAnalyzer analyzer = rules()) {
            byte[] graph;
            try (TurboAnalysisResult detected = analyzer.detectPii(text);
                    TurboAnalysisResult analyzed = analyzer.analyzeGraph(text)) {
                assertPiiKinds(detected);
                assertPiiKinds(analyzed);
                graph = analyzed.copyGraphBytes();
            }
            try (TurboAnalysisResult inspected = analyzer.inspectGraph(graph)) {
                assertPiiKinds(inspected);
            }
        }
    }

    /** Every PII row's numeric kind is the registry id of its label; no other row has one. */
    private static void assertPiiKinds(TurboAnalysisResult result) {
        Set<TurboPiiKind> seen = new TreeSet<>();
        for (TurboSpanView row : result.spans()) {
            if (row.layerKind() != TurboLayerKind.PII) {
                assertTrue(row.piiKindCode().isEmpty(), "a " + row.layerKind() + " row has a PII kind");
                continue;
            }
            TurboPiiKind kind = TurboPiiKind.fromName(row.label().orElseThrow()).orElseThrow();
            assertEquals(kind.nativeCode(), row.piiKindCode().orElseThrow(), row.label().orElseThrow());
            assertEquals(kind, row.piiKind().orElseThrow());
            assertTrue(row.piiCustomKindIndex().isEmpty());
            assertTrue(row.tokenRange().isEmpty(), "a PII row has no token range");
            TurboSpanData copy = row.materialize();
            assertEquals(kind, copy.piiKind().orElseThrow());
            assertEquals(row.classification(), copy.classification());
            seen.add(kind);
        }
        assertEquals(Set.of(TurboPiiKind.EMAIL, TurboPiiKind.IPV4, TurboPiiKind.CA_SIN), seen);
    }

    /**
     * Same TGRF fixture as {@code core/tests/analysis_pii_evidence.rs}: a Luhn-passed card with a
     * label fact, a custom extractor mention with detector identity and score, and a
     * {@code turbo:pii-candidates} layer with kept and label-rejected candidates.
     */
    private static final String PII_EVIDENCE_GRAPH_HEX =
            "5447524602004300000050414e20343533322d303135312d313238332d303336362c20656d706c6f79656520452d34322c20616e6420756e"
                    + "6c6162656c6c65642030343620343534203238362e4300000050414e20343533322d303135312d313238332d303336362c20656d706c6f79"
                    + "656520452d34322c20616e6420756e6c6162656c6c65642030343620343534203238362e000100000000000000000c000000747572626f3a"
                    + "736f75726365010000003000000000000000000000000000000000000000014300000050414e20343533322d303135312d313238332d3033"
                    + "36362c20656d706c6f79656520452d34322c20616e6420756e6c6162656c6c65642030343620343534203238362e00000000020000000900"
                    + "0000747572626f3a706969000000000100020000000009000000747572626f3a706969010000003000000000000000000000000000000000"
                    + "0000010b0100000000000002000000010000000400000017000000010000002200000026000000010000000b000000656d706c6f7965652d"
                    + "69640400000010000000747572626f2d7069693a637572736f7205000000302e312e3015000000747572626f2d7069693a63612d6964656e"
                    + "7469747905000000302e312e300b00000061636d653a656d702d696405000000322e312e300b00000061636d653a73636f72657203000000"
                    + "312e300200000008000000000001000200000000000200030000000201010300000000030000000300000005070000000200000002000000"
                    + "030000000200000000010000703f0103000102000000100000003435333230313531313238333033363604000000452d3432140000007475"
                    + "72626f3a7069692d63616e64696461746573000000000100030000000009000000747572626f3a7069690100000030000000000000000000"
                    + "00000000000100000009000000747572626f3a70696900000000011501000000000000030000000100000004000000170000000100000022"
                    + "00000026000000010000003700000042000000010000000b000000656d706c6f7965652d69640400000010000000747572626f2d7069693a"
                    + "637572736f7205000000302e312e3015000000747572626f2d7069693a63612d6964656e7469747905000000302e312e300b00000061636d"
                    + "653a656d702d696405000000322e312e300b00000061636d653a73636f72657203000000312e300300000008000000000001001c00000003"
                    + "0000000000020001000400000002010103000000000300000003000000050700000002010103000000020000000300000004000000030000"
                    + "0000010000703f010300010003000000010000000001010000000401";

    /**
     * Same TGRF fixture as {@code build_bound_parent_and_max_u32_graph} in
     * {@code core/tests/analysis_pii_evidence.rs}: a 2-mention {@code turbo:pii} layer
     * {@code "actual"} with {@code Rule(0xFFFF_FFFFL)} and {@code DenyHit(0xFFFF_FFFFL, 1L)}, an
     * unrelated 1-mention {@code turbo:pii} layer {@code "audit"}, and a
     * {@code turbo:pii-candidates} layer {@code "audit"} bound via provenance to {@code "actual"}
     * referring to mention 1.
     */
    private static final String PII_BOUND_PARENT_AND_MAX_U32_GRAPH_HEX =
            "5447524602000d0000006140622e636f206340642e636f0d0000006140622e636f206340642e636f000100000000000000000c0000007475"
                    + "72626f3a736f75726365010000003000000000000000000000000000000000000000010d0000006140622e636f206340642e636f00000000"
                    + "0300000009000000747572626f3a7069690600000061637475616c0100020000000008000000746573743a70696901000000300000000000"
                    + "00000000000000000000000000018b000000000000000200000001000000000000000600000001000000070000000d000000000000000100"
                    + "00000a000000637573746f6d3a646574010000003102000000050000000500000002000000000000000200000005ffffffff04ffffffff01"
                    + "00000002000000010000000200000002000000000002000000060000006140622e636f060000006340642e636f09000000747572626f3a70"
                    + "69690500000061756469740100010000000008000000746573743a7069690100000030000000000000000000000000000000000000015f00"
                    + "0000000000000100000001000000000000000600000000000000010000000c000000637573746f6d3a6f7468657201000000310100000005"
                    + "00000001000000000001000000010100000001000000010000000001000000060000006140622e636f14000000747572626f3a7069692d63"
                    + "616e646964617465730500000061756469740100010000000008000000746573743a70696901000000300000000000000000000000000000"
                    + "0100000009000000747572626f3a7069690600000061637475616c015c000000000000000100000001000000070000000d00000000000000"
                    + "010000000a000000637573746f6d3a646574010000003101000000050000000100000000000100000005ffffffff01000000010000000100"
                    + "000000010000000101000000";

    @Test
    public void piiRowsCarryEvidenceFactsDetectorScoreAndDetailGatedCandidates() {
        byte[] fixture = HexFormat.of().parseHex(PII_EVIDENCE_GRAPH_HEX);
        List<TurboPiiEvidence> expectedCardEvidence = List.of(
                new TurboPiiEvidence.Checksum(TurboPiiEvidence.ChecksumAlgorithm.LUHN, true),
                new TurboPiiEvidence.Label(0, 3, 3));
        List<TurboPiiEvidence> expectedEmpEvidence = List.of(new TurboPiiEvidence.Rule(7));
        TurboPiiDetector expectedEmpDetector = new TurboPiiDetector("acme:emp-id", "2.1.0");
        TurboPiiScore expectedEmpScore = new TurboPiiScore(
                0.9375f, TurboPiiScore.Kind.PROBABILITY, new TurboPiiDetector("acme:scorer", "1.0"), true);

        try (TurboAnalyzer defaultAnalyzer = rules();
                TurboAnalysisResult inspected = defaultAnalyzer.inspectGraph(fixture)) {
            assertTrue(inspected.piiCandidates().isEmpty(), "candidates are withheld without withPiiDetail(true)");
            List<? extends TurboSpanView> spans = inspected.spans();
            assertEquals(2, spans.size());

            TurboSpanView card = spans.get(0);
            assertEquals(Optional.of(TurboPiiKind.CARD), card.piiKind());
            assertEquals(PII_LABELLED_CHECKSUM, card.classification());
            assertEquals(Optional.of(new TurboPiiDetector("turbo-pii:cursor", "0.1.0")), card.piiDetector());
            assertEquals(expectedCardEvidence, card.piiEvidence());
            assertEquals(Optional.empty(), card.piiScore());
            assertEquals(Optional.empty(), card.value());

            TurboSpanView emp = spans.get(1);
            assertEquals(Optional.empty(), emp.piiKind());
            assertEquals(OptionalInt.of(0), emp.piiCustomKindIndex());
            assertEquals(Optional.of("employee-id"), emp.label());
            assertEquals(Optional.of(expectedEmpDetector), emp.piiDetector());
            assertEquals(expectedEmpEvidence, emp.piiEvidence());
            assertEquals(Optional.of(expectedEmpScore), emp.piiScore());
            assertEquals(Optional.empty(), emp.value());

            for (TurboSpanData copy : inspected.copySpans()) {
                String rendered = copy.toString();
                for (String forbidden : List.of("4532", "E-42", "046 454 286", "046454286", "PAN")) {
                    assertFalse(rendered.contains(forbidden), "default span leaked " + forbidden + ": " + rendered);
                }
            }
            assertEquals(expectedCardEvidence, inspected.copySpan(0).piiEvidence());
            assertEquals(Optional.of(expectedEmpDetector), inspected.copySpan(1).piiDetector());
            assertEquals(Optional.of(expectedEmpScore), inspected.copySpan(1).piiScore());
        }

        TurboAnalyzer.Config detailConfig = new TurboAnalyzer.Config("en", null).withPiiDetail(true);
        try (TurboAnalyzer detailAnalyzer = binding().createAnalyzer(detailConfig);
                TurboAnalysisResult detailed = detailAnalyzer.inspectGraph(fixture)) {
            List<TurboPiiCandidate> cands = detailed.piiCandidates();
            assertEquals(3, cands.size());

            TurboPiiCandidate c0 = cands.get(0);
            assertEquals(Optional.of(TurboPiiKind.CARD), c0.piiKind());
            assertEquals(new TurboPiiDisposition.Kept(0), c0.disposition());
            assertEquals(expectedCardEvidence, c0.evidence());

            TurboPiiCandidate c1 = cands.get(1);
            assertEquals(OptionalInt.of(0), c1.piiCustomKindIndex());
            assertEquals("employee-id", c1.label());
            assertEquals(expectedEmpDetector, c1.detector());
            assertEquals(expectedEmpEvidence, c1.evidence());
            assertEquals(Optional.of(expectedEmpScore), c1.score());
            assertEquals(new TurboPiiDisposition.Kept(1), c1.disposition());

            TurboPiiCandidate c2 = cands.get(2);
            assertEquals(Optional.of(TurboPiiKind.CA_SIN), c2.piiKind());
            assertEquals(
                    new TurboPiiDisposition.RejectedByRule(TurboPiiDisposition.RejectReason.LABEL_REQUIRED),
                    c2.disposition());

            String directText = "IMEI: 490154203237518 and unlabelled 046 454 286.";
            try (TurboAnalysisResult detected = detailAnalyzer.detectPii(directText);
                    TurboAnalysisResult analyzed = detailAnalyzer.analyzeGraph(directText)) {
                assertTrue(detected.piiCandidates().stream().anyMatch(c -> c.disposition()
                        .equals(new TurboPiiDisposition.RejectedByRule(
                                TurboPiiDisposition.RejectReason.LABEL_REQUIRED))));
                assertFalse(analyzed.piiCandidates().isEmpty());
            }

            byte[] boundFixture = HexFormat.of().parseHex(PII_BOUND_PARENT_AND_MAX_U32_GRAPH_HEX);
            try (TurboAnalysisResult bound = detailAnalyzer.inspectGraph(boundFixture)) {
                List<? extends TurboSpanView> actualSpans = bound.spans(TurboLayerKind.PII, "actual");
                assertEquals(2, actualSpans.size());
                assertEquals(List.of(new TurboPiiEvidence.Rule(0xFFFF_FFFFL)), actualSpans.get(0).piiEvidence());
                assertEquals(
                        List.of(new TurboPiiEvidence.DenyHit(0xFFFF_FFFFL, 1L)), actualSpans.get(1).piiEvidence());
                assertEquals(List.of(new TurboPiiEvidence.Rule(0xFFFF_FFFFL)), bound.copySpan(0).piiEvidence());
                assertEquals(
                        List.of(new TurboPiiEvidence.DenyHit(0xFFFF_FFFFL, 1L)), bound.copySpan(1).piiEvidence());

                List<TurboPiiCandidate> boundCands = bound.piiCandidates();
                assertEquals(1, boundCands.size());
                TurboPiiCandidate boundCand = boundCands.get(0);
                assertEquals("audit", boundCand.instance());
                assertEquals("actual", boundCand.piiInstance());
                assertEquals(new TurboPiiDisposition.Kept(1), boundCand.disposition());
                assertEquals(List.of(new TurboPiiEvidence.Rule(0xFFFF_FFFFL)), boundCand.evidence());
            }
        }
    }

    @Test
    public void graphRoundTripUsesStoredSourceAndOnlyConfiguredLayers() {
        String text = "John Smith flew to São Paulo 🚀. Email: test@example.com.";
        Graph exported;
        try (TurboAnalyzer analyzer = rules()) {
            exported = Graph.copy(analyzer.analyzeGraph(text));
        }
        assertTrue(exported.bytes().length > 32);
        assertEquals("TGRF", new String(exported.bytes(), 0, 4, StandardCharsets.US_ASCII));
        TreeSet<Integer> kinds = new TreeSet<>();
        for (Row row : exported.rows()) {
            kinds.add(row.layerKind());
            assertFalse(row.coveredSourceText(text).isEmpty());
        }
        assertEquals(
                new TreeSet<>(List.of(SENTENCES, TOKENS_LAYER, PII)),
                kinds,
                "unconfigured POS, lemma, and NER layers must be absent, not fabricated");

        try (TurboAnalyzer other = rules()) {
            assertEquals(exported.rows(), Row.copy(other.inspectGraph(exported.bytes())));
            RuntimeException failure =
                    assertThrows(RuntimeException.class, () -> Row.copy(other.inspectGraph(new byte[] {1, 2, 3, 4, 5})));
            assertStatus(failure, INVALID_ARGUMENT, "TURBO_E_INVALID_ARGUMENT");
        }
    }

    @Test
    public void textCrossesAsTheSameUtf8WhateverTheBridge() {
        // Characters outside the Basic Multilingual Plane, combining marks and a NUL: the rows
        // are addressed in bytes, UTF-16 units and scalars, and every binding must agree.
        String text = "A\u0000b 🚀🚀 é 東京 z.";
        try (TurboAnalyzer analyzer = rules()) {
            List<Row> tokens = Row.copy(analyzer.tokenize(text));
            assertFalse(tokens.isEmpty());
            Row last = tokens.get(tokens.size() - 1);
            Row.Range source = last.sourceRange();
            assertEquals(text.getBytes(StandardCharsets.UTF_8).length, source.byteEnd());
            assertEquals(text.length(), source.utf16End());
            assertEquals(text.codePointCount(0, text.length()), source.codepointEnd());
            assertArrayEquals(
                    tokens.toArray(), Row.copy(analyzer.tokenize(text)).toArray(), "the same text gives the same rows again");
        }
    }

    /** Asserts a failure is the binding's native exception with {@code code} and {@code name}. */
    protected void assertStatus(Throwable failure, int code, String name) {
        assertEquals(code, org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).code(), binding().kind() + ": " + failure);
        assertEquals(name, org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).statusName(), binding().kind() + ": " + failure);
    }

    static List<String> labels(List<Row> rows) {
        List<String> out = new ArrayList<>();
        for (Row row : rows) {
            out.add(row.label());
        }
        return out;
    }

    static List<Row> ofKind(List<Row> rows, int kind) {
        List<Row> out = new ArrayList<>();
        for (Row row : rows) {
            if (row.layerKind() == kind) {
                out.add(row);
            }
        }
        return out;
    }
}
