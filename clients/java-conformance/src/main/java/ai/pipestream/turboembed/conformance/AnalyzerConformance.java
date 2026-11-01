package ai.pipestream.turboembed.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;
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

    /** Native status codes. */
    protected static final int INVALID_ARGUMENT = 256;

    protected static final int UNSUPPORTED_TASK = 514;
    protected static final int BUNDLE_NOT_FOUND = 1280;

    /** Native {@code TURBO_PII_CONFIDENCE_FORMAT}. */
    protected static final int PII_FORMAT = 1;

    static final String TEXT =
            "Ann met Bob in Paris, and the quiet old dog slept by the river. Bob walked the old dog.";
    static final String[] TOKENS = {
        "Ann", "met", "Bob", "in", "Paris", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river", "."
    };
    static final List<String> POS = List.of(
            "PUNCT", "DET", "ADJ", "DET", "ADJ", "ADJ", "PUNCT", "PUNCT", "VERB", "PUNCT", "PUNCT", "PUNCT", "DET",
            "ADJ", "PUNCT", "PUNCT");
    static final List<String> LEMMAS = List.of(
            "Ann", "meet", "Bob", "in", "Pari", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river",
            ".");
    static final double FIRST_POS_PROBABILITY = 0.7409762693307227;
    static final double BOB_PROBABILITY = 0.23651068468515454;

    /** The binding under test. */
    protected abstract Binding binding();

    private Analyzer rules() {
        return binding().openAnalyzer("en", null);
    }

    private Analyzer heads() {
        return binding().openAnalyzer("en", Testdata.headsBundle());
    }

    @Test
    public void ruleSentencesAndTokensCarrySourceRangesWithoutLabelsOrProbabilities() {
        String text = "Hello 🚀 Café 東京! Dr. John Smith works.";
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        try (Analyzer analyzer = rules()) {
            List<Row> sentences = analyzer.detectSentences(text);
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

            List<Row> tokens = analyzer.tokenize(text);
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
    public void posRowsComeFromThePartOfSpeechHeadWithProbabilities() {
        try (Analyzer analyzer = heads()) {
            List<Row> rows = analyzer.posTag(TOKENS);
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
            assertEquals(FIRST_POS_PROBABILITY, rows.get(0).probability(), 1.0e-6);
        }
    }

    @Test
    public void lemmaRowsComeFromTheLemmatizerHeadReadingTheGivenTags() {
        try (Analyzer analyzer = heads()) {
            List<Row> rows = analyzer.lemmatize(TOKENS, POS.toArray(new String[0]));
            assertEquals(LEMMAS, labels(rows));
            assertTrue(rows.stream().allMatch(row -> row.probability() == null));
            RuntimeException failure = assertThrows(
                    RuntimeException.class, () -> analyzer.lemmatize(new String[] {"Ann"}, new String[] {"XYZ"}));
            assertStatus(failure, INVALID_ARGUMENT, "TURBO_E_INVALID_ARGUMENT");
            assertTrue(failure.getMessage().contains("not a label of the tag head"), failure.getMessage());
        }
    }

    @Test
    public void entityRowsAreFreshPerCallAndFoundInTextWithSourceRanges() {
        try (Analyzer analyzer = heads()) {
            List<Row> first = analyzer.findTokens(TOKENS);
            assertEquals(1, first.size());
            Row bob = first.get(0);
            assertEquals(ENTITIES, bob.layerKind());
            assertEquals(new Row.Tokens(2, 3), bob.tokenRange());
            assertEquals("PER", bob.label());
            assertEquals(BOB_PROBABILITY, bob.probability(), 1.0e-6);
            assertNull(bob.viewRange());
            assertNull(bob.sourceRange());

            analyzer.findTokens(new String[] {"Bo", "gave", "Ann", "a", "map"});
            assertEquals(first, analyzer.findTokens(TOKENS), "an intervening call must not change a later one");

            String sentence = TEXT.substring(0, TEXT.indexOf(". ") + 1);
            List<Row> inText = analyzer.findText(sentence);
            assertEquals(1, inText.size());
            assertEquals("Bob", inText.get(0).coveredSourceText(sentence));
            assertEquals(new Row.Tokens(2, 3), inText.get(0).tokenRange());
            assertEquals("PER", inText.get(0).label());
        }
    }

    @Test
    public void theGraphCarriesEveryHeadLayerFromOneEncoderPass() {
        try (Analyzer analyzer = heads()) {
            Graph graph = analyzer.analyzeGraph(TEXT);
            assertTrue(graph.bytes().length > 4);
            assertEquals("TGRF", new String(graph.bytes(), 0, 4, StandardCharsets.US_ASCII));
            TreeSet<Integer> kinds = new TreeSet<>();
            for (Row row : graph.rows()) {
                kinds.add(row.layerKind());
            }
            assertEquals(
                    new TreeSet<>(List.of(SENTENCES, TOKENS_LAYER, POS_LAYER, LEMMAS_LAYER, ENTITIES, DEPENDENCIES)),
                    kinds,
                    "the text holds no PII");

            List<Row> tokens = ofKind(graph.rows(), TOKENS_LAYER);
            List<Row> pos = ofKind(graph.rows(), POS_LAYER);
            List<Row> lemmas = ofKind(graph.rows(), LEMMAS_LAYER);
            List<Row> deps = ofKind(graph.rows(), DEPENDENCIES);
            assertEquals(22, tokens.size());
            assertEquals(tokens.size(), pos.size());
            assertEquals(tokens.size(), lemmas.size());
            assertEquals(tokens.size(), deps.size());
            assertEquals(POS, labels(pos.subList(0, TOKENS.length)));
            assertEquals(LEMMAS, labels(lemmas.subList(0, TOKENS.length)));
            for (int i = 0; i < tokens.size(); i++) {
                assertEquals(tokens.get(i).sourceRange(), pos.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), lemmas.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), deps.get(i).sourceRange());
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
            assertEquals(List.of("Bob", "dog"), entities.stream().map(row -> row.coveredSourceText(TEXT)).toList());
            assertEquals(new Row.Tokens(20, 21), entities.get(1).tokenRange());

            assertEquals(graph.rows(), analyzer.inspectGraph(graph.bytes()));
        }
    }

    @Test
    public void aBundleWithoutHeadsLeavesTheLearnedTasksUnsupportedByName() {
        try (Analyzer analyzer = binding().openAnalyzer("en", Testdata.tinyBertBundle())) {
            RuntimeException pos = assertThrows(RuntimeException.class, () -> analyzer.posTag(new String[] {"The"}));
            assertStatus(pos, UNSUPPORTED_TASK, "TURBO_E_UNSUPPORTED_TASK");
            assertTrue(pos.getMessage().contains("no head writing pos"), pos.getMessage());
            Graph graph = analyzer.analyzeGraph("The river.");
            assertFalse(ofKind(graph.rows(), TOKENS_LAYER).isEmpty());
            assertTrue(ofKind(graph.rows(), POS_LAYER).isEmpty());
            assertTrue(ofKind(graph.rows(), DEPENDENCIES).isEmpty());
        }
    }

    @Test
    public void aMissingBundleIsRefusedNamingTheDescriptorField() {
        RuntimeException failure = assertThrows(
                RuntimeException.class, () -> binding().openAnalyzer("en", Testdata.missingBundle()).close());
        assertStatus(failure, BUNDLE_NOT_FOUND, "TURBO_E_BUNDLE_NOT_FOUND");
        assertEquals(7, binding().statusField(failure), "the field number of bundle_path");
        assertTrue(failure.getMessage().contains("does-not-exist"), failure.getMessage());
    }

    @Test
    public void unconfiguredLearnedTasksAreUnsupportedInsteadOfFallingBack() {
        try (Analyzer analyzer = rules()) {
            assertStatus(
                    assertThrows(RuntimeException.class, () -> analyzer.posTag(new String[] {"The", "rivers", "went"})),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> analyzer.lemmatize(new String[] {"went"}, new String[] {"VERB"})),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> analyzer.findTokens(new String[] {"John", "Smith", "visited", "New", "York"})),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
            assertStatus(
                    assertThrows(
                            RuntimeException.class,
                            () -> analyzer.findText("John Smith visited New York and emailed alice@example.com.")),
                    UNSUPPORTED_TASK,
                    "TURBO_E_UNSUPPORTED_TASK");
        }
    }

    @Test
    public void emptyInputsProduceEmptyResultsWithoutABundle() {
        try (Analyzer analyzer = rules()) {
            assertEquals(0, analyzer.detectSentences("").size());
            assertEquals(0, analyzer.tokenize("").size());
            assertEquals(0, analyzer.posTag(new String[0]).size());
            assertEquals(0, analyzer.lemmatize(new String[0], new String[0]).size());
            assertEquals(0, analyzer.findTokens(new String[0]).size());
            assertEquals(0, analyzer.detectPii("").size());
        }
    }

    @Test
    public void piiIsASeparateTypedLayerWithANormalizedValue() {
        String text = "Email alice@example.com from 192.168.1.42.";
        try (Analyzer analyzer = rules()) {
            List<Row> rows = analyzer.detectPii(text);
            assertFalse(rows.isEmpty());
            assertTrue(rows.stream().allMatch(row -> row.layerKind() == PII));
            Row email = rows.stream().filter(row -> "email".equals(row.label())).findFirst().orElseThrow();
            assertEquals("alice@example.com", email.coveredSourceText(text));
            assertEquals("alice@example.com", email.value());
            assertEquals(PII_FORMAT, email.classification());
            assertNull(email.probability(), "PII evidence classification is not a probability");
        }
    }

    @Test
    public void graphRoundTripUsesStoredSourceAndOnlyConfiguredLayers() {
        String text = "John Smith flew to São Paulo 🚀. Email: test@example.com.";
        Graph exported;
        try (Analyzer analyzer = rules()) {
            exported = analyzer.analyzeGraph(text);
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

        try (Analyzer other = rules()) {
            assertEquals(exported.rows(), other.inspectGraph(exported.bytes()));
            RuntimeException failure =
                    assertThrows(RuntimeException.class, () -> other.inspectGraph(new byte[] {1, 2, 3, 4, 5}));
            assertStatus(failure, INVALID_ARGUMENT, "TURBO_E_INVALID_ARGUMENT");
        }
    }

    @Test
    public void textCrossesAsTheSameUtf8WhateverTheBridge() {
        // Characters outside the Basic Multilingual Plane, combining marks and a NUL: the rows
        // are addressed in bytes, UTF-16 units and scalars, and every binding must agree.
        String text = "A\u0000b 🚀🚀 é 東京 z.";
        try (Analyzer analyzer = rules()) {
            List<Row> tokens = analyzer.tokenize(text);
            assertFalse(tokens.isEmpty());
            Row last = tokens.get(tokens.size() - 1);
            Row.Range source = last.sourceRange();
            assertEquals(text.getBytes(StandardCharsets.UTF_8).length, source.byteEnd());
            assertEquals(text.length(), source.utf16End());
            assertEquals(text.codePointCount(0, text.length()), source.codepointEnd());
            assertArrayEquals(
                    tokens.toArray(), analyzer.tokenize(text).toArray(), "the same text gives the same rows again");
        }
    }

    /** Asserts a failure is the binding's native exception with {@code code} and {@code name}. */
    protected void assertStatus(Throwable failure, int code, String name) {
        assertEquals(code, binding().statusCode(failure), binding().name() + ": " + failure);
        assertEquals(name, binding().statusName(failure), binding().name() + ": " + failure);
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
