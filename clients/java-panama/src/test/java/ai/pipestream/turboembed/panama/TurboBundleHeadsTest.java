package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * The learned layers over the heads of a bundle: the sealed small BERT in testdata with its four
 * seeded heads. Its output is deterministic, so the expected labels are the heads' real output
 * on this text, not linguistically right ones; the binding must not correct them.
 */
final class TurboBundleHeadsTest {

    private static final String TEXT =
            "Ann met Bob in Paris, and the quiet old dog slept by the river. Bob walked the old dog.";
    private static final String[] TOKENS = {
        "Ann", "met", "Bob", "in", "Paris", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river", "."
    };
    private static final List<String> POS = List.of(
            "PUNCT", "DET", "ADJ", "DET", "ADJ", "ADJ", "PUNCT", "PUNCT", "VERB", "PUNCT", "PUNCT", "PUNCT", "DET",
            "ADJ", "PUNCT", "PUNCT");
    private static final List<String> LEMMAS = List.of(
            "Ann", "meet", "Bob", "in", "Pari", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river",
            ".");

    @Test
    void posRowsComeFromThePartOfSpeechHeadWithProbabilities() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(TestPaths.headsBundle()));
                TurboAnalysisResult result = analyzer.posTag(TOKENS)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.POS);
            assertEquals(POS, rows.stream().map(row -> row.label().orElseThrow()).toList());
            for (int i = 0; i < rows.size(); i++) {
                TurboSpanView row = rows.get(i);
                assertTrue(row.viewRange().isEmpty(), "pre-tokenized POS rows have no text view range");
                assertTrue(row.sourceRange().isEmpty(), "pre-tokenized POS rows have no source range");
                assertEquals(new TurboSpanData.TokenRange(i, i + 1L), row.tokenRange().orElseThrow());
                assertTrue(row.probability().isPresent());
                assertTrue(row.probability().getAsDouble() > 0.0 && row.probability().getAsDouble() <= 1.0);
            }
            assertEquals(0.7409762693307227, rows.get(0).probability().orElseThrow(), 1.0e-12);
        }
    }

    @Test
    void lemmaRowsComeFromTheLemmatizerHeadReadingTheGivenTags() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(TestPaths.headsBundle()))) {
            try (TurboAnalysisResult result = analyzer.lemmatize(TOKENS, POS.toArray(String[]::new))) {
                List<TurboSpanView> rows = result.spans(TurboLayerKind.LEMMAS);
                assertEquals(LEMMAS, rows.stream().map(row -> row.label().orElseThrow()).toList());
                assertTrue(rows.stream().allMatch(row -> row.probability().isEmpty()));
            }
            TurboException ex = assertThrows(
                    TurboException.class, () -> analyzer.lemmatize(new String[] {"Ann"}, new String[] {"XYZ"}));
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, ex.code());
            assertTrue(ex.getMessage().contains("not a label of the tag head"), ex.getMessage());
        }
    }

    @Test
    void entityRowsAreFreshPerCallAndFoundInTextWithSourceRanges() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(TestPaths.headsBundle()))) {
            List<TurboSpanData> first = analyzer.findTokensCopied(TOKENS);
            assertEquals(1, first.size());
            assertEntity(first.get(0), 2, 3, "PER", 0.23651067881995255);
            assertTrue(first.get(0).viewRange().isEmpty() && first.get(0).sourceRange().isEmpty());

            analyzer.findTokensCopied(new String[] {"Bo", "gave", "Ann", "a", "map"});
            assertEquals(first, analyzer.findTokensCopied(TOKENS), "an intervening call must not change a later one");

            String sentence = TEXT.substring(0, TEXT.indexOf(". ") + 1);
            try (TurboAnalysisResult result = analyzer.findText(sentence)) {
                List<TurboSpanView> rows = result.spans(TurboLayerKind.ENTITIES);
                assertEquals(1, rows.size());
                assertEquals("Bob", rows.get(0).coveredSourceText(sentence));
                assertEquals(new TurboSpanData.TokenRange(2, 3), rows.get(0).tokenRange().orElseThrow());
                assertEquals("PER", rows.get(0).label().orElseThrow());
            }
        }
    }

    @Test
    void theGraphCarriesEveryHeadLayerFromOneEncoderPass() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(TestPaths.headsBundle()));
                TurboAnalysisResult graph = analyzer.analyzeGraph(TEXT)) {
            assertTrue(graph.hasGraphMagic());
            Set<TurboLayerKind> kinds = EnumSet.noneOf(TurboLayerKind.class);
            for (TurboSpanView row : graph.spans()) {
                kinds.add(row.layerKind());
            }
            assertEquals(EnumSet.complementOf(EnumSet.of(TurboLayerKind.PII)), kinds, "the text holds no PII");

            List<TurboSpanView> tokens = graph.spans(TurboLayerKind.TOKENS);
            List<TurboSpanView> pos = graph.spans(TurboLayerKind.POS);
            List<TurboSpanView> lemmas = graph.spans(TurboLayerKind.LEMMAS);
            List<TurboSpanView> deps = graph.spans(TurboLayerKind.DEPENDENCIES);
            assertEquals(22, tokens.size());
            assertEquals(tokens.size(), pos.size());
            assertEquals(tokens.size(), lemmas.size());
            assertEquals(tokens.size(), deps.size());
            // The first sentence's rows are those of the pre-tokenized tasks.
            assertEquals(POS, pos.subList(0, TOKENS.length).stream().map(row -> row.label().orElseThrow()).toList());
            assertEquals(LEMMAS, lemmas.subList(0, TOKENS.length).stream().map(row -> row.label().orElseThrow()).toList());
            for (int i = 0; i < tokens.size(); i++) {
                assertEquals(tokens.get(i).sourceRange(), pos.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), lemmas.get(i).sourceRange());
                assertEquals(tokens.get(i).sourceRange(), deps.get(i).sourceRange());
                assertEquals(new TurboSpanData.TokenRange(i, i + 1L), deps.get(i).tokenRange().orElseThrow());
            }

            // Dependencies: the relation is the label and the head token is read through dependencyHead.
            long roots = 0;
            for (TurboSpanView dep : deps) {
                assertTrue(dep.label().isPresent());
                assertTrue(dep.probability().isEmpty());
                if (dep.dependencyHead().isEmpty()) {
                    roots++;
                    assertEquals(0L, dep.classification());
                    assertEquals("root", dep.label().orElseThrow());
                } else {
                    long head = dep.dependencyHead().getAsLong();
                    assertEquals(head + 1L, dep.classification());
                    assertTrue(head < tokens.size());
                    assertTrue(head != dep.tokenRange().orElseThrow().start(), "a token is not its own head");
                }
            }
            assertEquals(2, roots, "one root a sentence");
            assertEquals(0L, deps.get(4).dependencyHead().orElseThrow(), "the head of Paris is Ann");
            assertTrue(deps.get(1).dependencyHead().isEmpty(), "met is the first sentence's root");
            assertTrue(pos.get(0).dependencyHead().isEmpty(), "only dependency rows have a head");

            List<TurboSpanView> entities = graph.spans(TurboLayerKind.ENTITIES);
            assertEquals(
                    List.of("Bob", "dog"), entities.stream().map(row -> row.coveredSourceText(TEXT)).toList());
            assertEquals(new TurboSpanData.TokenRange(20, 21), entities.get(1).tokenRange().orElseThrow());

            byte[] bytes = graph.copyGraphBytes();
            List<TurboSpanData> copied = graph.copySpans();
            try (TurboAnalysisResult inspected = analyzer.inspectGraph(bytes)) {
                assertEquals(copied, inspected.copySpans());
                assertEquals(
                        copied.stream().filter(row -> row.dependencyHead().isPresent()).count(),
                        tokens.size() - roots);
            }
        }
    }

    @Test
    void aBundleWithoutHeadsLeavesTheLearnedTasksUnsupportedByName() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(TurboAnalyzer.Config.forBundle(TestPaths.tinyBertBundle()))) {
            TurboException pos = assertThrows(TurboException.class, () -> analyzer.posTag(new String[] {"The"}));
            assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, pos.code());
            assertTrue(pos.getMessage().contains("no head writing pos"), pos.getMessage());
            try (TurboAnalysisResult graph = analyzer.analyzeGraph("The river.")) {
                assertFalse(graph.spans(TurboLayerKind.TOKENS).isEmpty());
                assertTrue(graph.spans(TurboLayerKind.POS).isEmpty());
                assertTrue(graph.spans(TurboLayerKind.DEPENDENCIES).isEmpty());
            }
        }
    }

    private static void assertEntity(TurboSpanData row, long start, long end, String label, double probability) {
        assertEquals(TurboLayerKind.ENTITIES, row.layerKind());
        assertEquals(new TurboSpanData.TokenRange(start, end), row.tokenRange().orElseThrow());
        assertEquals(label, row.label().orElseThrow());
        assertEquals(probability, row.probability().orElseThrow(), 1.0e-12);
    }
}
