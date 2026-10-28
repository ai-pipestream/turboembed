package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Configured statistical model fixtures: predictions must come from these files, not defaults. */
final class TurboModelFixturesTest {

    @Test
    void sentenceAndTokenizerModelsReportProbabilities() {
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en",
                TestPaths.fixture("en-plain.bin"),
                TestPaths.fixture("tokenizer-abbreviations.bin"),
                null,
                null,
                null);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult sentences = analyzer.detectSentences(
                        "First sentence here. Second sentence follows.");
                TurboAnalysisResult tokens = analyzer.tokenize("Hello world!")) {
            assertEquals(2, sentences.spanCount());
            assertTrue(sentences.spans().stream()
                    .allMatch(span -> span.probability().isPresent()
                            && span.probability().getAsDouble() > 0.0
                            && span.probability().getAsDouble() <= 1.0));
            assertTrue(tokens.spanCount() >= 2);
            assertTrue(tokens.spans().stream()
                    .allMatch(span -> span.probability().isPresent()
                            && span.probability().getAsDouble() > 0.0
                            && span.probability().getAsDouble() <= 1.0));
        }
    }

    @Test
    void posFixtureProducesTokenOnlyModelRows() {
        String[] tokens = {"The", "driver", "stopped", "the", "bus", "at", "the", "station", "."};
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en", null, null, TestPaths.fixture("pos-fixture.bin"), null, null);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult result = analyzer.posTag(tokens)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.POS);
            assertEquals(tokens.length, rows.size());
            // This is the tiny fixture model's actual production-tagger output, including
            // its surprising PUNCT prediction for "at"; the binding must not correct it.
            assertEquals(
                    List.of("DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT", "DET", "NOUN", "PUNCT"),
                    rows.stream().map(row -> row.label().orElseThrow()).toList());
            for (int i = 0; i < rows.size(); i++) {
                TurboSpanView row = rows.get(i);
                assertTrue(row.viewRange().isEmpty(), "pre-tokenized POS rows have no text view range");
                assertTrue(row.sourceRange().isEmpty(), "pre-tokenized POS rows have no source range");
                assertEquals(new TurboSpanData.TokenRange(i, i + 1L), row.tokenRange().orElseThrow());
                assertTrue(row.probability().isPresent());
                assertTrue(row.probability().getAsDouble() > 0.0 && row.probability().getAsDouble() <= 1.0);
            }
        }
    }

    @Test
    void dictionaryLemmasUseProductionDictionarySemanticsAndExplicitMisses() {
        String[] tokens = {"The", "rivers", "went", "leaves", "line", "silt", "U.S.", "zorped"};
        String[] tags = {"DET", "NOUN", "VERB", "NOUN", "NOUN", "NOUN", "PROPN", "NOUN"};
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en", null, null, null, TestPaths.fixture("lemmatizer-dictionary.txt"), null);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult result = analyzer.lemmatize(tokens, tags)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.LEMMAS);
            assertEquals(tokens.length, rows.size());
            assertEquals("the", rows.get(0).label().orElseThrow());
            assertEquals("river", rows.get(1).label().orElseThrow());
            assertEquals("go", rows.get(2).label().orElseThrow(), "the last duplicate dictionary row wins");
            assertEquals("leaf", rows.get(3).label().orElseThrow(), "the first #-separated lemma wins");
            assertEquals("line", rows.get(4).label().orElseThrow(), "fields after the lemma are ignored");
            assertTrue(rows.get(5).label().isEmpty(), "an empty # lemma field is an explicit miss");
            assertEquals("United States", rows.get(6).label().orElseThrow());
            assertTrue(rows.get(7).label().isEmpty(), "a dictionary miss must not become a lowercased token");
            assertTrue(rows.stream().allMatch(row -> row.probability().isEmpty()));
        }
    }

    @Test
    void maxentLemmaFixtureProducesModelLemmasAndProbabilities() {
        String[] tokens = {"stopped", "runs", "fallen"};
        String[] tags = {"VBD", "VBZ", "VBN"};
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en", null, null, null, TestPaths.fixture("lemma-fixture-rust.bin"), null);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult result = analyzer.lemmatize(tokens, tags)) {
            List<TurboSpanView> rows = result.spans(TurboLayerKind.LEMMAS);
            assertEquals(
                    List.of("stop", "run", "fall"),
                    rows.stream().map(row -> row.label().orElseThrow()).toList());
            assertTrue(rows.stream().allMatch(row -> row.probability().isPresent()));
        }
    }

    @Test
    void nerFixtureIsFreshPerCallAndFindsTextWithSourceRanges() {
        String[] tokens = {
            "Maria", "Lindqvist", "joined", "Harbor", "Light", "Systems", "in", "March", "2019", "."
        };
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en", null, null, null, null, TestPaths.fixture("ner-fixture-rust.bin"));
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config)) {
            List<TurboSpanData> first = analyzer.findTokensCopied(tokens);
            assertEquals(3, first.size());
            assertEntity(first.get(0), 0, 2, "person");
            assertEntity(first.get(1), 3, 6, "organization");
            assertEntity(first.get(2), 7, 9, "date");
            assertTrue(first.stream().allMatch(row -> row.viewRange().isEmpty() && row.sourceRange().isEmpty()));

            analyzer.findTokensCopied(new String[] {"Bo", "gave", "Ann", "a", "map"});
            List<TurboSpanData> repeated = analyzer.findTokensCopied(tokens);
            assertEquals(first, repeated, "an intervening call must not leak adaptive data into a later call");

            String text = "Maria Lindqvist joined Harbor Light Systems in March 2019.";
            try (TurboAnalysisResult result = analyzer.findText(text)) {
                List<TurboSpanView> rows = result.spans(TurboLayerKind.ENTITIES);
                assertEquals(3, rows.size());
                assertEquals("Maria Lindqvist", rows.get(0).coveredSourceText(text));
                assertEquals("Harbor Light Systems", rows.get(1).coveredSourceText(text));
                assertEquals("March 2019", rows.get(2).coveredSourceText(text));
                assertTrue(rows.stream().allMatch(row -> row.tokenRange().isPresent()));
            }
        }
    }

    @Test
    void nerBioFixturePredictionIsNotOverriddenByARecognizer() {
        // Regression for the rejected binding: the production NameFinder calls
        // the single token "City" a person at low confidence. A hardcoded
        // location recognizer formerly replaced that prediction with 1.0.
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en", null, null, null, null, TestPaths.fixture("ner-bio-fixture.bin"));
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult result = analyzer.findTokens(new String[] {"City"})) {
            assertEquals(1, result.spanCount());
            TurboSpanView entity = result.span(0);
            assertEquals(TurboLayerKind.ENTITIES, entity.layerKind());
            assertEquals("person", entity.label().orElseThrow());
            assertEquals(new TurboSpanData.TokenRange(0, 1), entity.tokenRange().orElseThrow());
            assertEquals(0.23293448617830537, entity.probability().orElseThrow(), 1.0e-12);
        }
    }

    @Test
    void fullyConfiguredGraphContainsEveryTypedLayerAndModelProbabilities() {
        TurboAnalyzer.Config config = new TurboAnalyzer.Config(
                "en",
                TestPaths.fixture("en-plain.bin"),
                TestPaths.fixture("tokenizer-abbreviations.bin"),
                TestPaths.fixture("pos-fixture.bin"),
                TestPaths.fixture("lemmatizer-dictionary.txt"),
                TestPaths.fixture("ner-fixture-rust.bin"));
        String text = "Maria Lindqvist joined Harbor Light Systems in March 2019. Email alice@example.com.";
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config);
                TurboAnalysisResult graph = analyzer.analyzeGraph(text)) {
            assertTrue(graph.hasGraphMagic());
            Set<TurboLayerKind> kinds = EnumSet.noneOf(TurboLayerKind.class);
            for (TurboSpanView row : graph.spans()) {
                kinds.add(row.layerKind());
            }
            assertEquals(EnumSet.allOf(TurboLayerKind.class), kinds);

            List<TurboSpanView> tokens = graph.spans(TurboLayerKind.TOKENS);
            List<TurboSpanView> pos = graph.spans(TurboLayerKind.POS);
            List<TurboSpanView> lemmas = graph.spans(TurboLayerKind.LEMMAS);
            assertFalse(tokens.isEmpty());
            assertEquals(tokens.size(), pos.size());
            assertEquals(tokens.size(), lemmas.size());
            assertTrue(graph.spans(TurboLayerKind.SENTENCES).stream()
                    .allMatch(row -> row.probability().isPresent()));
            assertTrue(tokens.stream().allMatch(row -> row.probability().isPresent()));
            assertTrue(pos.stream().allMatch(row -> row.label().isPresent() && row.probability().isPresent()));
            assertTrue(lemmas.stream().allMatch(row -> row.probability().isEmpty()));
            assertTrue(graph.spans(TurboLayerKind.ENTITIES).stream()
                    .anyMatch(row -> "Maria Lindqvist".equals(row.coveredSourceText(text))));
            assertTrue(graph.spans(TurboLayerKind.PII).stream()
                    .anyMatch(row -> "alice@example.com".equals(row.coveredSourceText(text))));

            byte[] bytes = graph.copyGraphBytes();
            List<TurboSpanData> copied = graph.copySpans();
            try (TurboAnalysisResult inspected = analyzer.inspectGraph(bytes)) {
                assertEquals(copied, inspected.copySpans());
            }
        }
    }

    @Test
    void lemmaDictionaryFileCanBeSelectedByPathWithoutNativeFallback() {
        Path dictionary = TestPaths.fixture("lemmatizer-dictionary.txt");
        assertTrue(dictionary.toString().endsWith(".txt"));
        String[] lemmas;
        TurboAnalyzer.Config config = new TurboAnalyzer.Config("en", null, null, null, dictionary, null);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create(config)) {
            lemmas = analyzer.lemmatizeCopied(
                            new String[] {"The", "zorped"}, new String[] {"DET", "NOUN"})
                    .stream()
                    .map(row -> row.label().orElse(null))
                    .toArray(String[]::new);
        }
        assertArrayEquals(new String[] {"the", null}, lemmas);
        assertNull(lemmas[1]);
    }

    private static void assertEntity(TurboSpanData row, long start, long end, String label) {
        assertEquals(TurboLayerKind.ENTITIES, row.layerKind());
        assertEquals(new TurboSpanData.TokenRange(start, end), row.tokenRange().orElseThrow());
        assertEquals(label, row.label().orElseThrow());
        assertTrue(row.probability().isPresent());
        assertTrue(row.probability().getAsDouble() > 0.98);
    }
}
