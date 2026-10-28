package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.opennlp.Span;
import ai.pipestream.turboembed.opennlp.TurboLemmatizerME;
import ai.pipestream.turboembed.opennlp.TurboNameFinderME;
import ai.pipestream.turboembed.opennlp.TurboPOSTaggerME;
import ai.pipestream.turboembed.opennlp.TurboSentenceDetectorME;
import ai.pipestream.turboembed.opennlp.TurboTokenizerME;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * End-to-end JUnit 5 verification of the Java 22+ Panama FFM bindings over {@code libturbo} and
 * the Apache OpenNLP drop-in adapters.
 */
final class TurboPanamaTest {

    private static Path repoRoot() {
        String prop = System.getProperty("turbo.repo.root");
        if (prop != null && !prop.isBlank()) {
            return Path.of(prop).toAbsolutePath().normalize();
        }
        return Path.of("../..").toAbsolutePath().normalize();
    }

    @Test
    void versionStatusAndLastErrorMessageWorkOverPanamaFfm() {
        String version = TurboLibrary.version();
        assertNotNull(version);
        assertTrue(version.startsWith("0.1.0"), "unexpected version: " + version);

        assertEquals("TURBO_OK", TurboLibrary.statusName(TurboLibrary.TURBO_OK));
        assertEquals(
                "TURBO_E_INVALID_ARGUMENT",
                TurboLibrary.statusName(TurboLibrary.TURBO_E_INVALID_ARGUMENT));
        assertEquals(
                "TURBO_E_BUNDLE_NOT_FOUND",
                TurboLibrary.statusName(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND));

        Path missingModel = repoRoot().resolve("testdata/does-not-exist.bin");
        TurboException ex = assertThrows(
                TurboException.class,
                () -> TurboAnalyzer.create(new TurboAnalyzer.Config("en", missingModel, null, null, null, null)));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, ex.code());
        assertEquals("TURBO_E_BUNDLE_NOT_FOUND", ex.statusName());
        assertTrue(TurboLibrary.lastErrorMessage().contains("sent_model_path"));
    }

    @Test
    void openNlpSentenceDetectorAndCustomModelFixturesWork() {
        String text = "Dr. John Smith works in New York. He flew to San Francisco yesterday!";
        try (TurboSentenceDetectorME detector = new TurboSentenceDetectorME()) {
            String[] sentences = detector.sentDetect(text);
            assertEquals(2, sentences.length);
            assertEquals("Dr. John Smith works in New York.", sentences[0]);
            assertEquals("He flew to San Francisco yesterday!", sentences[1]);

            Span[] spans = detector.sentPosDetect(text);
            assertEquals(2, spans.length);
            for (int i = 0; i < spans.length; i++) {
                assertEquals(sentences[i], text.substring(spans[i].getStart(), spans[i].getEnd()));
                assertEquals(sentences[i], spans[i].getCoveredText(text).toString());
            }
            assertEquals(2, detector.probs().length);
        }

        Path sentModel = repoRoot().resolve("analysis/goldens/data/fixtures/en-plain.bin");
        try (TurboSentenceDetectorME meDetector = new TurboSentenceDetectorME(sentModel)) {
            Span[] spans = meDetector.sentPosDetect("First sentence here. Second sentence follows.");
            assertEquals(2, spans.length);
            assertTrue(meDetector.probs()[0] > 0.0);
        }
    }

    @Test
    void openNlpTokenizerPosTaggerAndLemmatizerWork() {
        try (TurboAnalyzer shared = TurboAnalyzer.create();
                TurboTokenizerME tokenizer = new TurboTokenizerME(shared);
                TurboPOSTaggerME posTagger = new TurboPOSTaggerME(shared);
                TurboLemmatizerME lemmatizer = new TurboLemmatizerME(shared)) {

            String text = "The rivers went fast.";
            String[] tokens = tokenizer.tokenize(text);
            assertArrayEquals(new String[] {"The", "rivers", "went", "fast", "."}, tokens);

            Span[] tokSpans = tokenizer.tokenizePos(text);
            assertEquals(tokens.length, tokSpans.length);
            for (int i = 0; i < tokens.length; i++) {
                assertEquals(tokens[i], text.substring(tokSpans[i].getStart(), tokSpans[i].getEnd()));
            }

            String[] tags = posTagger.tag(tokens);
            assertEquals(tokens.length, tags.length);
            assertEquals(tokens.length, posTagger.probs().length);
            for (double p : posTagger.probs()) {
                assertTrue(p > 0.0 && p <= 1.0);
            }

            String[] explicitTags = new String[] {"DET", "NOUN", "VERB", "ADV", "PUNCT"};
            String[] lemmas = lemmatizer.lemmatize(tokens, explicitTags);
            assertEquals(tokens.length, lemmas.length);
            assertEquals("the", lemmas[0]);
            assertEquals("river", lemmas[1]);
            assertEquals("go", lemmas[2]);
            assertEquals(tokens.length, lemmatizer.probs().length);
        }

        Path tokModel = repoRoot().resolve("analysis/goldens/data/fixtures/tokenizer-abbreviations.bin");
        try (TurboTokenizerME meTok = new TurboTokenizerME(tokModel)) {
            String[] toks = meTok.tokenize("Hello world!");
            assertTrue(toks.length >= 2);
            assertEquals(toks.length, meTok.probs().length);
        }
    }

    @Test
    void openNlpTokenNameFinderExtractsEntitiesAndPii() {
        try (TurboNameFinderME finder = new TurboNameFinderME()) {
            String[] tokens = new String[] {
                "John", "Smith", "visited", "New", "York", "and", "San", "Francisco"
            };
            Span[] spans = finder.find(tokens);
            assertTrue(Arrays.stream(spans).anyMatch(
                    s -> s.getStart() == 0 && s.getEnd() == 2 && "person".equals(s.getType())));
            assertTrue(Arrays.stream(spans).anyMatch(
                    s -> s.getStart() == 3 && s.getEnd() == 5 && "location".equals(s.getType())));
            assertTrue(Arrays.stream(spans).anyMatch(
                    s -> s.getStart() == 6 && s.getEnd() == 8 && "location".equals(s.getType())));
            assertEquals(spans.length, finder.probs().length);

            finder.clearAdaptiveData();
            assertEquals(0, finder.probs().length);

            String rawText = "Contact John Smith in New York at alice@example.com or 192.168.1.42.";
            Span[] textSpans = finder.findInText(rawText);
            assertTrue(Arrays.stream(textSpans).anyMatch(
                    s -> "person".equals(s.getType())
                            && "John Smith".equals(rawText.substring(s.getStart(), s.getEnd()))));
            assertTrue(Arrays.stream(textSpans).anyMatch(
                    s -> "location".equals(s.getType())
                            && "New York".equals(rawText.substring(s.getStart(), s.getEnd()))));
            assertTrue(Arrays.stream(textSpans).anyMatch(
                    s -> "email".equals(s.getType())
                            && "alice@example.com".equals(rawText.substring(s.getStart(), s.getEnd()))));
        }
    }

    @Test
    void utf16OffsetParityAcrossAsciiMultibyteCjkAndAstralEmojis() {
        // 🚀 (U+1F680), 🦀 (U+1F980), 🧪 (U+1F9EA), 😀 (U+1F600): each is 4 UTF-8 bytes, 2 UTF-16 code units, 1 codepoint.
        String text = "🚀 Café 東京 🦀 Dr. John Smith 🧪 visited São Paulo 😀 face and emailed dev@example.com!";
        byte[] utf8Bytes = text.getBytes(StandardCharsets.UTF_8);

        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboTokenizerME tokenizer = new TurboTokenizerME(analyzer);
                TurboNameFinderME nameFinder = new TurboNameFinderME(analyzer)) {

            Span[] tokSpans = tokenizer.tokenizePos(text);
            String[] tokStrings = tokenizer.tokenize(text);
            assertEquals(tokSpans.length, tokStrings.length);
            assertTrue(tokSpans.length >= 12);

            for (int i = 0; i < tokSpans.length; i++) {
                Span s = tokSpans[i];
                String viaUtf16 = text.substring(s.getStart(), s.getEnd());
                String viaByteSlice = new String(
                        Arrays.copyOfRange(utf8Bytes, s.getByteStart(), s.getByteEnd()),
                        StandardCharsets.UTF_8);
                int cpCount = text.codePointCount(s.getStart(), s.getEnd());
                assertEquals(tokStrings[i], viaUtf16, "UTF-16 substring mismatch at token " + i);
                assertEquals(tokStrings[i], viaByteSlice, "UTF-8 byte slice mismatch at token " + i);
                assertEquals(cpCount, s.getCodepointEnd() - s.getCodepointStart());
            }

            Span[] entities = nameFinder.findInText(text);
            assertTrue(entities.length > 0);
            for (Span entity : entities) {
                String coveredUtf16 = text.substring(entity.getStart(), entity.getEnd());
                String coveredUtf8 = new String(
                        Arrays.copyOfRange(utf8Bytes, entity.getByteStart(), entity.getByteEnd()),
                        StandardCharsets.UTF_8);
                assertEquals(coveredUtf8, coveredUtf16);
            }
            assertTrue(Arrays.stream(entities).anyMatch(
                    s -> "John Smith".equals(text.substring(s.getStart(), s.getEnd()))));
            assertTrue(Arrays.stream(entities).anyMatch(
                    s -> "São Paulo".equals(text.substring(s.getStart(), s.getEnd()))));
            assertTrue(Arrays.stream(entities).anyMatch(
                    s -> "😀 face".equals(text.substring(s.getStart(), s.getEnd()))));
            assertTrue(Arrays.stream(entities).anyMatch(
                    s -> "dev@example.com".equals(text.substring(s.getStart(), s.getEnd()))));
        }
    }

    @Test
    void documentGraphWireExportAndInspectionRoundTrip() {
        String text = "John Smith flew from New York to São Paulo 🚀. Email: test@example.com.";
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            DocumentGraphBuffer exported = analyzer.analyzeGraph(text);
            assertTrue(exported.hasValidMagic());
            assertTrue(exported.byteLength() > 32);
            assertFalse(exported.spans().isEmpty());

            DocumentGraphBuffer inspected = analyzer.inspectGraph(text, exported.wireBytes());
            assertTrue(inspected.hasValidMagic());
            assertEquals(exported.spans(), inspected.spans());
            assertTrue(inspected.spans().stream().anyMatch(s -> "sentence".equals(s.label())));
            assertTrue(inspected.spans().stream().anyMatch(s -> "person".equals(s.label())));
            assertTrue(inspected.spans().stream().anyMatch(s -> "location".equals(s.label())));

            for (TurboSpan span : inspected.spans()) {
                String sub = span.coveredText(text);
                assertFalse(sub.isEmpty());
            }

            // Corrupted wire bytes must raise TurboException with TURBO_E_INVALID_ARGUMENT
            byte[] badBytes = new byte[] {1, 2, 3, 4, 5};
            TurboException ex = assertThrows(
                    TurboException.class, () -> analyzer.inspectGraph(text, badBytes));
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, ex.code());
        }
    }

    @Test
    void turboSessionEmbedsTinyBertBundleAndChecksLifecycle() {
        Path bundle = repoRoot().resolve("testdata/tiny-bert-bundle");
        try (TurboSession session = TurboSession.open(bundle)) {
            float[][] vectors = session.embed(List.of("hello world", "panama foreign function memory"));
            assertEquals(2, vectors.length);
            assertTrue(vectors[0].length > 0);
            assertEquals(vectors[0].length, vectors[1].length);

            // Rerank on an embed-only model must surface TURBO_E_UNSUPPORTED_TASK cleanly
            TurboException unsupported = assertThrows(
                    TurboException.class,
                    () -> session.rerank("query", List.of("passage one")));
            assertEquals("TURBO_E_UNSUPPORTED_TASK", unsupported.statusName());
        }

        TurboAnalyzer analyzer = TurboAnalyzer.create();
        analyzer.close();
        // Idempotent second close() must be a safe no-op
        analyzer.close();
        assertThrows(IllegalStateException.class, () -> analyzer.tokenize("after close"));
    }
}
