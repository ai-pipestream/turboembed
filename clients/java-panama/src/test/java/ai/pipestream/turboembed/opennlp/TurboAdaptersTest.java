package ai.pipestream.turboembed.opennlp;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.panama.TestPaths;
import ai.pipestream.turboembed.panama.TurboAnalyzer;
import ai.pipestream.turboembed.panama.TurboException;
import ai.pipestream.turboembed.panama.TurboLibrary;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** OpenNLP-style adapters must preserve coordinate spaces and explicit unknowns. */
final class TurboAdaptersTest {

    @Test
    void ruleSentenceAndTokenizerAdaptersUseCoveredSourceTextAndNaNProbabilities() {
        String text = "Dr. John Smith works in New York. Café 🚀 opens!";
        try (TurboSentenceDetectorME detector = new TurboSentenceDetectorME();
                TurboTokenizerME tokenizer = new TurboTokenizerME()) {
            assertEquals(0, detector.probs().length);
            String[] sentences = detector.sentDetect(text);
            assertEquals(2, sentences.length);
            assertEquals("Dr. John Smith works in New York.", sentences[0]);
            assertTrue(Arrays.stream(detector.probs()).allMatch(Double::isNaN));

            Span[] sentenceSpans = detector.sentPosDetect(text);
            assertEquals(2, sentenceSpans.length);
            for (int i = 0; i < sentenceSpans.length; i++) {
                Span span = sentenceSpans[i];
                assertEquals(Span.CoordinateSpace.UTF16, span.getCoordinateSpace());
                assertEquals(sentences[i], span.getCoveredText(text).toString());
                assertTrue(Double.isNaN(span.getProb()));
            }

            String[] tokens = tokenizer.tokenize(text);
            Span[] tokenSpans = tokenizer.tokenizePos(text);
            assertEquals(tokens.length, tokenSpans.length);
            byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
            for (int i = 0; i < tokens.length; i++) {
                Span span = tokenSpans[i];
                assertEquals(tokens[i], span.getCoveredText(text).toString());
                assertEquals(
                        tokens[i],
                        new String(
                                Arrays.copyOfRange(
                                        utf8, Math.toIntExact(span.getByteStart()), Math.toIntExact(span.getByteEnd())),
                                StandardCharsets.UTF_8));
                assertEquals(i, span.getTokenStart());
                assertEquals(i + 1L, span.getTokenEnd());
                assertTrue(Double.isNaN(span.getProb()));
            }
            assertTrue(Arrays.stream(tokenizer.probs()).allMatch(Double::isNaN));
        }
    }

    @Test
    void configuredPosAndDictionaryLemmaAdaptersExposeExplicitUnknowns() {
        String[] tokens = {"The", "driver", "stopped", "the", "bus", "at", "the", "station", "."};
        try (TurboPOSTaggerME tagger = new TurboPOSTaggerME(TestPaths.fixture("pos-fixture.bin"))) {
            assertArrayEquals(
                    new String[] {"DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT", "DET", "NOUN", "PUNCT"},
                    tagger.tag(tokens));
            assertTrue(Arrays.stream(tagger.probs()).allMatch(prob -> prob > 0.0 && prob <= 1.0));
        }

        try (TurboLemmatizerME lemmatizer =
                new TurboLemmatizerME(TestPaths.fixture("lemmatizer-dictionary.txt"))) {
            String[] lemmas = lemmatizer.lemmatize(
                    new String[] {"The", "rivers", "went", "zorped"},
                    new String[] {"DET", "NOUN", "VERB", "NOUN"});
            assertArrayEquals(new String[] {"the", "river", "go", null}, lemmas);
            assertNull(lemmas[3]);
            assertTrue(Arrays.stream(lemmatizer.probs()).allMatch(Double::isNaN));
        }
    }

    @Test
    void sharedDefaultAnalyzerAdaptersThrowUnsupportedForStatisticalTasks() {
        try (TurboAnalyzer shared = TurboAnalyzer.create();
                TurboPOSTaggerME tagger = new TurboPOSTaggerME(shared);
                TurboLemmatizerME lemmatizer = new TurboLemmatizerME(shared);
                TurboNameFinderME finder = new TurboNameFinderME(shared)) {
            TurboException pos = assertThrows(TurboException.class, () -> tagger.tag(new String[] {"The"}));
            TurboException lemma = assertThrows(
                    TurboException.class, () -> lemmatizer.lemmatize(new String[] {"went"}, new String[] {"VERB"}));
            TurboException ner = assertThrows(
                    TurboException.class, () -> finder.find(new String[] {"John", "Smith"}));
            for (TurboException ex : new TurboException[] {pos, lemma, ner}) {
                assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, ex.code());
            }
        }
    }

    @Test
    void nameFinderSpansNeverConflateTokenAndCharacterOffsets() {
        String[] tokens = {
            "Maria", "Lindqvist", "joined", "Harbor", "Light", "Systems", "in", "March", "2019", "."
        };
        String text = "Maria Lindqvist joined Harbor Light Systems in March 2019.";
        try (TurboNameFinderME finder = new TurboNameFinderME(TestPaths.fixture("ner-fixture-rust.bin"))) {
            Span[] tokenSpans = finder.find(tokens);
            assertEquals(3, tokenSpans.length);
            Span person = tokenSpans[0];
            assertEquals(Span.CoordinateSpace.TOKEN, person.getCoordinateSpace());
            assertEquals(0, person.getStart());
            assertEquals(2, person.getEnd());
            assertEquals("person", person.getType());
            assertArrayEquals(new String[] {"Maria", "Lindqvist"}, person.getCoveredTokens(tokens));
            assertThrows(IllegalStateException.class, () -> person.getCoveredText(text));
            assertThrows(IllegalStateException.class, person::getByteStart);
            assertThrows(IllegalStateException.class, person::getCodepointStart);

            double[] probsAfterFind = finder.probs();
            finder.clearAdaptiveData();
            assertArrayEquals(probsAfterFind, finder.probs(), "clearAdaptiveData does not erase last-call probs");
            assertArrayEquals(tokenSpans, finder.find(tokens));

            Span[] textSpans = finder.findInText(text);
            assertEquals(3, textSpans.length);
            assertEquals(Span.CoordinateSpace.UTF16, textSpans[0].getCoordinateSpace());
            assertEquals("Maria Lindqvist", textSpans[0].getCoveredText(text).toString());
            assertEquals(0, textSpans[0].getTokenStart());
            assertEquals(2, textSpans[0].getTokenEnd());
            assertTrue(textSpans[0].hasSourceCoordinates());
        }
    }

    @Test
    void manualSpanFactoriesKeepCoordinateSpacesDistinct() {
        Span token = Span.token(0, 2, "person", 0.9);
        Span chars = Span.utf16(0, 2, "person", 0.9, 0, 3, 0, 2);
        assertFalse(token.equals(chars));
        assertEquals("Ma", chars.getCoveredText("Maria").toString());
        assertThrows(IllegalStateException.class, () -> token.getCoveredText("Maria"));
        assertThrows(IllegalArgumentException.class, () -> Span.token(2, 1, null, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Span.token(0, 1, null, 1.1));
    }

    @Test
    void adapterCloseIsIdempotentAndSharedAnalyzerOwnershipIsHonored() {
        TurboAnalyzer shared = TurboAnalyzer.create();
        TurboTokenizerME sharedTokenizer = new TurboTokenizerME(shared);
        assertArrayEquals(new String[] {"Still", "open", "."}, sharedTokenizer.tokenize("Still open."));
        sharedTokenizer.close();
        sharedTokenizer.close();
        assertThrows(IllegalStateException.class, () -> sharedTokenizer.tokenize("Closed."));

        assertFalse(shared.isClosed(), "a non-owning adapter must not close the shared analyzer");
        try (var result = shared.tokenize("Shared analyzer remains usable.")) {
            assertTrue(result.spanCount() > 0);
        }
        shared.close();

        TurboTokenizerME owned = new TurboTokenizerME();
        owned.close();
        owned.close();
        assertThrows(IllegalStateException.class, () -> owned.tokenize("Closed."));
    }
}
