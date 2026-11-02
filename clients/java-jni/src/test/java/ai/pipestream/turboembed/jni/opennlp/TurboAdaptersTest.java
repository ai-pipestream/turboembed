package ai.pipestream.turboembed.jni.opennlp;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.jni.TestPaths;
import ai.pipestream.turboembed.jni.TurboAnalyzer;
import ai.pipestream.turboembed.jni.TurboException;
import ai.pipestream.turboembed.jni.TurboLibrary;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** OpenNLP-style adapters must preserve coordinate spaces and explicit unknowns. */
final class TurboAdaptersTest {

    /** The first sentence of the bundle heads test, as the rule tokenizer splits it. */
    private static final String[] TOKENS = {
        "Ann", "met", "Bob", "in", "Paris", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the", "river", "."
    };
    /** The seeded part-of-speech head's output on {@link #TOKENS}; the adapter must not correct it. */
    private static final String[] POS = {
        "PUNCT", "DET", "ADJ", "DET", "ADJ", "ADJ", "PUNCT", "PUNCT", "VERB", "PUNCT", "PUNCT", "PUNCT", "DET", "ADJ",
        "PUNCT", "PUNCT"
    };

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
    void bundleHeadAdaptersReturnTheHeadsOutputAndExplicitUnknowns() {
        try (TurboPOSTaggerME tagger = new TurboPOSTaggerME(TestPaths.headsBundle())) {
            assertArrayEquals(POS, tagger.tag(TOKENS));
            assertTrue(Arrays.stream(tagger.probs()).allMatch(prob -> prob > 0.0 && prob <= 1.0));
        }

        try (TurboLemmatizerME lemmatizer = new TurboLemmatizerME(TestPaths.headsBundle())) {
            String[] lemmas = lemmatizer.lemmatize(TOKENS, POS);
            assertArrayEquals(
                    new String[] {
                        "Ann", "meet", "Bob", "in", "Pari", ",", "and", "the", "quiet", "old", "dog", "slept", "by", "the",
                        "river", "."
                    },
                    lemmas);
            assertTrue(Arrays.stream(lemmatizer.probs()).allMatch(Double::isNaN), "the lemmatizer head scores no lemma");
        }
    }

    @Test
    void sharedDefaultAnalyzerAdaptersThrowUnsupportedForLearnedTasks() {
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
        String text = "Ann met Bob in Paris, and the quiet old dog slept by the river.";
        try (TurboNameFinderME finder = new TurboNameFinderME(TestPaths.headsBundle())) {
            Span[] tokenSpans = finder.find(TOKENS);
            assertEquals(1, tokenSpans.length);
            Span person = tokenSpans[0];
            assertEquals(Span.CoordinateSpace.TOKEN, person.getCoordinateSpace());
            assertEquals(2, person.getStart());
            assertEquals(3, person.getEnd());
            assertEquals("PER", person.getType());
            assertArrayEquals(new String[] {"Bob"}, person.getCoveredTokens(TOKENS));
            assertThrows(IllegalStateException.class, () -> person.getCoveredText(text));
            assertThrows(IllegalStateException.class, person::getByteStart);
            assertThrows(IllegalStateException.class, person::getCodepointStart);

            double[] probsAfterFind = finder.probs();
            finder.clearAdaptiveData();
            assertArrayEquals(probsAfterFind, finder.probs(), "clearAdaptiveData does not erase last-call probs");
            assertArrayEquals(tokenSpans, finder.find(TOKENS));

            Span[] textSpans = finder.findInText(text);
            assertEquals(1, textSpans.length);
            assertEquals(Span.CoordinateSpace.UTF16, textSpans[0].getCoordinateSpace());
            assertEquals("Bob", textSpans[0].getCoveredText(text).toString());
            assertEquals(2, textSpans[0].getTokenStart());
            assertEquals(3, textSpans[0].getTokenEnd());
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
