package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.TurboException;
import ai.pipestream.turboembed.TurboLayerKind;
import ai.pipestream.turboembed.TurboPiiConfidence;
import ai.pipestream.turboembed.TurboSpanData;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Typed Panama result, coordinate, unsupported-task, PII, and graph tests. */
final class TurboPanamaTest {

    @Test
    void versionStatusLayoutAndMissingModelErrorsAreExact() {
        String version = TurboLibrary.version();
        assertNotNull(version);
        assertTrue(version.startsWith("0.1.0"), "unexpected version: " + version);
        assertEquals("TURBO_OK", TurboLibrary.statusName(TurboLibrary.TURBO_OK));
        assertEquals(
                "TURBO_E_UNSUPPORTED_TASK",
                TurboLibrary.statusName(TurboLibrary.TURBO_E_UNSUPPORTED_TASK));
        assertEquals(120L, TurboLibrary.TURBO_SPAN_LAYOUT.byteSize());
        assertEquals(8L, TurboLibrary.TURBO_SPAN_LAYOUT.byteAlignment());
        assertEquals(
                112L,
                TurboLibrary.TURBO_SPAN_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("prob")));

        Path missingBundle = TestPaths.repoRoot().resolve("testdata/does-not-exist");
        TurboException ex = assertThrows(
                TurboException.class, () -> TurboAnalyzer.create(new TurboAnalyzer.Config("en", missingBundle)));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, ex.code());
        assertEquals("TURBO_E_BUNDLE_NOT_FOUND", ex.statusName());
        assertEquals(7, ex.field(), "the field number of bundle_path");
        assertTrue(TurboLibrary.lastErrorMessage().contains("bundle_path"));
    }

    @Test
    void retiredModelPathFieldsAreRefusedAndTheEarlierDescriptorSizeStillCreates() {
        assertEquals(120L, TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteSize());
        assertEquals(
                (long) TurboLibrary.TURBO_ANALYZER_DESC_SIZE_V1,
                TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("bundle_path")));
        String[] retired = {"sent_model_path", "tok_model_path", "pos_model_path", "lemma_model_path", "ner_model_path"};
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < retired.length; i++) {
                MemorySegment desc = arena.allocate(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT);
                desc.set(ValueLayout.JAVA_INT, 0L, (int) TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteSize());
                long offset = TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteOffset(
                        MemoryLayout.PathElement.groupElement(retired[i]));
                TurboLibrary.writeTextInPlace(arena, desc, offset, "model.bin");
                MemorySegment err = TurboLibrary.allocateError(arena);
                assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, create(arena, desc, err), retired[i]);
                assertEquals(i + 2, err.get(ValueLayout.JAVA_INT, 8L), "turbo_error.field names " + retired[i]);
                String message = TurboLibrary.lastErrorMessage();
                assertTrue(message.contains(retired[i]) && message.contains("retired"), message);
            }

            // A descriptor from before bundle_path is accepted and names no bundle.
            MemorySegment desc = arena.allocate(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT);
            desc.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_ANALYZER_DESC_SIZE_V1);
            TurboLibrary.writeTextInPlace(
                    arena,
                    desc,
                    TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("bundle_path")),
                    TestPaths.headsBundle().toString());
            MemorySegment err = TurboLibrary.allocateError(arena);
            assertEquals(TurboLibrary.TURBO_OK, create(arena, desc, err));
            MemorySegment handle = created;
            try {
                MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
                MemorySegment tokens = TurboLibrary.allocateTextArray(arena, new String[] {"The"});
                int code = (int) TurboLibrary.TURBO_ANALYZER_POS_TAG.invokeExact(handle, tokens, 1, outPtr, err);
                assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, code, "size 104 names no bundle");
            } catch (Throwable t) {
                throw new AssertionError(t);
            } finally {
                try {
                    TurboLibrary.TURBO_ANALYZER_RELEASE.invokeExact(handle);
                } catch (Throwable t) {
                    throw new AssertionError(t);
                }
            }

            // An unknown size is refused as such.
            desc.set(ValueLayout.JAVA_INT, 0L, 112);
            assertEquals(TurboLibrary.TURBO_E_INVALID_STRUCT_SIZE, create(arena, desc, err));
        }
    }

    @Test
    void legacySizedDescriptorAllocationAndModernAndInvalidControlsWork() {
        StructLayout legacyLayout = MemoryLayout.structLayout(
                ValueLayout.JAVA_INT.withName("struct_size"),
                ValueLayout.JAVA_INT.withName("flags"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("language"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("sent_model_path"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("tok_model_path"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("pos_model_path"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("lemma_model_path"),
                TurboLibrary.TURBO_TEXT_LAYOUT.withName("ner_model_path"));
        assertEquals(TurboLibrary.TURBO_ANALYZER_DESC_SIZE_V1, legacyLayout.byteSize());
        assertEquals(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteAlignment(), legacyLayout.byteAlignment());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = TurboLibrary.allocateError(arena);

            // 1. Exact legacy byte size and alignment (104 bytes, alignment 8): no extension bytes exist.
            MemorySegment legacyDesc = arena.allocate(
                    TurboLibrary.TURBO_ANALYZER_DESC_SIZE_V1,
                    TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteAlignment());
            assertEquals(104L, legacyDesc.byteSize());
            legacyDesc.set(ValueLayout.JAVA_INT, 0L, TurboLibrary.TURBO_ANALYZER_DESC_SIZE_V1);
            long langOffset = legacyLayout.byteOffset(MemoryLayout.PathElement.groupElement("language"));
            TurboLibrary.writeTextInPlace(arena, legacyDesc, langOffset, "en");
            assertEquals(TurboLibrary.TURBO_OK, create(arena, legacyDesc, err));
            MemorySegment legacyHandle = created;
            try {
                MemorySegment text = TurboLibrary.allocateText(arena, "Dr. Smith arrived. She waved.");
                MemorySegment sentOut = arena.allocate(ValueLayout.ADDRESS);
                int sentCode = (int) TurboLibrary.TURBO_ANALYZER_DETECT_SENTENCES.invokeExact(
                        legacyHandle, text, sentOut, err);
                assertEquals(TurboLibrary.TURBO_OK, sentCode);
                MemorySegment sentResult = sentOut.get(ValueLayout.ADDRESS, 0L);
                try {
                    MemorySegment spansPtr = arena.allocate(ValueLayout.ADDRESS);
                    MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
                    int spansCode = (int) TurboLibrary.TURBO_ANALYSIS_RESULT_SPANS.invokeExact(
                            sentResult, spansPtr, countPtr, err);
                    assertEquals(TurboLibrary.TURBO_OK, spansCode);
                    assertEquals(2, countPtr.get(ValueLayout.JAVA_INT, 0L));
                } finally {
                    TurboLibrary.TURBO_ANALYSIS_RESULT_RELEASE.invokeExact(sentResult);
                }

                MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
                MemorySegment tokens = TurboLibrary.allocateTextArray(arena, new String[] {"Dr.", "Smith"});
                int posCode = (int) TurboLibrary.TURBO_ANALYZER_POS_TAG.invokeExact(
                        legacyHandle, tokens, 2, outPtr, err);
                assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, posCode);
            } catch (Throwable t) {
                throw new AssertionError(t);
            } finally {
                try {
                    TurboLibrary.TURBO_ANALYZER_RELEASE.invokeExact(legacyHandle);
                } catch (Throwable t) {
                    throw new AssertionError(t);
                }
            }

            // Retired field inside an exact 104-byte allocation is still refused by field number.
            long nerOffset = legacyLayout.byteOffset(MemoryLayout.PathElement.groupElement("ner_model_path"));
            TurboLibrary.writeTextInPlace(arena, legacyDesc, nerOffset, "ner.bin");
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, create(arena, legacyDesc, err));
            assertEquals(6, err.get(ValueLayout.JAVA_INT, 8L));

            // 2. Modern 120-byte descriptor control with sealed heads bundle.
            MemorySegment modernDesc = arena.allocate(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT);
            modernDesc.set(ValueLayout.JAVA_INT, 0L, (int) TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteSize());
            long bundleOffset = TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteOffset(
                    MemoryLayout.PathElement.groupElement("bundle_path"));
            TurboLibrary.writeTextInPlace(arena, modernDesc, bundleOffset, TestPaths.headsBundle().toString());
            assertEquals(TurboLibrary.TURBO_OK, create(arena, modernDesc, err));
            MemorySegment modernHandle = created;
            try {
                MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
                MemorySegment tokens = TurboLibrary.allocateTextArray(arena, new String[] {"The", "river"});
                int posCode = (int) TurboLibrary.TURBO_ANALYZER_POS_TAG.invokeExact(
                        modernHandle, tokens, 2, outPtr, err);
                assertEquals(TurboLibrary.TURBO_OK, posCode);
                MemorySegment posResult = outPtr.get(ValueLayout.ADDRESS, 0L);
                TurboLibrary.TURBO_ANALYSIS_RESULT_RELEASE.invokeExact(posResult);
            } catch (Throwable t) {
                throw new AssertionError(t);
            } finally {
                try {
                    TurboLibrary.TURBO_ANALYZER_RELEASE.invokeExact(modernHandle);
                } catch (Throwable t) {
                    throw new AssertionError(t);
                }
            }

            // 3. Invalid-size controls on a 4-byte allocation: rejected before reading unavailable fields.
            MemorySegment shortDesc = arena.allocate(
                    ValueLayout.JAVA_INT.byteSize(),
                    TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteAlignment());
            assertEquals(4L, shortDesc.byteSize());
            for (int badSize : new int[] {0, 4, 96, 112, 128}) {
                shortDesc.set(ValueLayout.JAVA_INT, 0L, badSize);
                assertEquals(
                        TurboLibrary.TURBO_E_INVALID_STRUCT_SIZE,
                        create(arena, shortDesc, err),
                        "size " + badSize);
                assertEquals(MemorySegment.NULL, created);
            }
        }
    }

    private MemorySegment created;

    private int create(Arena arena, MemorySegment desc, MemorySegment err) {
        MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
        int code;
        try {
            code = (int) TurboLibrary.TURBO_ANALYZER_CREATE.invokeExact(desc, outPtr, err);
        } catch (Throwable t) {
            throw new AssertionError(t);
        }
        created = outPtr.get(ValueLayout.ADDRESS, 0L);
        return code;
    }

    @Test
    void defaultSentenceAndTokenViewsUseSourceRangesWithoutLabelsOrProbabilities() {
        String text = "Hello 🚀 Café 東京! Dr. John Smith works.";
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);

        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult sentences = analyzer.detectSentences(text);
                TurboAnalysisResult tokens = analyzer.tokenize(text)) {

            assertEquals(2, sentences.spanCount());
            for (TurboSpanView sentence : sentences.spans(TurboLayerKind.SENTENCES)) {
                assertEquals(TurboLayerKind.SENTENCES, sentence.layerKind());
                assertEquals("turbo:sentences", sentence.layer());
                assertEquals("", sentence.instance());
                assertTrue(sentence.label().isEmpty(), "covered sentence text must not masquerade as a label");
                assertTrue(sentence.value().isEmpty());
                assertTrue(sentence.probability().isEmpty(), "rule sentences must not invent a probability");
                assertTrue(sentence.tokenRange().isEmpty());
                assertEquals(sentence.viewRange(), sentence.sourceRange());
                assertFalse(sentence.coveredSourceText(text).isEmpty());
            }
            assertEquals("Hello 🚀 Café 東京!", sentences.span(0).coveredSourceText(text));

            List<TurboSpanView> tokenViews = tokens.spans(TurboLayerKind.TOKENS);
            assertFalse(tokenViews.isEmpty());
            for (int i = 0; i < tokenViews.size(); i++) {
                TurboSpanView token = tokenViews.get(i);
                assertEquals(i, token.itemIndex());
                assertEquals(0, token.fragmentIndex());
                assertEquals(0, token.viewId());
                assertTrue(token.label().isEmpty(), "covered token text must not masquerade as a label");
                assertTrue(token.probability().isEmpty(), "rule tokens must not invent a probability");
                assertEquals(new TurboSpanData.TokenRange(i, i + 1L), token.tokenRange().orElseThrow());

                TurboSpanData.CoordinateRange source = token.sourceRange().orElseThrow();
                String byUtf16 = token.coveredSourceText(text);
                String byBytes = new String(
                        Arrays.copyOfRange(utf8, Math.toIntExact(source.byteStart()), Math.toIntExact(source.byteEnd())),
                        StandardCharsets.UTF_8);
                assertEquals(byUtf16, byBytes);
                assertEquals(
                        text.codePointCount(
                                Math.toIntExact(source.utf16Start()), Math.toIntExact(source.utf16End())),
                        source.codepointEnd() - source.codepointStart());
            }
            assertEquals(
                    tokenViews.stream().map(token -> token.coveredSourceText(text)).toList(),
                    analyzer.tokenizeToStrings(text));
        }
    }

    @Test
    void unconfiguredStatisticalTasksAreUnsupportedInsteadOfFallingBack() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            TurboException pos = assertThrows(
                    TurboException.class, () -> analyzer.posTag(new String[] {"The", "rivers", "went"}));
            assertUnsupported(pos);

            TurboException morph = assertThrows(
                    TurboException.class, () -> analyzer.morphology(new String[] {"The", "rivers", "went"}));
            assertUnsupported(morph);

            TurboException lemma = assertThrows(
                    TurboException.class,
                    () -> analyzer.lemmatize(new String[] {"went"}, new String[] {"VERB"}));
            assertUnsupported(lemma);

            TurboException nerTokens = assertThrows(
                    TurboException.class,
                    () -> analyzer.findTokens(new String[] {"John", "Smith", "visited", "New", "York"}));
            assertUnsupported(nerTokens);

            TurboException nerText = assertThrows(
                    TurboException.class,
                    () -> analyzer.findText("John Smith visited New York and emailed alice@example.com."));
            assertUnsupported(nerText);
        }
    }

    @Test
    void emptyInputsAndJavaArgumentValidationDoNotCallFallbacks() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            try (TurboAnalysisResult sentences = analyzer.detectSentences("");
                    TurboAnalysisResult tokens = analyzer.tokenize("");
                    TurboAnalysisResult pos = analyzer.posTag(new String[0]);
                    TurboAnalysisResult morph = analyzer.morphology(new String[0]);
                    TurboAnalysisResult lemmas = analyzer.lemmatize(new String[0], new String[0]);
                    TurboAnalysisResult entities = analyzer.findTokens(new String[0]);
                    TurboAnalysisResult pii = analyzer.detectPii("")) {
                assertEquals(0, sentences.spanCount());
                assertEquals(0, tokens.spanCount());
                assertEquals(0, pos.spanCount());
                assertEquals(0, morph.spanCount());
                assertEquals(0, lemmas.spanCount());
                assertEquals(0, entities.spanCount());
                assertEquals(0, pii.spanCount());
            }

            assertThrows(NullPointerException.class, () -> analyzer.detectSentences(null));
            assertThrows(NullPointerException.class, () -> analyzer.posTag(new String[] {"ok", null}));
            assertThrows(NullPointerException.class, () -> analyzer.morphology(new String[] {"ok", null}));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> analyzer.lemmatize(new String[] {"went"}, new String[] {"VERB", "NOUN"}));
            assertThrows(IndexOutOfBoundsException.class, () -> analyzer.inspectGraph(new byte[4], 3, 2));
        }
    }

    @Test
    void piiIsSeparateTypedLayerWithNormalizedValue() {
        String text = "Email alice@example.com from 192.168.1.42.";
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult result = analyzer.detectPii(text)) {
            List<TurboSpanView> rows = result.spans();
            assertFalse(rows.isEmpty());
            assertTrue(rows.stream().allMatch(row -> row.layerKind() == TurboLayerKind.PII));
            TurboSpanView email = rows.stream()
                    .filter(row -> row.label().orElse("").equals("email"))
                    .findFirst()
                    .orElseThrow();
            assertEquals("alice@example.com", email.coveredSourceText(text));
            assertEquals("alice@example.com", email.value().orElseThrow());
            assertEquals(TurboPiiConfidence.FORMAT, email.piiConfidence().orElseThrow());
            assertTrue(email.probability().isEmpty(), "PII evidence classification is not a probability");
        }
    }

    @Test
    void graphRoundTripUsesStoredSourceAndOnlyConfiguredLayers() {
        String text = "John Smith flew to São Paulo 🚀. Email: test@example.com.";
        byte[] exportedBytes;
        List<TurboSpanData> exportedSpans;

        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult exported = analyzer.analyzeGraph(text)) {
            assertTrue(exported.hasGraph());
            assertTrue(exported.hasGraphMagic());
            assertTrue(exported.graphByteLength() > 32);
            exportedBytes = exported.copyGraphBytes();
            exportedSpans = exported.copySpans();
            assertEquals("TGRF", new String(exportedBytes, 0, 4, StandardCharsets.US_ASCII));

            Set<TurboLayerKind> kinds = EnumSet.noneOf(TurboLayerKind.class);
            for (TurboSpanData span : exportedSpans) {
                kinds.add(span.layerKind());
                assertFalse(span.coveredSourceText(text).isEmpty());
            }
            assertEquals(
                    EnumSet.of(TurboLayerKind.SENTENCES, TurboLayerKind.TOKENS, TurboLayerKind.PII),
                    kinds,
                    "unconfigured POS, lemma, and NER layers must be absent, not fabricated");
        }

        try (TurboAnalyzer otherAnalyzer = TurboAnalyzer.create();
                TurboAnalysisResult inspected = otherAnalyzer.inspectGraph(exportedBytes)) {
            assertTrue(inspected.hasGraphMagic());
            assertArrayEquals(exportedBytes, inspected.copyGraphBytes());
            assertEquals(exportedSpans, inspected.copySpans());
        }

        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            TurboException ex = assertThrows(
                    TurboException.class, () -> analyzer.inspectGraph(new byte[] {1, 2, 3, 4, 5}));
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, ex.code());
        }
    }

    private static void assertUnsupported(TurboException ex) {
        assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, ex.code());
        assertEquals("TURBO_E_UNSUPPORTED_TASK", ex.statusName());
    }
}
