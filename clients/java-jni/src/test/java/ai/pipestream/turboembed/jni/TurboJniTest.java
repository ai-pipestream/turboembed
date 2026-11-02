package ai.pipestream.turboembed.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** The bridge itself: loading, the native strings, errors, handles and lifetimes. */
final class TurboJniTest {
    private static final String TEXT = "Lifetime views stay typed. Copies survive!";

    @Test
    void theLibraryLoadsOnceAndAnswersVersionStatusNamesAndTheLastError() {
        TurboLibrary.load();
        TurboLibrary.load();
        String version = TurboLibrary.version();
        assertNotNull(version);
        assertTrue(version.startsWith("0.1.0"), "unexpected version: " + version);
        assertTrue(version.endsWith(" cpu"), "the backend list ends with cpu: " + version);
        assertEquals("TURBO_OK", TurboLibrary.statusName(TurboLibrary.TURBO_OK));
        assertEquals("TURBO_E_UNSUPPORTED_TASK", TurboLibrary.statusName(TurboLibrary.TURBO_E_UNSUPPORTED_TASK));
        assertEquals("TURBO_E_UNKNOWN", TurboLibrary.statusName(-7));
        assertEquals(28, TurboLibrary.TURBO_SPAN_INTS);

        TurboException ex = assertThrows(
                TurboException.class,
                () -> TurboAnalyzer.create(new TurboAnalyzer.Config("en", TestPaths.repoRoot().resolve("testdata/does-not-exist"))));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, ex.code());
        assertEquals("TURBO_E_BUNDLE_NOT_FOUND", ex.statusName());
        assertEquals(7, ex.field(), "the field number of bundle_path");
        assertTrue(ex.getMessage().contains("field=7"), ex.getMessage());
        assertTrue(TurboLibrary.lastErrorMessage().contains("bundle_path"), TurboLibrary.lastErrorMessage());
    }

    @Test
    void bridgeRefusalsSetTheLastErrorMessageToo() {
        TurboException ex = assertThrows(TurboException.class, () -> TurboJni.resultRows(0L));
        assertEquals("TURBO_E_INVALID_HANDLE", ex.statusName());
        assertTrue(ex.getMessage().contains("result handle is released"), ex.getMessage());
        assertTrue(TurboLibrary.lastErrorMessage().contains("result handle is released"), TurboLibrary.lastErrorMessage());
        TurboException mismatch = assertThrows(
                TurboException.class, () -> TurboJni.analyzerPosTag(0L, new byte[] {1, 2}, new int[] {1}));
        assertEquals("TURBO_E_INVALID_HANDLE", mismatch.statusName());
        assertTrue(TurboLibrary.lastErrorMessage().contains("analyzer handle is released"), TurboLibrary.lastErrorMessage());
    }

    @Test
    void javaArgumentValidationHappensBeforeTheBridge() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            assertThrows(NullPointerException.class, () -> analyzer.detectSentences(null));
            assertThrows(NullPointerException.class, () -> analyzer.posTag(new String[] {"ok", null}));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> analyzer.lemmatize(new String[] {"went"}, new String[] {"VERB", "NOUN"}));
            assertThrows(IndexOutOfBoundsException.class, () -> analyzer.inspectGraph(new byte[4], 3, 2));
            assertThrows(IndexOutOfBoundsException.class, () -> analyzer.inspectGraph(new byte[4], -1, 2));
            assertEquals("en", new TurboAnalyzer.Config(" \t", null).language());
            assertEquals("fr", TurboAnalyzer.Config.forLanguage("fr").language());
        }
    }

    @Test
    void copiedSpansSurviveCloseWhileViewsFail() {
        TurboAnalysisResult result;
        TurboSpanView view;
        List<TurboSpanData> copies;
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            result = analyzer.tokenize(TEXT);
            view = result.span(0);
            copies = result.copySpans();
            assertEquals("Lifetime", view.coveredSourceText(TEXT));
        }

        assertFalse(result.isClosed(), "closing the analyzer must not close an independent result");
        assertEquals("Lifetime", view.coveredSourceText(TEXT));
        result.close();
        result.close();
        assertTrue(result.isClosed());
        assertThrows(IllegalStateException.class, view::layerKind);
        assertThrows(IllegalStateException.class, () -> result.span(0));
        assertThrows(IllegalStateException.class, result::copySpans);
        assertThrows(IllegalStateException.class, result::copyGraphBytes);
        assertThrows(IllegalStateException.class, result::retain);

        assertEquals("Lifetime", copies.get(0).coveredSourceText(TEXT));
        assertThrows(UnsupportedOperationException.class, () -> copies.add(copies.get(0)));
    }

    @Test
    void retainedResultOwnerSurvivesOriginalOwnerClose() {
        TurboAnalysisResult original;
        TurboAnalysisResult retained;
        TurboSpanView originalView;
        TurboSpanView retainedView;
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            original = analyzer.analyzeGraph(TEXT);
            retained = original.retain();
            originalView = original.spans(TurboLayerKind.TOKENS).get(1);
            retainedView = retained.spans(TurboLayerKind.TOKENS).get(1);
        }

        original.close();
        assertThrows(IllegalStateException.class, originalView::layerKind);
        assertEquals("views", retainedView.coveredSourceText(TEXT));
        assertEquals("views", retained.copySpan(retainedView.index()).coveredSourceText(TEXT));
        assertTrue(retained.hasGraph());
        assertTrue(retained.hasGraphMagic(), "the graph crosses from the native result the retained owner still holds");
        assertEquals(retained.graphByteLength(), retained.copyGraphBytes().length);
        retained.close();
        retained.close();
        assertThrows(IllegalStateException.class, retainedView::layerKind);
    }

    @Test
    void theGraphCrossesOnceAndCopiesOutExplicitly() {
        TurboAnalysisResult result;
        byte[] copied;
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            result = analyzer.analyzeGraph(TEXT);
            assertTrue(result.hasGraph());
            assertTrue(result.hasGraphMagic());
            copied = result.copyGraphBytes();
            byte[] again = result.copyGraphBytes();
            assertArrayEquals(copied, again);
            assertTrue(copied != again, "each copy-out is the caller's own array");
            copied[0] = 'X';
            assertTrue(result.hasGraphMagic(), "a caller's edit of a copy does not reach the result");
            try (TurboAnalysisResult plain = analyzer.tokenize(TEXT)) {
                assertFalse(plain.hasGraph());
                assertEquals(0L, plain.graphByteLength());
                assertFalse(plain.hasGraphMagic());
                assertEquals(0, plain.copyGraphBytes().length);
            }
        }
        result.close();
        assertThrows(IllegalStateException.class, result::copyGraphBytes);
        assertEquals(result.graphByteLength(), copied.length);
    }

    @Test
    void retainedAnalyzerSurvivesOriginalAnalyzerClose() {
        TurboAnalyzer original = TurboAnalyzer.create();
        TurboAnalyzer retained = original.retain();
        original.close();
        original.close();
        assertTrue(original.isClosed());
        assertFalse(retained.isClosed());
        assertThrows(IllegalStateException.class, () -> original.tokenize(TEXT));
        assertThrows(IllegalStateException.class, original::retain);

        try (TurboAnalysisResult result = retained.tokenize(TEXT)) {
            assertEquals("Lifetime", result.span(0).coveredSourceText(TEXT));
        }
        retained.close();
        retained.close();
        assertThrows(IllegalStateException.class, () -> retained.tokenize(TEXT));
    }

    @Test
    void spanIndexesAreChecked() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult result = analyzer.tokenize("One token.")) {
            assertThrows(IndexOutOfBoundsException.class, () -> result.span(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> result.span(result.spanCount()));
            assertThrows(IndexOutOfBoundsException.class, () -> result.copySpan(result.spanCount()));
            assertEquals(result.spanCount(), result.spans().size());
            assertEquals(result.copySpans(), result.spans().stream().map(TurboSpanView::materialize).toList());
        }
    }

    @Test
    void sessionCloseIsIdempotentAndCallsAfterCloseFailBeforeTheBridge() {
        TurboSession session = TurboSession.open(TestPaths.tinyBertBundle());
        session.close();
        session.close();
        assertTrue(session.isClosed());
        assertThrows(IllegalStateException.class, () -> session.embed(List.of("after close")));
        assertThrows(IllegalStateException.class, () -> session.rerank("query", List.of("passage")));
        try (TurboSession open = TurboSession.open(TestPaths.tinyBertBundle())) {
            assertThrows(NullPointerException.class, () -> open.embed(null));
            assertThrows(NullPointerException.class, () -> open.embed(java.util.Arrays.asList("ok", null)));
            TurboException ex = assertThrows(TurboException.class, () -> open.embed(List.of()));
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, ex.code(), "an empty batch is the library's refusal");
        }
        TurboException missing = assertThrows(
                TurboException.class, () -> TurboSession.open(TestPaths.repoRoot().resolve("testdata/does-not-exist")));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, missing.code());
    }

    @Test
    void analyzerResultAndSessionReadsAreSafeUnderConcurrentUse() throws Exception {
        String text = "Concurrent readers keep exact source offsets 🚀.";
        ExecutorService executor = Executors.newFixedThreadPool(8);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult shared = analyzer.tokenize(text);
                TurboSession session = TurboSession.open(TestPaths.tinyBertBundle())) {
            List<String> expected = surfaces(shared, text);
            float[][] vector = session.embed(List.of(text));

            List<Future<List<String>>> futures = new java.util.ArrayList<>();
            for (int i = 0; i < 32; i++) {
                futures.add(executor.submit(() -> {
                    List<String> fromShared = surfaces(shared, text);
                    float[][] again = session.embed(List.of(text));
                    assertArrayEquals(vector[0], again[0], 0.0f);
                    try (TurboAnalysisResult own = analyzer.tokenize(text)) {
                        List<String> fromOwn = surfaces(own, text);
                        assertEquals(fromShared, fromOwn);
                        return fromOwn;
                    }
                }));
            }
            for (Future<List<String>> future : futures) {
                assertEquals(expected, future.get());
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private static List<String> surfaces(TurboAnalysisResult result, String text) {
        return result.spans().stream().map(span -> span.coveredSourceText(text)).collect(Collectors.toList());
    }
}
