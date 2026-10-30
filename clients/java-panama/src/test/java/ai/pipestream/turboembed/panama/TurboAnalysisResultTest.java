package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Lifetime, retain, copy-out, and concurrency tests for typed native result views. */
final class TurboAnalysisResultTest {
    private static final String TEXT = "Lifetime views stay typed. Copies survive!";

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
            original = analyzer.tokenize(TEXT);
            retained = original.retain();
            originalView = original.span(1);
            retainedView = retained.span(1);
        }

        original.close();
        assertThrows(IllegalStateException.class, originalView::layerKind);
        assertEquals("views", retainedView.coveredSourceText(TEXT));
        assertEquals("views", retained.copySpan(1).coveredSourceText(TEXT));
        assertTrue(retained.withSpansSegment(seg -> seg.byteSize()) > 0L);
        assertTrue(retained.withStringPoolSegment(seg -> seg.byteSize()) > 0L);
        retained.close();
        retained.close();
        assertThrows(IllegalStateException.class, retainedView::layerKind);
    }

    @Test
    void escapedSegmentsAndSlicesAreInvalidatedWhenCallbackReturns() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult result = analyzer.analyzeGraph(TEXT)) {
            AtomicReference<MemorySegment> escapedSlice = new AtomicReference<>();
            MemorySegment escapedSpans = result.withSpansSegment(segment -> {
                escapedSlice.set(segment.asSlice(0L, TurboLibrary.TURBO_SPAN_LAYOUT.byteSize()));
                assertTrue(segment.scope().isAlive());
                return segment;
            });
            assertFalse(escapedSpans.scope().isAlive());
            assertFalse(escapedSlice.get().scope().isAlive());
            assertThrows(
                    IllegalStateException.class,
                    () -> escapedSpans.get(ValueLayout.JAVA_INT, 0L));
            assertThrows(
                    IllegalStateException.class,
                    () -> escapedSlice.get().get(ValueLayout.JAVA_INT, 0L));

            MemorySegment escapedPool = result.withStringPoolSegment(segment -> segment.asSlice(0L, 4L));
            assertFalse(escapedPool.scope().isAlive());
            assertThrows(
                    IllegalStateException.class,
                    () -> escapedPool.get(ValueLayout.JAVA_BYTE, 0L));

            MemorySegment escapedGraph = result.withGraphSegment(segment -> segment.asSlice(0L, 4L));
            assertFalse(escapedGraph.scope().isAlive());
            assertThrows(
                    IllegalStateException.class,
                    () -> escapedGraph.get(ValueLayout.JAVA_BYTE, 0L));

            assertFalse(result.isClosed());
            assertEquals("Lifetime views stay typed.", result.span(0).coveredSourceText(TEXT));
        }
    }

    @Test
    void reentrantCloseInsideSegmentCallbackThrowsInsteadOfDeadlocking() {
        TurboAnalysisResult result;
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            result = analyzer.analyzeGraph(TEXT);
        }
        try {
            assertThrows(
                    IllegalStateException.class,
                    () -> result.withSpansSegment(segment -> {
                        result.close();
                        return segment.byteSize();
                    }));
            assertThrows(
                    IllegalStateException.class,
                    () -> result.withStringPoolSegment(segment -> {
                        result.close();
                        return segment.byteSize();
                    }));
            assertThrows(
                    IllegalStateException.class,
                    () -> result.withGraphSegment(segment -> {
                        result.close();
                        return segment.byteSize();
                    }));
            assertFalse(result.isClosed());
            assertEquals("Lifetime views stay typed.", result.span(0).coveredSourceText(TEXT));
        } finally {
            result.close();
        }
        assertTrue(result.isClosed());
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

        try (TurboAnalysisResult result = retained.tokenize(TEXT)) {
            assertEquals("Lifetime", result.span(0).coveredSourceText(TEXT));
        }
        retained.close();
        retained.close();
        assertThrows(IllegalStateException.class, () -> retained.tokenize(TEXT));
    }

    @Test
    void graphSegmentCallbackIsBorrowedAndCopyOutIsExplicit() {
        TurboAnalysisResult result;
        byte[] copied;
        try (TurboAnalyzer analyzer = TurboAnalyzer.create()) {
            result = analyzer.analyzeGraph(TEXT);
            long segmentLength = result.withGraphSegment(segment -> {
                assertTrue(segment.isReadOnly());
                assertThrows(
                        IllegalArgumentException.class,
                        () -> segment.set(ValueLayout.JAVA_BYTE, 0L, (byte) 'X'));
                assertEquals((byte) 'T', segment.get(ValueLayout.JAVA_BYTE, 0L));
                assertEquals((byte) 'G', segment.get(ValueLayout.JAVA_BYTE, 1L));
                assertEquals((byte) 'R', segment.get(ValueLayout.JAVA_BYTE, 2L));
                assertEquals((byte) 'F', segment.get(ValueLayout.JAVA_BYTE, 3L));
                return segment.byteSize();
            });
            assertEquals(result.graphByteLength(), segmentLength);
            long spansBytes = result.withSpansSegment(segment -> {
                assertTrue(segment.isReadOnly());
                return segment.byteSize();
            });
            assertEquals(result.spanCount() * TurboLibrary.TURBO_SPAN_LAYOUT.byteSize(), spansBytes);
            long poolBytes = result.withStringPoolSegment(segment -> {
                assertTrue(segment.isReadOnly());
                return segment.byteSize();
            });
            assertTrue(poolBytes > 0);
            copied = result.copyGraphBytes();
        }

        result.close();
        assertThrows(IllegalStateException.class, result::copyGraphBytes);
        assertThrows(IllegalStateException.class, () -> result.withGraphSegment(segment -> segment.byteSize()));
        assertEquals(segmentLength(copied), copied.length);
        assertEquals((byte) 'T', copied[0]);
    }

    @Test
    void spanIndexesAreChecked() {
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult result = analyzer.tokenize("One token.")) {
            assertThrows(IndexOutOfBoundsException.class, () -> result.span(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> result.span(result.spanCount()));
            assertThrows(IndexOutOfBoundsException.class, () -> result.copySpan(result.spanCount()));
        }
    }

    @Test
    void analyzerAndResultReadsAreSafeUnderConcurrentUse() throws Exception {
        String text = "Concurrent readers keep exact source offsets 🚀.";
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult shared = analyzer.tokenize(text);
                ExecutorService executor = Executors.newFixedThreadPool(8)) {
            List<String> expected = shared.spans().stream()
                    .map(span -> span.coveredSourceText(text))
                    .toList();

            List<Future<List<String>>> futures = new java.util.ArrayList<>();
            for (int i = 0; i < 32; i++) {
                futures.add(executor.submit(() -> {
                    List<String> fromShared = shared.spans().stream()
                            .map(span -> span.coveredSourceText(text))
                            .toList();
                    try (TurboAnalysisResult own = analyzer.tokenize(text)) {
                        List<String> fromOwn = own.spans().stream()
                                .map(span -> span.coveredSourceText(text))
                                .toList();
                        assertEquals(fromShared, fromOwn);
                        return fromOwn;
                    }
                }));
            }
            for (Future<List<String>> future : futures) {
                assertEquals(expected, future.get());
            }
        }
    }

    private static long segmentLength(byte[] bytes) {
        return bytes.length;
    }
}
