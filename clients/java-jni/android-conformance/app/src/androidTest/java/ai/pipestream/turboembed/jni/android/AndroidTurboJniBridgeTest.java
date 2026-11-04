package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.TurboException;
import ai.pipestream.turboembed.conformance.Testdata;
import ai.pipestream.turboembed.jni.TurboAnalysisResult;
import ai.pipestream.turboembed.jni.TurboAnalyzer;
import ai.pipestream.turboembed.jni.TurboLibrary;
import ai.pipestream.turboembed.jni.TurboSession;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;

/**
 * Exercises the {@code ai.pipestream.turboembed.jni} package-level classes on ART that require a
 * real bundle path (session lifecycle, concurrent readers, and {@code Testdata} resolution through
 * {@code turbo.repo.root}).
 */
@RunWith(AndroidJupiterRunner.class)
public final class AndroidTurboJniBridgeTest {
    static {
        AndroidConformanceSetup.ensureReady();
    }

    @Test
    public void sessionCloseIsIdempotentAndCallsAfterCloseFailBeforeTheBridge() {
        TurboSession session = TurboSession.open(Testdata.tinyBertBundle());
        session.close();
        session.close();
        assertTrue(session.isClosed());
        assertThrows(IllegalStateException.class, () -> session.embed(List.of("after close")));
        assertThrows(IllegalStateException.class, () -> session.rerank("query", List.of("passage")));
        try (TurboSession open = TurboSession.open(Testdata.tinyBertBundle())) {
            assertThrows(NullPointerException.class, () -> open.embed(null));
            assertThrows(NullPointerException.class, () -> open.embed(Arrays.asList("ok", null)));
            TurboException ex = assertThrows(TurboException.class, () -> open.embed(List.of()));
            assertEquals(TurboLibrary.TURBO_E_INVALID_ARGUMENT, ex.code());
        }
        TurboException missing = assertThrows(
                TurboException.class, () -> TurboSession.open(Testdata.missingBundle()));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, missing.code());
    }

    @Test
    public void analyzerResultAndSessionReadsAreSafeUnderConcurrentUse() throws Exception {
        String text = "Concurrent readers keep exact source offsets 🚀.";
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try (TurboAnalyzer analyzer = TurboAnalyzer.create();
                TurboAnalysisResult shared = analyzer.tokenize(text);
                TurboSession session = TurboSession.open(Testdata.tinyBertBundle())) {
            List<String> expected = surfaces(shared, text);
            float[][] vector = session.embed(List.of(text));

            List<Future<List<String>>> futures = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
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

    @Test
    public void turboPathsFindsTheExtractedAssetBundlesThroughRepoRootProperty() {
        assertTrue(Files.isDirectory(Testdata.tinyBertBundle()));
        assertTrue(Files.isDirectory(Testdata.headsBundle()));
        assertTrue(Files.isRegularFile(Testdata.tinyBertBundle().resolve("manifest.json")));
        assertTrue(Files.isRegularFile(Testdata.headsBundle().resolve("manifest.json")));
    }

    private static List<String> surfaces(TurboAnalysisResult result, String text) {
        return result.spans().stream().map(span -> span.coveredSourceText(text)).collect(Collectors.toList());
    }
}
