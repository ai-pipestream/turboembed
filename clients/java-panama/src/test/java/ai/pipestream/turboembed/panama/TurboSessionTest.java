package ai.pipestream.turboembed.panama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

/** Embedding session lifecycle and native-shape validation tests. */
final class TurboSessionTest {

    @Test
    void embedsTinyBertAndRejectsRerankWithNativeStatus() {
        try (TurboSession session = TurboSession.open(TestPaths.tinyBertBundle())) {
            float[][] vectors = session.embed(List.of("hello world", "panama foreign function memory"));
            assertEquals(2, vectors.length);
            assertTrue(vectors[0].length > 0);
            assertEquals(vectors[0].length, vectors[1].length);
            assertFalse(session.isClosed());

            TurboException unsupported = assertThrows(
                    TurboException.class, () -> session.rerank("query", List.of("passage one")));
            assertEquals(TurboLibrary.TURBO_E_UNSUPPORTED_TASK, unsupported.code());
            assertEquals("TURBO_E_UNSUPPORTED_TASK", unsupported.statusName());
        }
    }

    @Test
    void closeIsIdempotentAndCallsAfterCloseFailBeforeNativeUse() {
        TurboSession session = TurboSession.open(TestPaths.tinyBertBundle());
        session.close();
        session.close();
        assertTrue(session.isClosed());
        assertThrows(IllegalStateException.class, () -> session.embed(List.of("after close")));
        assertThrows(IllegalStateException.class, () -> session.rerank("query", List.of("passage")));
    }

    @Test
    void nullElementsAreRejectedAndConcurrentEmbedsAreSerialized() throws Exception {
        try (TurboSession session = TurboSession.open(TestPaths.tinyBertBundle());
                ExecutorService executor = Executors.newFixedThreadPool(4)) {
            assertThrows(NullPointerException.class, () -> session.embed(null));
            ArrayList<String> withNull = new ArrayList<>();
            withNull.add("ok");
            withNull.add(null);
            assertThrows(NullPointerException.class, () -> session.embed(withNull));

            float[][] expected = session.embed(List.of("same input"));
            List<Future<float[][]>> futures = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                futures.add(executor.submit(() -> session.embed(List.of("same input"))));
            }
            for (Future<float[][]> future : futures) {
                float[][] actual = future.get();
                assertEquals(expected.length, actual.length);
                assertEquals(expected[0].length, actual[0].length);
                for (int i = 0; i < expected[0].length; i++) {
                    assertEquals(expected[0][i], actual[0][i], 0.0f);
                }
            }
        }
    }
}
