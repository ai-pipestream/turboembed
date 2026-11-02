package ai.pipestream.turboembed.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The session behaviour every binding shares over the sealed small BERT bundle: the shape of
 * the vectors, their determinism, and the native refusal of a task the bundle does not offer.
 *
 * <p>A binding runs the suite by extending this class and returning itself from
 * {@link #binding()}.
 */
public abstract class SessionConformance {
    /** Native {@code TURBO_E_UNSUPPORTED_TASK}. */
    protected static final int UNSUPPORTED_TASK = 514;

    /** The binding under test. */
    protected abstract Binding binding();

    @Test
    public void embedsTheSmallBertBundleAndRefusesRerankWithTheNativeStatus() {
        try (Session session = binding().openSession(Testdata.tinyBertBundle())) {
            float[][] vectors = session.embed(List.of("hello world", "foreign function memory"));
            assertEquals(2, vectors.length);
            assertTrue(vectors[0].length > 0);
            assertEquals(vectors[0].length, vectors[1].length);
            for (float[] vector : vectors) {
                for (float value : vector) {
                    assertTrue(Float.isFinite(value));
                }
            }

            RuntimeException failure =
                    assertThrows(RuntimeException.class, () -> session.rerank("query", List.of("passage one")));
            assertEquals(UNSUPPORTED_TASK, binding().statusCode(failure), binding().name() + ": " + failure);
            assertEquals("TURBO_E_UNSUPPORTED_TASK", binding().statusName(failure));
        }
    }

    @Test
    public void theSameTextsEmbedToTheSameVectors() {
        List<String> texts = List.of("same input", "Café 🚀 東京");
        try (Session session = binding().openSession(Testdata.tinyBertBundle())) {
            float[][] first = session.embed(texts);
            float[][] second = session.embed(texts);
            assertEquals(first.length, second.length);
            for (int i = 0; i < first.length; i++) {
                assertArrayEquals(first[i], second[i], 0.0f);
            }
            float[][] single = session.embed(List.of(texts.get(1)));
            assertEquals(1, single.length);
            assertEquals(first[1].length, single[0].length);
        }
    }
}
