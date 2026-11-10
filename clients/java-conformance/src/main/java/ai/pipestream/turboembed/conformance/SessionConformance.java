package ai.pipestream.turboembed.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.TurboEmbedOptions;
import ai.pipestream.turboembed.TurboException;
import ai.pipestream.turboembed.TurboModelInfo;
import ai.pipestream.turboembed.TurboRerankOptions;
import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.TurboSession;
import ai.pipestream.turboembed.TurboSessionOptions;
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
    /** Native {@code TURBO_E_UNSUPPORTED_OPTION}. */
    protected static final int UNSUPPORTED_OPTION = 513;
    /** Native {@code TURBO_E_CAPACITY}. */
    protected static final int CAPACITY = 771;
    /** Native {@code TURBO_E_INVALID_ARGUMENT}. */
    protected static final int INVALID_ARGUMENT = 256;

    /** The binding under test. */
    protected abstract TurboRuntime binding();

    @Test
    public void embedsTheSmallBertBundleAndRefusesRerankWithTheNativeStatus() {
        try (TurboSession session = binding().openSession(Testdata.tinyBertBundle())) {
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
            assertEquals(UNSUPPORTED_TASK, org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).code(), binding().kind() + ": " + failure);
            assertEquals("TURBO_E_UNSUPPORTED_TASK", org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).statusName());
        }
    }

    @Test
    public void theSameTextsEmbedToTheSameVectors() {
        List<String> texts = List.of("same input", "Café 🚀 東京");
        try (TurboSession session = binding().openSession(Testdata.tinyBertBundle())) {
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

    private static int code(Runnable call) {
        RuntimeException failure = assertThrows(RuntimeException.class, call::run);
        return org.junit.jupiter.api.Assertions.assertInstanceOf(TurboException.class, failure).code();
    }

    private static void assertSameRows(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int i = 0; i < want.length; i++) {
            assertArrayEquals(want[i], got[i], 0.0f, "row " + i);
        }
    }

    @Test
    public void theModelInfoIsTheBundles() {
        try (TurboSession session = binding().openSession(Testdata.tinyBertBundle())) {
            TurboModelInfo info = session.modelInfo();
            assertEquals(0, info.device());
            assertEquals(TurboSessionOptions.Task.EMBED, info.task());
            assertEquals(32, info.dim());
            assertEquals(TurboEmbedOptions.Pooling.MEAN, info.pooling());
            assertEquals(TurboEmbedOptions.Normalize.L2, info.normalize());
            assertEquals(64, info.maxSeq());
            assertEquals(64, info.maxBatch());
            assertEquals("turboembed/tiny-bert", info.modelId());
            assertEquals("query: ", info.prefixQuery());
            assertEquals("", info.prefixDocument());
            assertEquals(List.of(), info.outputDims());
            assertEquals(64, info.manifestSha256().length());
            assertEquals(0, info.rerankPairTemplate());
        }
    }

    @Test
    public void embedOptionsReachTheNativeWrite() {
        List<String> texts = List.of("hello world", "Café 🚀 東京");
        try (TurboSession session = binding().openSession(Testdata.tinyBertBundle())) {
            float[][] plain = session.embed(texts);
            // The query role puts the bundle's prefix before each text; this bundle has no
            // document prefix.
            assertSameRows(
                    session.embed(List.of("query: hello world", "query: Café 🚀 東京")),
                    session.embed(texts, TurboEmbedOptions.query()));
            assertSameRows(plain, session.embed(texts, TurboEmbedOptions.document()));
            // The model's dim is an output dim; any other width this bundle does not list.
            assertSameRows(plain, session.embed(texts, TurboEmbedOptions.DEFAULT.withOutputDim(32)));
            assertEquals(UNSUPPORTED_OPTION, code(() -> session.embed(texts, TurboEmbedOptions.DEFAULT.withOutputDim(16))));
            // Unnormalized vectors are the normalized ones scaled.
            float[][] raw = session.embed(texts, TurboEmbedOptions.DEFAULT.withNormalize(TurboEmbedOptions.Normalize.NONE));
            for (int i = 0; i < raw.length; i++) {
                double norm = 0;
                for (float v : raw[i]) {
                    norm += (double) v * v;
                }
                norm = Math.sqrt(norm);
                for (int j = 0; j < raw[i].length; j++) {
                    assertEquals(plain[i][j], raw[i][j] / norm, 1e-5, "row " + i + " value " + j);
                }
            }
            // Truncation to two tokens with no cutting allowed is over capacity.
            TurboEmbedOptions tight = TurboEmbedOptions.DEFAULT.withTruncation(TurboEmbedOptions.Truncate.NONE, 3);
            assertEquals(CAPACITY, code(() -> session.embed(List.of("one two three four five"), tight)));
            float[][] cut = session.embed(
                    List.of("one two three four five"),
                    TurboEmbedOptions.DEFAULT.withTruncation(TurboEmbedOptions.Truncate.RIGHT, 3));
            assertEquals(32, cut[0].length);
        }
    }

    @Test
    public void binaryCodesAreTheSignBitsOfTheVectors() {
        List<String> texts = List.of("hello world", "foreign function memory");
        try (TurboSession session = binding().openSession(Testdata.tinyBertBundle())) {
            float[][] vectors = session.embed(texts);
            byte[][] codes = session.embedCodes(texts, TurboEmbedOptions.DEFAULT.withBinary());
            assertEquals(texts.size(), codes.length);
            for (int i = 0; i < codes.length; i++) {
                assertEquals(4, codes[i].length);
                for (int j = 0; j < 32; j++) {
                    boolean bit = (codes[i][j / 8] & (0x80 >>> (j % 8))) != 0;
                    assertEquals(vectors[i][j] > 0, bit, "row " + i + " dimension " + j);
                }
            }
            // Codes and vectors are read through their own calls.
            assertThrows(IllegalArgumentException.class, () -> session.embed(texts, TurboEmbedOptions.DEFAULT.withBinary()));
            assertThrows(IllegalArgumentException.class, () -> session.embedCodes(texts, TurboEmbedOptions.DEFAULT));
            // Int8 codes over each dimension's range [-1, 1]: a value's code is its step.
            float[] ranges = new float[64];
            java.util.Arrays.fill(ranges, 0, 32, -1.0f);
            java.util.Arrays.fill(ranges, 32, 64, 1.0f);
            byte[][] int8 = session.embedCodes(texts, TurboEmbedOptions.DEFAULT.withInt8(ranges));
            float step = 2.0f / 255.0f;
            for (int i = 0; i < int8.length; i++) {
                assertEquals(32, int8[i].length);
                for (int j = 0; j < 32; j++) {
                    int want = Math.max(0, Math.min(255, (int) Math.floor((vectors[i][j] + 1.0f) / step))) - 128;
                    assertEquals(want, int8[i][j], 1, "row " + i + " dimension " + j);
                }
            }
            // Ranges that are not two a dimension are refused before the native write reads them.
            for (float[] wrong : new float[][] {new float[2], new float[63], new float[66]}) {
                assertEquals(INVALID_ARGUMENT,
                        code(() -> session.embedCodes(texts, TurboEmbedOptions.DEFAULT.withInt8(wrong))));
            }
        }
    }

    @Test
    public void sessionOptionsChooseTheDeviceAndShape() {
        java.nio.file.Path bundle = Testdata.tinyBertBundle();
        List<String> texts = List.of("hello world", "foreign function memory");
        float[][] plain;
        try (TurboSession session = binding().openSession(bundle)) {
            plain = session.embed(texts);
        }
        // EXACT is F32 throughout, which this F32 bundle computes in already.
        try (TurboSession exact = binding().openSession(bundle,
                TurboSessionOptions.DEFAULT.withPrecision(TurboSessionOptions.Precision.EXACT).withCpuThreads(1))) {
            assertSameRows(plain, exact.embed(texts));
        }
        // A session of one row refuses two texts.
        try (TurboSession one = binding().openSession(bundle, TurboSessionOptions.DEFAULT.withShape(1, 0))) {
            assertEquals(1, one.embed(List.of(texts.get(0))).length);
            assertEquals(CAPACITY, code(() -> one.embed(texts)));
        }
        // Reranking runs on the CPU wherever no GPU offers it, so selecting for it always opens.
        try (TurboSession picked = binding().openSession(bundle,
                TurboSessionOptions.fastestFor(TurboSessionOptions.Task.RERANK))) {
            assertTrue(picked.modelInfo().device() >= 0);
            assertEquals(32, picked.embed(texts)[0].length);
        }
        // A device index past the runtime's devices is the C ABI's invalid argument.
        assertEquals(INVALID_ARGUMENT, code(() -> binding().openSession(bundle, TurboSessionOptions.onDevice(4096))));
        // Rerank options reach the native write: the embedding bundle still refuses the task.
        try (TurboSession session = binding().openSession(bundle)) {
            assertEquals(UNSUPPORTED_TASK,
                    code(() -> session.rerank("query", List.of("passage"), new TurboRerankOptions(4, 16, 0))));
        }
    }
}
