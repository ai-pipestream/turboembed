// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import ai.pipestream.turboembed.*;
import java.nio.*;
import java.nio.file.Path;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import static org.junit.jupiter.api.Assertions.*;

/** Explicit hardware suite: both Intel GPU and CPU must be available. */
@EnabledIfEnvironmentVariable(named = "TURBOEMBED_PREPARED_SDK", matches = ".+")
class NativeContractTest {
    private static final Path SDK = Path.of(System.getenv().getOrDefault("TURBOEMBED_PREPARED_SDK", "."));
    private static Path bundle() {
        String value = System.getenv("TURBOEMBED_PREPARED_BUNDLE");
        assertNotNull(value, "hardware tests require TURBOEMBED_PREPARED_BUNDLE");
        return Path.of(value);
    }
    private static float[] read(ExecutionSlot slot, String... text) {
        slot.writeText(text);
        try (EmbeddingResult result = slot.execute()) {
            float[] output = new float[result.batch() * result.dimension()];
            result.readInto(FloatBuffer.wrap(output)); return output;
        }
    }
    private static void near(float[] expected, float[] actual, double max, double rmse) {
        assertEquals(expected.length, actual.length);
        double squared = 0;
        for (int i = 0; i < expected.length; i++) {
            assertTrue(Float.isFinite(actual[i]), "nonfinite value at " + i);
            double difference = Math.abs((double) expected[i] - actual[i]);
            assertTrue(difference <= max, "difference " + difference + " at " + i);
            squared += difference * difference;
        }
        assertTrue(Math.sqrt(squared / expected.length) <= rmse);
    }
    private static void helloTokens(ExecutionSlot slot) {
        TokenBuffers input = slot.inputs();
        for (IntBuffer buffer : new IntBuffer[]{input.ids(), input.attentionMask(), input.tokenTypes()}) {
            buffer.clear(); for (int i = 0; i < buffer.capacity(); i++) { buffer.put(i, 0); }
        }
        int[] tokens = {101, 7592, 2088, 102};
        for (int i = 0; i < tokens.length; i++) {
            input.ids().put(i, tokens[i]); input.attentionMask().put(i, 1);
        }
        // Upload covers fixed storage even when the caller has consumed its view.
        input.ids().position(input.ids().capacity()); input.attentionMask().limit(0);
        slot.upload();
    }
    @Test void textPreparedAndCpuGpuParity() {
        try (var provider = FfmTurboEmbed.open(SDK);
             var gpu = provider.context(Device.AUTO, 0);
             var cpu = provider.context(Device.OPENVINO_CPU, 0);
             var gm = gpu.loadModel(bundle()); var cm = cpu.loadModel(bundle());
             var gs = gm.slot(1, 32); var cs = cm.slot(1, 32)) {
            assertEquals(1, gpu.info().device()); assertEquals(2, cpu.info().device());
            assertFalse(gpu.info().name().isBlank()); assertFalse(gpu.info().runtimeVersion().isBlank());
            assertEquals(384, gm.info().dimension()); assertEquals(30522, gm.info().vocabularySize());
            assertEquals("sentence-transformers/all-MiniLM-L6-v2", gm.info().modelId());
            assertEquals("1110a243fdf4706b3f48f1d95db1a4f5529b4d41", gm.info().revision());
            assertEquals("be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037", gm.info().tokenizerSha256());
            assertTrue(gm.info().normalized());
            for (String text : new String[]{"hello world", "", "a\0b", "café 東京 🙂"}) {
                near(read(cs, text), read(gs, text), 5e-4, 1e-4);
            }
            float[] text = read(gs, "hello world"); helloTokens(gs);
            SlotStats before = gs.stats();
            try (var result = gs.execute()) {
                float[] prepared = new float[384]; result.readInto(FloatBuffer.wrap(prepared));
                near(text, prepared, 1e-6, 1e-7);
                assertEquals(1, result.batch()); assertEquals(384, result.dimension());
            }
            SlotStats after = gs.stats();
            assertEquals(before.executions() + 1, after.executions());
            assertEquals(before.inputWriteBytes(), after.inputWriteBytes());
            assertEquals(before.outputReadBytes() + 384 * 4, after.outputReadBytes());
            assertEquals(32 * 3 * 4, after.ownedInputBytes()); assertEquals(384 * 4, after.ownedOutputBytes());
            assertThrows(IllegalArgumentException.class, () -> gm.slot(0, 32));
            assertThrows(IllegalArgumentException.class, () -> gm.slot(1, 257));
            NativeException missing = assertThrows(NativeException.class,
                () -> provider.context(Device.OPENVINO_GPU, Integer.MAX_VALUE));
            assertEquals(4, missing.code());
        }
    }
    @Test void outputsRespectPositionsCapacityOrderAndAliases() {
        try (var provider = FfmTurboEmbed.open(SDK); var ctx = provider.context(Device.OPENVINO_GPU, 0);
             var model = ctx.loadModel(bundle()); var slot = model.slot(1, 32)) {
            float[] expected = read(slot, "hello world"); slot.writeText("hello world");
            try (var result = slot.execute()) {
                assertThrows(IllegalArgumentException.class, () -> result.readInto(FloatBuffer.allocate(383)));
                assertThrows(ReadOnlyBufferException.class, () -> result.readInto(FloatBuffer.allocate(384).asReadOnlyBuffer()));
                for (int kind = 0; kind < 4; kind++) {
                    FloatBuffer out = kind == 0 ? FloatBuffer.allocate(386) : ByteBuffer.allocateDirect(386 * 4)
                        .order(kind == 1 ? ByteOrder.nativeOrder() :
                            (ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN))
                        .asFloatBuffer();
                    if (kind == 3) {
                        out = ByteBuffer.allocateDirect(386 * 4 + 1).position(1).slice()
                            .order(ByteOrder.nativeOrder()).asFloatBuffer();
                    }
                    out.put(0, 42); out.put(385, 43); out.position(1); out.limit(385);
                    result.readInto(out); assertEquals(385, out.position());
                    FloatBuffer alias = out.duplicate(); alias.position(1);
                    float[] actual = new float[384]; alias.get(actual); near(expected, actual, 1e-6, 1e-7);
                    out.clear(); assertEquals(42, out.get(0)); assertEquals(43, out.get(385));
                }
            }
        }
    }
    @SuppressWarnings("try") // Explicit early and repeated close calls exercise resource ownership.
    @Test void leasesThreadOwnershipAndParentClose() throws Exception {
        try (var provider = FfmTurboEmbed.open(SDK); var ctx = provider.context(Device.OPENVINO_GPU, 0);
             var model = ctx.loadModel(bundle()); var slot = model.slot(1, 32)) {
            IntBuffer retained = slot.inputs().ids();
            model.close(); ctx.close(); provider.close();
            assertThrows(IllegalStateException.class, model::info);
            assertThrows(IllegalStateException.class, ctx::info);
            assertThrows(IllegalStateException.class, () -> provider.context(Device.AUTO, 0));
            slot.writeText("hello world");
            EmbeddingResult old = slot.execute(); OpenClView view = old.openCl();
            try (old) {
                assertNotEquals(0, view.context()); assertNotEquals(0, view.queue()); assertNotEquals(0, view.buffer());
                assertEquals(1536, view.byteSize());
                assertThrows(IllegalStateException.class, slot::close);
                assertThrows(IllegalStateException.class, slot::execute);
                assertThrows(IllegalStateException.class, slot::upload);
                try (var executor = Executors.newSingleThreadExecutor()) {
                    executor.submit(() -> {
                        assertThrows(IllegalStateException.class, slot::stats);
                        assertThrows(IllegalStateException.class, old::close);
                        assertThrows(IllegalStateException.class, view::buffer);
                        assertThrows(java.lang.WrongThreadException.class, () -> retained.get(0));
                    }).get(30, TimeUnit.SECONDS);
                }
            }
            try (var fresh = slot.execute()) {
                assertThrows(IllegalStateException.class, old::dimension);
                assertThrows(IllegalStateException.class, view::buffer);
                old.close(); assertEquals(384, fresh.dimension());
            }
            slot.close(); assertThrows(IllegalStateException.class, () -> retained.get(0));
        }
    }
    @Test void failedWritesInvalidatePreviousInputs() {
        try (var provider = FfmTurboEmbed.open(SDK); var ctx = provider.context(Device.OPENVINO_CPU, 0);
             var model = ctx.loadModel(bundle()); var slot = model.slot(1, 32)) {
            slot.writeText("valid");
            assertThrows(IllegalArgumentException.class, () -> slot.writeText("\uD800"));
            assertEquals(1, assertThrows(NativeException.class, slot::execute).code());
            slot.writeText("valid"); assertThrows(NullPointerException.class, () -> slot.writeText((String) null));
            assertEquals(1, assertThrows(NativeException.class, slot::execute).code());
            slot.writeText("valid"); assertEquals(1, assertThrows(NativeException.class, () -> slot.writeText()).code());
            assertEquals(1, assertThrows(NativeException.class, slot::execute).code());
            helloTokens(slot); slot.inputs().ids().clear().put(0, 30522);
            assertEquals(1, assertThrows(NativeException.class, slot::upload).code());
            assertEquals(1, assertThrows(NativeException.class, slot::execute).code());
            slot.writeText("valid");
            try (var result = slot.execute()) {
                assertEquals(3, assertThrows(NativeException.class, result::openCl).code());
            }
        }
    }
    @Test void sharedModelCreatesIndependentConcurrentSlots() throws Exception {
        try (var provider = FfmTurboEmbed.open(SDK); var ctx = provider.context(Device.OPENVINO_GPU, 0);
             var model = ctx.loadModel(bundle()); var executor = Executors.newFixedThreadPool(2)) {
            Callable<float[]> work = () -> {
                try (var slot = model.slot(2, 32)) { return read(slot, "hello world", "café 東京 🙂"); }
            };
            Future<float[]> first = executor.submit(work), second = executor.submit(work);
            near(first.get(90, TimeUnit.SECONDS), second.get(90, TimeUnit.SECONDS), 1e-6, 1e-7);
        }
    }
    @SuppressWarnings("try") // Exercise independent provider unload and retained children.
    @Test void independentProvidersIsolateErrorsAndLibraryOwnership() {
        try (var first = FfmTurboEmbed.open(SDK); var second = FfmTurboEmbed.open(SDK);
             var c1 = first.context(Device.OPENVINO_GPU, 0); var c2 = second.context(Device.OPENVINO_GPU, 0);
             var m1 = c1.loadModel(bundle()); var m2 = c2.loadModel(bundle());
             var s1 = m1.slot(1, 32); var s2 = m2.slot(1, 32)) {
            float[] reference = read(s2, "hello world");
            first.close(); c1.close(); m1.close();
            near(reference, read(s1, "hello world"), 1e-6, 1e-7);
            assertEquals(1, assertThrows(NativeException.class, () -> s1.writeText()).code());
            near(reference, read(s2, "hello world"), 1e-6, 1e-7);
            s1.close();
            near(reference, read(s2, "hello world"), 1e-6, 1e-7);
            assertEquals(1, c2.info().device());
        }
    }

}
