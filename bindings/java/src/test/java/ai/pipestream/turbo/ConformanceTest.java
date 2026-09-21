package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * The conformance cases that define the binding, run through the Java API
 * against the mock provider (the same cases the Rust and C suites carry).
 * Needs {@code turbo.library} (libturbo) and {@code turbo.bundles} (the
 * mock bundle root); the Maven build sets both.
 */
class ConformanceTest {
    static final Path BUNDLES = Path.of(System.getProperty("turbo.bundles"));

    static String bundle(String kind) {
        return BUNDLES.resolve(kind).toString();
    }

    /** The mock accelerator: AUTO never picks the mock CPU device. */
    static int mockDevice(Turbo rt) {
        int idx = rt.selectDevice();
        assertNotEquals(DeviceKind.CPU, rt.device(idx).kind(), "AUTO must never select a CPU");
        assertEquals("mock", rt.device(idx).providerId());
        return idx;
    }

    @Test
    void abiVersionMatchesTheHeader() {
        assertEquals(TURBO_ABI_VERSION(), Turbo.abiVersion());
    }

    @Test
    void devicesAreEnumeratedAndAutoNeverSelectsCpu() {
        try (Turbo rt = Turbo.create()) {
            List<DeviceInfo> devices = rt.devices();
            assertFalse(devices.isEmpty());
            assertTrue(devices.stream().anyMatch(d -> d.kind() == DeviceKind.CPU), "the mock offers a CPU device");
            assertTrue(devices.stream().anyMatch(d -> d.kind() != DeviceKind.CPU), "and an accelerator");
            mockDevice(rt);
            TurboException e = assertThrows(TurboException.class, () -> rt.selectDevice(SelectPolicy.EXPLICIT, "nonexistent", 0));
            assertEquals(TURBO_E_DEVICE_NOT_FOUND(), e.code(), e.getMessage());
        }
    }

    @Test
    void capabilityCellsAreHonest() {
        try (Turbo rt = Turbo.create()) {
            int idx = mockDevice(rt);
            Capability embed = rt.capability(idx, Task.EMBED, Modality.TEXT);
            assertEquals(CapStatus.SUPPORTED, embed.status());
            assertTrue(embed.offered());
            Capability audio = rt.capability(idx, Task.EMBED, Modality.AUDIO);
            assertEquals(CapStatus.UNSUPPORTED, audio.status());
            assertFalse(audio.offered());
        }
    }

    @Test
    void embeddingIsDeterministicAndUnitNorm() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("embedding"))) {
            ModelInfo info = model.info();
            assertEquals(ModelKind.EMBEDDING, info.kind());
            assertEquals("mock", info.providerId());
            assertTrue(info.dim() > 0);
            try (Session s = model.createSession(4, 0)) {
                s.writeText(List.of("hello world", "hello world", "something else"), EmbedOptions.defaults());
                float[] a;
                try (Result r = s.run()) {
                    assertEquals(1, r.outputCount());
                    Result.Output out = r.output(0);
                    assertEquals("embeddings", out.name());
                    assertArrayEquals(new long[] {3, info.dim()}, out.shape());
                    assertEquals(Placement.HOST, r.placement());
                    a = r.readFloats(0);
                }
                assertEquals(3 * info.dim(), a.length);
                for (int row = 0; row < 3; row++) {
                    double norm = 0;
                    for (int i = 0; i < info.dim(); i++) {
                        norm += a[row * info.dim() + i] * a[row * info.dim() + i];
                    }
                    assertEquals(1.0, Math.sqrt(norm), 1e-4, "row " + row + " is unit norm");
                }
                for (int i = 0; i < info.dim(); i++) {
                    assertEquals(a[i], a[info.dim() + i], "identical texts embed identically");
                }
                assertTrue(s.stats().runs() == 1);
            }
        }
    }

    @Test
    void unsupportedOptionsNameTheirField() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("embedding"));
                Session s = model.createSession(2, 16)) {
            DeviceInfo dev = rt.device(ctx.deviceIndex());
            assertFalse(dev.has(TURBO_CAP_OPT_POOLING_OVERRIDE()), "the mock does not override pooling");
            TurboException e = assertThrows(
                    TurboException.class, () -> s.writeText(List.of("x"), EmbedOptions.defaults().withPooling(Pooling.CLS)));
            assertEquals(TURBO_E_UNSUPPORTED_OPTION(), e.code(), e.getMessage());
            assertEquals(6, e.field(), "pooling is field 6");
            assertEquals("TURBO_E_UNSUPPORTED_OPTION", e.statusName());
            // A budget the session cannot hold is refused, never clamped.
            TurboException cap = assertThrows(
                    TurboException.class, () -> s.writeText(List.of("x"), EmbedOptions.defaults().withMaxTokens(64)));
            assertEquals(TURBO_E_CAPACITY(), cap.code(), cap.getMessage());
            assertEquals(3, cap.field());
        }
    }

    @Test
    void rerankReturnsScoresAndSortedOrder() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("reranker"));
                Session s = model.createSession(4, 0)) {
            s.writePairs("what is turbo", List.of("turbo is a library", "unrelated text", "what is turbo"), RerankOptions.defaults().withReturnSorted(true));
            try (Result r = s.run()) {
                assertEquals(2, r.outputCount());
                float[] scores = r.readFloats(0);
                int[] sorted = r.readInts(1);
                assertEquals(3, scores.length);
                assertEquals(3, sorted.length);
                for (int i = 1; i < sorted.length; i++) {
                    assertTrue(scores[sorted[i - 1]] >= scores[sorted[i]], "sorted is descending by score");
                }
                for (float f : scores) {
                    assertTrue(f >= 0 && f <= 1, "sigmoid scores");
                }
            }
        }
    }

    @Test
    void tokenClassificationYieldsSpansInsideTheText() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("token-classifier"));
                Session s = model.createSession(2, 0)) {
            String text = "Ada visited Berlin";
            s.writeTextClassify(List.of(text), ClassifyOptions.defaults());
            try (Result r = s.run()) {
                List<String> labels = model.info().labels();
                assertEquals("O", labels.get(0));
                for (Span span : r.spans()) {
                    assertEquals(0, span.row());
                    assertTrue(span.byteStart() < span.byteEnd() && span.byteEnd() <= text.length(), span.toString());
                    assertTrue(span.label() > 0 && span.label() < labels.size(), span.toString());
                    assertTrue(span.score() > 0 && span.score() <= 1, span.toString());
                }
            }
        }
    }

    @Test
    void aHeldResultBlocksTheSessionAndReleasingAParentKeepsChildrenAlive() {
        Turbo rt = Turbo.create();
        Context ctx = rt.createContext(mockDevice(rt));
        Model model = ctx.loadModel(bundle("embedding"));
        Session s = model.createSession(2, 0);
        // Parents released first: the session keeps them alive.
        rt.close();
        ctx.close();
        model.close();
        s.writeText(List.of("still works"), EmbedOptions.defaults());
        Result r = s.run();
        TurboException e = assertThrows(TurboException.class, () -> s.writeText(List.of("again"), EmbedOptions.defaults()));
        assertEquals(TURBO_E_BUSY(), e.code(), "a live result leases the session");
        r.close();
        s.writeText(List.of("again"), EmbedOptions.defaults());
        try (Result r2 = s.run()) {
            assertEquals(1, r2.outputCount());
        }
        s.close();
    }

    @Test
    void concurrentUseOfOneSessionIsBusyNeverWrong() throws Exception {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("embedding"));
                Session s = model.createSession(8, 0)) {
            int threads = 4;
            CountDownLatch start = new CountDownLatch(1);
            AtomicInteger ok = new AtomicInteger();
            AtomicInteger busy = new AtomicInteger();
            AtomicInteger other = new AtomicInteger();
            Thread[] ts = new Thread[threads];
            for (int t = 0; t < threads; t++) {
                ts[t] = new Thread(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < 50; i++) {
                            try {
                                s.writeText(List.of("thread text"), EmbedOptions.defaults());
                                try (Result r = s.run()) {
                                    assertEquals(model.info().dim(), r.readFloats(0).length);
                                }
                                ok.incrementAndGet();
                            } catch (TurboException e) {
                                if (e.code() == TURBO_E_BUSY()) {
                                    busy.incrementAndGet();
                                } else {
                                    other.incrementAndGet();
                                }
                            }
                        }
                    } catch (InterruptedException ignored) {
                    }
                });
                ts[t].start();
            }
            start.countDown();
            for (Thread t : ts) {
                t.join();
            }
            assertEquals(0, other.get(), "only TURBO_E_BUSY is acceptable under contention");
            assertTrue(ok.get() > 0, "some runs complete");
        }
    }
}
