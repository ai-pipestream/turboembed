package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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

    @Test
    void generationStepsUntilLengthAndCancelIsReported() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("generative"))) {
            assertEquals(ModelKind.GENERATIVE, model.info().kind());
            List<Message> prompt = List.of(Message.user("say something"));
            try (Generation g = model.createGeneration(GenerateDesc.defaults().withMaxNewTokens(6))) {
                g.prompt(prompt);
                List<Integer> tokens = new ArrayList<>();
                StringBuilder text = new StringBuilder();
                Chunk first = null;
                Chunk last = g.drain(c -> {
                    for (int id : c.tokens()) {
                        tokens.add(id);
                    }
                    text.append(c.text());
                    if (!c.done()) {
                        assertEquals(FinishReason.NONE, c.finishReason(), "an unfinished chunk names no reason");
                    }
                    return true;
                });
                assertTrue(last.done());
                assertEquals(FinishReason.LENGTH, last.finishReason());
                assertFalse(tokens.isEmpty(), "the stream produced tokens");
                assertTrue(tokens.size() <= 6, "max_new_tokens holds: " + tokens.size());
                assertFalse(text.isEmpty(), "the stream produced text");
                for (int id : tokens) {
                    assertTrue(id >= 0 && (model.info().vocabSize() == 0 || id < model.info().vocabSize()), "token in vocabulary: " + id);
                }
            }
            try (Generation g = model.createGeneration(GenerateDesc.defaults().withMaxNewTokens(50))) {
                g.prompt(prompt);
                Chunk c = g.step();
                assertTrue(c.promptTokens() > 0, "the first chunk reports the prompt size");
                assertEquals(0, c.sequence());
                assertFalse(c.done());
                g.cancel();
                Chunk end = g.step();
                assertTrue(end.done());
                assertEquals(FinishReason.CANCELLED, end.finishReason());
            }
            // A sink that stops ends with CANCELLED through the drain helper.
            try (Generation g = model.createGeneration(GenerateDesc.defaults().withMaxNewTokens(50))) {
                g.prompt(prompt);
                Chunk end = g.drain(c -> false);
                assertEquals(FinishReason.CANCELLED, end.finishReason());
            }
        }
    }

    @Test
    void generationIsRepeatableWithASeedAndRefusesUnsupportedOptionsByField() {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("generative"))) {
            List<Message> prompt = List.of(Message.user("say something"));
            List<Integer> a = new ArrayList<>();
            List<Integer> b = new ArrayList<>();
            for (List<Integer> sink : List.of(a, b)) {
                GenerateDesc d = GenerateDesc.defaults().withMaxNewTokens(8).withSampling(0.9f, 0, 0f, 0f).withSeed(42L);
                try (Generation g = model.createGeneration(d)) {
                    g.prompt(prompt);
                    g.drain(c -> {
                        for (int id : c.tokens()) {
                            sink.add(id);
                        }
                        return true;
                    });
                }
            }
            assertEquals(a, b, "one seed reproduces one token sequence");
            if (!rt.device(ctx.deviceIndex()).has(TURBO_CAP_OPT_GEN_N())) {
                GenerateDesc d = GenerateDesc.defaults().withMaxNewTokens(2);
                d = new GenerateDesc(d.maxNewTokens(), 0, 3, 0f, 0, 0f, 0f, 0f, 0f, 0f, null, List.of(), new int[0], Map.of(), 0, false);
                GenerateDesc desc = d;
                TurboException e = assertThrows(TurboException.class, () -> model.createGeneration(desc).close());
                assertEquals(TURBO_E_UNSUPPORTED_OPTION(), e.code(), e.getMessage());
                assertEquals(4, e.field(), "the rejection names n_sequences: " + e.getMessage());
            }
        }
    }

    @Test
    void tokenizerEncodesDecodesAndCounts() {
        try (Turbo rt = Turbo.create(); Tokenizer tok = rt.createTokenizer(BUNDLES.resolve("../minilm-tokenizer").toString())) {
            TokenizerInfo info = tok.info();
            assertEquals("wordpiece", info.kind());
            assertTrue(info.vocabSize() > 1000);
            assertEquals(2, info.specialsPerSequence());
            Encoding enc = tok.encode(List.of("hello world", "a longer sentence with several words"), 16, EncodeOptions.defaults().withMaxTokens(16));
            assertEquals(2, enc.rows());
            assertEquals(4, enc.lengths()[0], "[CLS] hello world [SEP]");
            assertTrue(enc.lengths()[1] > enc.lengths()[0]);
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 16; c++) {
                    assertEquals(c < enc.lengths()[r] ? 1 : 0, enc.mask()[r * 16 + c], "mask row " + r + " col " + c);
                    if (c >= enc.lengths()[r]) {
                        assertEquals(info.padId(), enc.ids()[r * 16 + c], "padding carries the pad id");
                    }
                }
            }
            assertEquals("hello world", tok.decode(enc.row(0), true));
            assertEquals(4, tok.count("hello world", true));
            assertEquals(2, tok.count("hello world", false));
            TurboException e = assertThrows(TurboException.class,
                    () -> tok.encode(List.of("one two three four five six seven eight nine ten"), 6,
                            EncodeOptions.defaults().withMaxTokens(6).withTruncate(Truncate.NONE)));
            assertEquals(TURBO_E_CAPACITY(), e.code(), e.getMessage());
        }
    }

    @Test
    void aHeldResultMakesEveryRunOnAnotherThreadBusy() throws Exception {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("embedding"));
                Session s = model.createSession(2, 0)) {
            s.writeText(List.of("held"), EmbedOptions.defaults());
            AtomicInteger busy = new AtomicInteger();
            AtomicInteger other = new AtomicInteger();
            try (Result held = s.run()) {
                Thread t = new Thread(() -> {
                    for (int i = 0; i < 20; i++) {
                        try {
                            s.writeText(List.of("contender"), EmbedOptions.defaults());
                            other.incrementAndGet();
                        } catch (TurboException e) {
                            if (e.code() == TURBO_E_BUSY()) {
                                busy.incrementAndGet();
                            } else {
                                other.incrementAndGet();
                            }
                        }
                    }
                });
                t.start();
                t.join();
                assertEquals(1, held.outputCount());
            }
            assertEquals(20, busy.get(), "every attempt while the result is held is BUSY");
            assertEquals(0, other.get(), "nothing else happened");
            s.writeText(List.of("after"), EmbedOptions.defaults());
            try (Result r = s.run()) {
                assertEquals(model.info().dim(), r.readFloats(0).length);
            }
        }
    }

    @Test
    void cancelFromAnotherThreadIsNeverBusyAndEndsTheStream() throws Exception {
        try (Turbo rt = Turbo.create();
                Context ctx = rt.createContext(mockDevice(rt));
                Model model = ctx.loadModel(bundle("generative"));
                Generation g = model.createGeneration(GenerateDesc.defaults().withMaxNewTokens(64))) {
            g.prompt(List.of(Message.user("say something")));
            Chunk first = g.step();
            assertFalse(first.done());
            CountDownLatch cancelled = new CountDownLatch(1);
            AtomicInteger failures = new AtomicInteger();
            Thread t = new Thread(() -> {
                try {
                    g.cancel();
                } catch (TurboException e) {
                    failures.incrementAndGet();
                }
                cancelled.countDown();
            });
            t.start();
            cancelled.await();
            t.join();
            assertEquals(0, failures.get(), "cancel from another thread never fails");
            Chunk next = g.step();
            assertTrue(next.done(), "the step after a cancel is the last one");
            assertEquals(FinishReason.CANCELLED, next.finishReason());
        }
    }

    @Test
    void indexedArrayAccessorsAddressTheField() {
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
            java.lang.foreign.MemorySegment d = ai.pipestream.turbo.ffi.turbo_buffer_desc.allocate(arena);
            ai.pipestream.turbo.ffi.turbo_buffer_desc.struct_size(d, 4242);
            ai.pipestream.turbo.ffi.turbo_buffer_desc.shape(d, 0, 7L);
            ai.pipestream.turbo.ffi.turbo_buffer_desc.shape(d, 2, 9L);
            assertEquals(4242, ai.pipestream.turbo.ffi.turbo_buffer_desc.struct_size(d), "the setter must not touch struct_size");
            assertEquals(7L, ai.pipestream.turbo.ffi.turbo_buffer_desc.shape(d, 0));
            assertEquals(9L, ai.pipestream.turbo.ffi.turbo_buffer_desc.shape(d, 2));
            assertEquals(7L, ai.pipestream.turbo.ffi.turbo_buffer_desc.shape(d).getAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG, 0));
        }
    }
}
