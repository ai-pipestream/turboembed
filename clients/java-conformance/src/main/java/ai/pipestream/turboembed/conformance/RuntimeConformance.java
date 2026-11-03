package ai.pipestream.turboembed.conformance;

import static org.junit.jupiter.api.Assertions.*;

import ai.pipestream.turboembed.*;
import ai.pipestream.turboembed.opennlp.TurboTokenizerME;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Actual native calls through the common interfaces and runtime loader. */
public abstract class RuntimeConformance {
    protected abstract TurboBridge bridge();

    @Test
    public void freshJvmHonorsAutomaticAndExplicitSelection() throws Exception {
        String own = bridge().name();
        assertEquals(own, probe(own.toLowerCase(java.util.Locale.ROOT), null, true));
        assertEquals(own, probe("invalid", own, true));
        String automatic = Runtime.version().feature() >= 25 ? "PANAMA" : "JNI";
        if (automatic.equals(own)) {
            assertEquals(own, probe("auto", null, true));
        } else {
            assertTrue(probe("auto", null, false).contains("Selected PANAMA bridge is missing"));
        }
        assertTrue(probe("invalid", null, false).contains("turbo.bridge must be"));
    }

    private static String probe(String property, String explicit, boolean succeeds) throws Exception {
        String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        var command = new java.util.ArrayList<>(List.of(
                java.nio.file.Path.of(System.getProperty("java.home"), "bin", executable).toString(),
                "-Dturbo.bridge=" + property, "-cp", System.getProperty("java.class.path"),
                LoaderProbe.class.getName()));
        if (explicit != null) { command.add(explicit); }
        Process child = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            assertTrue(child.waitFor(20, java.util.concurrent.TimeUnit.SECONDS), "loader process timed out");
            String output = new String(child.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            assertEquals(succeeds ? 0 : 1, child.exitValue(), output);
            return output.strip();
        } finally {
            if (child.isAlive()) { child.destroyForcibly(); }
        }
    }

    @Test
    public void loadsTheChosenBridgeAndSharedAdaptersUseIt() {
        TurboRuntime runtime = TurboRuntime.load(bridge());
        assertEquals(bridge(), runtime.kind());
        // Each binding's test process selects itself with turbo.bridge.
        assertEquals(bridge(), TurboRuntime.load().kind());
        try (TurboAnalyzer analyzer = runtime.createAnalyzer(TurboAnalyzer.Config.defaults());
             TurboTokenizerME tokenizer = new TurboTokenizerME(analyzer)) {
            assertArrayEquals(new String[] {"Café", "🚀", "!"}, tokenizer.tokenize("Café 🚀!"));
        }
        try (TurboTokenizerME tokenizer = new TurboTokenizerME()) {
            assertArrayEquals(new String[] {"Hello", "!"}, tokenizer.tokenize("Hello!"));
        }
    }

    @Test
    public void retainedOwnersAndCopiedCoordinatesHaveIndependentLifetimes() {
        TurboRuntime runtime = TurboRuntime.load(bridge());
        TurboAnalyzer first = runtime.createAnalyzer(TurboAnalyzer.Config.defaults());
        try (TurboAnalyzer second = first.retain()) {
            first.close();
            assertTrue(first.isClosed());
            assertThrows(IllegalStateException.class, () -> first.tokenize("closed"));
            TurboAnalysisResult result = second.tokenize("Café 🚀!");
            TurboSpanView view = result.span(1);
            List<TurboSpanData> copied = result.copySpans();
            try (TurboAnalysisResult retained = result.retain()) {
                result.close();
                assertTrue(result.isClosed());
                assertThrows(IllegalStateException.class, view::label);
                assertEquals("🚀", retained.span(1).coveredSourceText("Café 🚀!"));
                assertEquals(copied, retained.copySpans());
                assertEquals(new TurboSpanData.CoordinateRange(6, 10, 5, 7, 5, 6),
                        copied.get(1).sourceRange().orElseThrow());
            } finally {
                result.close();
            }
            assertEquals("🚀", copied.get(1).coveredSourceText("Café 🚀!"));
        } finally {
            first.close();
        }
    }

    @Test
    public void bothImplementationsThrowTheCommonExceptionWithNativeDetails() {
        TurboException error = assertThrows(TurboException.class,
                () -> TurboRuntime.load(bridge()).openAnalyzer("en", Testdata.missingBundle()));
        assertEquals(1280, error.code());
        assertEquals("TURBO_E_BUNDLE_NOT_FOUND", error.statusName());
        assertEquals(7, error.field());
    }
}
