package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turboembed.jni.TestPaths;
import ai.pipestream.turboembed.jni.TurboAnalyzer;
import ai.pipestream.turboembed.jni.TurboException;
import ai.pipestream.turboembed.jni.TurboLibrary;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * Bridge smoke on ART: {@link System#loadLibrary(String)} discovery, version string, and a
 * bundle-not-found error. The fuller host {@code TurboJniTest} suite also runs under
 * instrumentation via the shared test sources; this class names the Android load path
 * explicitly so a missing {@code jniLibs} entry cannot hide behind a host property.
 */
@ExtendWith(AndroidConformanceSetup.class)
final class AndroidTurboJniBridgeTest {
    @Test
    void loadLibraryPathAnswersVersionAndBundleErrors() {
        TurboLibrary.load();
        String version = TurboLibrary.version();
        assertNotNull(version);
        assertTrue(version.startsWith("0."), "unexpected version: " + version);
        assertTrue(version.contains("cpu"), "cpu backend missing from: " + version);

        TurboException ex = assertThrows(
                TurboException.class,
                () -> TurboAnalyzer.create(
                        new TurboAnalyzer.Config("en", TestPaths.repoRoot().resolve("testdata/does-not-exist"))));
        assertEquals(TurboLibrary.TURBO_E_BUNDLE_NOT_FOUND, ex.code());
        assertEquals("TURBO_E_BUNDLE_NOT_FOUND", ex.statusName());
    }
}
