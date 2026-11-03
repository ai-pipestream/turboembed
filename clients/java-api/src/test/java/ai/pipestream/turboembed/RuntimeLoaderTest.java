package ai.pipestream.turboembed;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** Exercises the actual selection code without a native library or bridge dependency. */
final class RuntimeLoaderTest {
    @Test
    void selectionRequiresBothTheSupportedVersionAndForeignApi() {
        assertEquals(TurboBridge.JNI, RuntimeLoader.select("auto", 21, false, false));
        assertEquals(TurboBridge.JNI, RuntimeLoader.select("auto", 24, false, true));
        assertEquals(TurboBridge.JNI, RuntimeLoader.select("auto", 25, false, false));
        assertEquals(TurboBridge.PANAMA, RuntimeLoader.select("auto", 25, false, true));
        assertEquals(TurboBridge.PANAMA, RuntimeLoader.select("auto", 26, false, true));
    }

    @Test
    void androidNeverAutomaticallyLoadsPanama() {
        assertTrue(RuntimeLoader.isAndroid("Dalvik", ""));
        assertTrue(RuntimeLoader.isAndroid("ART", ""));
        assertTrue(RuntimeLoader.isAndroid("VM", "Android Runtime"));
        assertFalse(RuntimeLoader.isAndroid("OpenJDK 64-Bit Server VM", "OpenJDK Runtime Environment"));
        assertEquals(TurboBridge.JNI, RuntimeLoader.select("auto", 25, true, true));
        assertThrows(IllegalStateException.class, () -> RuntimeLoader.select("panama", 25, true, true));
    }

    @Test
    void explicitChoicesAreStrictAndJniWorksOnNewerRuntimes() {
        assertEquals(TurboBridge.JNI, RuntimeLoader.select("JNI", 25, false, true));
        assertEquals(TurboBridge.PANAMA, RuntimeLoader.select("panama", 25, false, true));
        assertThrows(IllegalStateException.class, () -> RuntimeLoader.select("panama", 21, false, false));
        assertThrows(IllegalArgumentException.class, () -> RuntimeLoader.select("unknown", 25, false, true));
        assertThrows(IllegalArgumentException.class, () -> RuntimeLoader.select("", 25, false, true));
        assertThrows(NullPointerException.class, () -> TurboRuntime.load(null));
    }

    @Test
    void apiOnlyClasspathReportsTheMissingSelectedImplementation() {
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> TurboRuntime.load(TurboBridge.JNI));
        assertTrue(failure.getMessage().contains("turboembed-jni"), failure.getMessage());
        assertInstanceOf(ClassNotFoundException.class, failure.getCause());
    }

    @Test
    void sharedValuesAreUsableWithoutLoadingEitherBridge() {
        assertEquals("en", new TurboAnalyzer.Config(" \t", null).language());
        assertEquals("pt", TurboAnalyzer.Config.forLanguage("pt").language());
        assertEquals(TurboLayerKind.MORPHOLOGY, TurboLayerKind.fromNative(8));
        assertThrows(IllegalStateException.class, () -> TurboLayerKind.fromNative(99));
        assertEquals(TurboPiiConfidence.CHECKSUM, TurboPiiConfidence.fromNative(2));
        TurboException failure = new TurboException(256, "TURBO_E_INVALID_ARGUMENT", 7, "bad bundle");
        assertEquals(7, failure.field());
        assertEquals(256, failure.code());
        assertEquals("TURBO_E_INVALID_ARGUMENT", failure.statusName());
        var range = new TurboSpanData.CoordinateRange(0, 4, 0, 2, 0, 1);
        assertEquals("🚀", range.coveredText("🚀 text"));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboSpanData.CoordinateRange(4, 0, 0, 2, 0, 1));
    }
}
