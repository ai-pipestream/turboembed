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
        assertEquals(TurboPiiConfidence.LABELLED_CHECKSUM, TurboPiiConfidence.fromNative(3));
        assertEquals(3, TurboPiiConfidence.LABELLED_CHECKSUM.nativeCode());
        assertEquals(TurboPiiConfidence.LABELLED, TurboPiiConfidence.fromNative(4));
        assertEquals(4, TurboPiiConfidence.LABELLED.nativeCode());
        assertEquals(TurboPiiConfidence.LABELLED, TurboPiiConfidence.fromNative(TurboPiiConfidence.LABELLED.nativeCode()));
        assertThrows(IllegalStateException.class, () -> TurboPiiConfidence.fromNative(5));
        TurboException failure = new TurboException(256, "TURBO_E_INVALID_ARGUMENT", 7, "bad bundle");
        assertEquals(7, failure.field());
        assertEquals(256, failure.code());
        assertEquals("TURBO_E_INVALID_ARGUMENT", failure.statusName());
        var range = new TurboSpanData.CoordinateRange(0, 4, 0, 2, 0, 1);
        assertEquals("🚀", range.coveredText("🚀 text"));
        assertThrows(IllegalArgumentException.class,
                () -> new TurboSpanData.CoordinateRange(4, 0, 0, 2, 0, 1));
    }

    @Test
    void jvmFeatureParsesSpecificationVersions() {
        assertEquals(21, RuntimeLoader.jvmFeature("21"));
        assertEquals(25, RuntimeLoader.jvmFeature("25"));
        assertEquals(8, RuntimeLoader.jvmFeature("1.8"));
        assertEquals(11, RuntimeLoader.jvmFeature("11.0.2"));
        assertEquals(0, RuntimeLoader.jvmFeature(""));
        assertEquals(0, RuntimeLoader.jvmFeature(null));
        assertEquals(0, RuntimeLoader.jvmFeature("x"));
    }

    @Test
    void selectedDoesNotReferenceRuntimeVersionSymbol() throws Exception {
        // android.jar does not declare the Runtime version API; keep the call out of this shared source.
        java.nio.file.Path source = java.nio.file.Path.of("src/main/java/ai/pipestream/turboembed/RuntimeLoader.java");
        if (!java.nio.file.Files.isRegularFile(source)) {
            source = java.nio.file.Path.of("../src/main/java/ai/pipestream/turboembed/RuntimeLoader.java");
        }
        String body = java.nio.file.Files.readString(source);
        assertFalse(body.contains("Runtime.version("), body);
        assertTrue(body.contains("java.specification.version"), body);
    }
}
