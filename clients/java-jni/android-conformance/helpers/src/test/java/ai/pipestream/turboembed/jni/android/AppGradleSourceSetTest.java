package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * Guards the AndroidSourceSet configure failure that blocked helpers-host: a bare
 * {@code exclude(...)} inside {@code getByName("androidTest")} does not compile against
 * AGP's {@code AndroidSourceSet}. Prefer directory {@code java.setSrcDirs(...)} plus
 * {@code tasks.withType<JavaCompile>()} excludes for host-only classes.
 */
final class AppGradleSourceSetTest {
    @Test
    void androidTestSourceSetUsesDirectoriesAndJavaCompileExcludes() throws IOException {
        Path script = appBuildGradle();
        String text = Files.readString(script);
        assertTrue(text.contains("getByName(\"androidTest\")"), script.toString());
        assertTrue(text.contains("java.setSrcDirs("), text);
        String androidTestBlock = braceBlockAfter(text, "getByName(\"androidTest\")");
        assertFalse(
                androidTestBlock.contains("exclude("),
                "exclude() must not hang off the AndroidSourceSet receiver in androidTest");
        assertTrue(text.contains("../../../java-api/src/main/java"), text);
        assertTrue(text.contains("../../../java-conformance/src/main/java"), text);
        assertTrue(text.contains("../../src/test/java"), text);
        // Individual .java file paths are not source directories.
        assertFalse(text.contains("TestPaths.java"), text);
        assertFalse(text.contains("TurboJniTest.java"), text);
        assertTrue(text.contains("tasks.withType<JavaCompile>()"), text);
        assertTrue(text.contains("**/JniAnalyzerConformanceTest.java"), text);
        assertTrue(text.contains("**/JniRuntimeConformanceTest.java"), text);
        assertTrue(text.contains("**/RuntimeConformance.java"), text);
        assertTrue(text.contains("**/LoaderProbe.java"), text);
        // android.jar lacks MatchException; keep the Android compile at language level 17.
        assertTrue(text.contains("JavaVersion.VERSION_17"), text);
        assertFalse(text.contains("JavaVersion.VERSION_21"), text);
        // Explicit Jupiter bridge runner and suite class list prevent 0-test connected runs.
        assertTrue(
                text.contains("ai.pipestream.turboembed.jni.android.AndroidJupiterRunnerBuilder"),
                text);
        for (String requiredSuite : AndroidTestReportVerifier.REQUIRED_SUITES.keySet()) {
            assertTrue(text.contains(requiredSuite), "missing suite in app/build.gradle.kts: " + requiredSuite);
        }
        // JUnit 5 jars ship duplicate META-INF/LICENSE*.md entries; packaging.resources must exclude them.
        assertTrue(text.contains("META-INF/LICENSE.md"), text);
        assertTrue(text.contains("META-INF/LICENSE-notice.md"), text);
    }

    /** Returns the brace-delimited block that starts at the first '{' after {@code marker}. */
    private static String braceBlockAfter(String text, String marker) {
        int at = text.indexOf(marker);
        assertTrue(at >= 0, "missing " + marker);
        int open = text.indexOf('{', at);
        assertTrue(open >= 0, "missing '{' after " + marker);
        int depth = 0;
        for (int i = open; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    return text.substring(open, i + 1);
                }
            }
        }
        throw new AssertionError("unbalanced braces after " + marker);
    }

    private static Path appBuildGradle() throws IOException {
        Path start = Path.of("").toAbsolutePath().normalize();
        Path fromHelpers = start.resolve("../app/build.gradle.kts").normalize();
        if (Files.isRegularFile(fromHelpers)) {
            return fromHelpers;
        }
        Path fromRoot = start.resolve("app/build.gradle.kts");
        if (Files.isRegularFile(fromRoot)) {
            return fromRoot;
        }
        throw new IOException("could not locate app/build.gradle.kts from " + start);
    }
}
