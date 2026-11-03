package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * Guards the AndroidSourceSet configure failure that blocked helpers-host: a bare
 * {@code exclude(...)} inside {@code getByName("androidTest")} does not compile against
 * AGP's {@code AndroidSourceSet}. Prefer an explicit {@code java.setSrcDirs(...)} list.
 */
final class AppGradleSourceSetTest {
    private static final Pattern BARE_EXCLUDE_ON_ANDROID_TEST = Pattern.compile(
            "getByName\\(\"androidTest\"\\)\\s*\\{[\\s\\S]*?\\n\\s*exclude\\(");

    @Test
    void androidTestSourceSetDoesNotCallBareExclude() throws IOException {
        Path script = appBuildGradle();
        String text = Files.readString(script);
        assertTrue(text.contains("getByName(\"androidTest\")"), script.toString());
        assertTrue(text.contains("java.setSrcDirs("), text);
        assertFalse(
                BARE_EXCLUDE_ON_ANDROID_TEST.matcher(text).find(),
                "exclude() must not hang off the AndroidSourceSet receiver in androidTest");
        assertTrue(text.contains("../../../java-api/src/main/java"), text);
        assertFalse(text.contains("JniAnalyzerConformanceTest.java"), text);
        assertFalse(text.contains("JniRuntimeConformanceTest.java"), text);
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
