package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Host unit tests for {@link AndroidTestReportVerifier}. */
final class AndroidTestReportVerifierTest {
    @TempDir
    Path temp;

    @Test
    void refusesMissingOrEmptyReportDirectory() {
        assertThrows(
                IllegalStateException.class,
                () -> AndroidTestReportVerifier.verify(temp.resolve("does-not-exist")));
        assertThrows(IllegalStateException.class, () -> AndroidTestReportVerifier.verify(temp));
    }

    @Test
    void refusesZeroTestConnectedReport() throws Exception {
        Path xml = temp.resolve("TEST-emulator-5554-app-.xml");
        Files.writeString(
                xml,
                "<?xml version='1.0' encoding='UTF-8' ?>\n"
                        + "<testsuite name=\"ai.pipestream.turboembed.jni.android.test\" tests=\"0\" failures=\"0\" errors=\"0\" skipped=\"0\" time=\"0.001\" />\n",
                StandardCharsets.UTF_8);
        IllegalStateException ex =
                assertThrows(IllegalStateException.class, () -> AndroidTestReportVerifier.verify(temp));
        assertTrue(ex.getMessage().contains("0 tests"), ex.getMessage());
    }

    @Test
    void refusesReportWithTestFailureOrMissingRequiredSuite() throws Exception {
        Path xml = temp.resolve("TEST-emulator-5554-app-.xml");
        Files.writeString(
                xml,
                "<?xml version='1.0' encoding='UTF-8' ?>\n"
                        + "<testsuite tests=\"1\" failures=\"1\" errors=\"0\" skipped=\"0\">\n"
                        + "  <testcase name=\"broken\" classname=\"ai.pipestream.turboembed.jni.TurboJniTest\">\n"
                        + "    <failure message=\"boom\">stack</failure>\n"
                        + "  </testcase>\n"
                        + "</testsuite>\n",
                StandardCharsets.UTF_8);
        IllegalStateException failed =
                assertThrows(IllegalStateException.class, () -> AndroidTestReportVerifier.verify(temp));
        assertTrue(failed.getMessage().contains("failures/errors"), failed.getMessage());

        Files.writeString(
                xml,
                "<?xml version='1.0' encoding='UTF-8' ?>\n"
                        + "<testsuite tests=\"1\" failures=\"0\" errors=\"0\" skipped=\"0\">\n"
                        + "  <testcase name=\"onlyOne\" classname=\"ai.pipestream.turboembed.jni.TurboJniTest\" />\n"
                        + "</testsuite>\n",
                StandardCharsets.UTF_8);
        IllegalStateException incomplete =
                assertThrows(IllegalStateException.class, () -> AndroidTestReportVerifier.verify(temp));
        assertTrue(incomplete.getMessage().contains("expected at least"), incomplete.getMessage());
    }

    @Test
    void acceptsCompletePassingReportAcrossAllRequiredSuites() throws Exception {
        StringBuilder xml = new StringBuilder();
        xml.append("<?xml version='1.0' encoding='UTF-8' ?>\n");
        xml.append("<testsuite tests=\"34\" failures=\"0\" errors=\"0\" skipped=\"0\">\n");
        for (Map.Entry<String, Integer> entry : AndroidTestReportVerifier.REQUIRED_SUITES.entrySet()) {
            for (int i = 0; i < entry.getValue(); i++) {
                xml.append("  <testcase name=\"test_")
                        .append(i)
                        .append("\" classname=\"")
                        .append(entry.getKey())
                        .append("\" time=\"0.01\" />\n");
            }
        }
        xml.append("</testsuite>\n");
        Path file = temp.resolve("debug/TEST-emulator-5554-app-.xml");
        Files.createDirectories(file.getParent());
        Files.writeString(file, xml.toString(), StandardCharsets.UTF_8);

        AndroidTestReportVerifier.Summary summary = AndroidTestReportVerifier.verify(temp);
        assertEquals(34, summary.totalPassed());
        assertEquals(0, summary.totalFailed());
        assertEquals(0, summary.totalErrors());
        assertEquals(0, summary.totalSkipped());
        assertEquals(
                13,
                summary.suites()
                        .get("ai.pipestream.turboembed.jni.android.AndroidAnalyzerConformanceTest")
                        .passed());
        assertEquals(
                2,
                summary.suites()
                        .get("ai.pipestream.turboembed.jni.android.AndroidSessionConformanceTest")
                        .passed());
        assertEquals(
                3,
                summary.suites()
                        .get("ai.pipestream.turboembed.jni.android.AndroidTurboJniBridgeTest")
                        .passed());
        assertEquals(
                10,
                summary.suites().get("ai.pipestream.turboembed.jni.TurboJniTest").passed());
        assertEquals(
                6,
                summary.suites().get("ai.pipestream.turboembed.opennlp.TurboAdaptersTest").passed());
    }
}
