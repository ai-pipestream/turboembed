// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Always-on failure coverage for {@link FfmTurboEmbed#open}. No prepared SDK is
 * installed or loaded here: every case asserts the adapter rejects an unusable
 * prefix loudly, with the offending library path in the message, instead of
 * guessing or falling back to another provider.
 */
class FfmOpenFailureTest {
    @TempDir Path temporary;

    private static Path bundledLibrary(Path prefix) {
        return prefix.resolve("lib").resolve("libturboembed_prepared.so.1");
    }

    @Test void openRejectsNonexistentPrefixWithDocumentedException() {
        Path prefix = temporary.resolve("no-such-prefix");
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> FfmTurboEmbed.open(prefix));
        assertTrue(failure.getMessage().contains("no installed prepared SDK library"),
            () -> "message should explain what is missing, was: " + failure.getMessage());
        assertTrue(failure.getMessage().contains(bundledLibrary(prefix).toString()),
            () -> "message should name the exact library path, was: " + failure.getMessage());
    }

    @Test void openFailsLoudWhenBundledLibraryIsMissingFromPrefix() throws IOException {
        Path prefix = temporary.resolve("sdk-prefix");
        Files.createDirectories(prefix.resolve("lib")); // layout exists, artifact absent
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> FfmTurboEmbed.open(prefix));
        assertTrue(failure.getMessage().contains(bundledLibrary(prefix).toString()),
            () -> "message should name the missing artifact, was: " + failure.getMessage());
    }

    @Test void openRejectsDirectoryPosedAsLibrary() throws IOException {
        Path prefix = temporary.resolve("dir-prefix");
        Files.createDirectories(bundledLibrary(prefix));
        assertThrows(IllegalArgumentException.class, () -> FfmTurboEmbed.open(prefix));
    }

    @Test void openFailsLoudOnUnloadableLibraryFile() throws IOException {
        Path prefix = temporary.resolve("garbage-prefix");
        Files.createDirectories(prefix.resolve("lib"));
        Files.writeString(bundledLibrary(prefix), "not an ELF shared object");
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> FfmTurboEmbed.open(prefix));
        assertTrue(failure.getMessage().contains("libturboembed_prepared.so.1"),
            () -> "loader failure should still name the offending file, was: " + failure.getMessage());
    }

    @Test void openRejectsNullPrefix() {
        assertThrows(NullPointerException.class, () -> FfmTurboEmbed.open(null));
    }

    @Test void libraryPathMatchesBundledLayoutOnSupportedPlatforms() {
        boolean linuxAmd64 = System.getProperty("os.name").equals("Linux")
            && System.getProperty("os.arch").equals("amd64");
        if (linuxAmd64) {
            assertEquals(Path.of("lib", "libturboembed_prepared.so.1"), FfmTurboEmbed.libraryPath());
        } else {
            assertThrows(UnsupportedOperationException.class, FfmTurboEmbed::libraryPath);
        }
    }
}
