package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Host verification of {@link RepoRootStaging} against the real sealed bundles. */
final class RepoRootStagingTest {
    @TempDir
    Path temp;

    @Test
    void stagesBothBundlesAndTheStubPomFromTheRepositoryTestdata() throws Exception {
        Path repo = repositoryRoot();
        Path staged = RepoRootStaging.stageFromRepository(temp.resolve("root"), repo);
        assertTrue(RepoRootStaging.looksLikeRepoRoot(staged));
        for (String file : RepoRootStaging.TINY_BERT_FILES) {
            Path original = repo.resolve("testdata/tiny-bert-bundle").resolve(file);
            Path copy = staged.resolve("testdata/tiny-bert-bundle").resolve(file);
            assertEquals(Files.size(original), Files.size(copy), file);
            assertTrue(Files.mismatch(original, copy) < 0, file);
        }
        for (String file : RepoRootStaging.TINY_BERT_HEADS_FILES) {
            Path original = repo.resolve("testdata/tiny-bert-heads-bundle").resolve(file);
            Path copy = staged.resolve("testdata/tiny-bert-heads-bundle").resolve(file);
            assertEquals(Files.size(original), Files.size(copy), file);
            assertTrue(Files.mismatch(original, copy) < 0, file);
        }
        assertTrue(Files.size(staged.resolve(RepoRootStaging.CONFORMANCE_POM)) > 0);
    }

    @Test
    void missingSourceFileSurfacesTheRelativePath() {
        IOException failure = assertThrows(
                IOException.class,
                () -> RepoRootStaging.stage(temp.resolve("broken"), relative -> {
                    throw new IOException("missing " + relative);
                }));
        assertTrue(failure.getMessage().contains("testdata/tiny-bert-bundle/"), failure.getMessage());
    }

    private static Path repositoryRoot() throws IOException {
        Path start = Path.of("").toAbsolutePath().normalize();
        for (Path dir = start; dir != null; dir = dir.getParent()) {
            if (Files.isDirectory(dir.resolve("testdata/tiny-bert-bundle"))
                    && Files.isDirectory(dir.resolve("testdata/tiny-bert-heads-bundle"))
                    && Files.isRegularFile(dir.resolve("clients/java-conformance/pom.xml"))) {
                return dir;
            }
        }
        // Gradle runs helpers tests with the module directory as the working directory.
        Path fromModule = start.resolve("../../..").normalize();
        if (Files.isDirectory(fromModule.resolve("testdata/tiny-bert-bundle"))) {
            return fromModule;
        }
        try (Stream<Path> ignored = Files.list(start)) {
            throw new IOException("could not locate repository root from " + start);
        }
    }
}
