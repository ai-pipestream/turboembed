package ai.pipestream.turboembed.panama;

import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/** Locates repository fixtures without embedding a workstation-specific path in tests. */
public final class TestPaths {
    private TestPaths() {}

    public static Path repoRoot() {
        String configured = System.getProperty("turbo.repo.root");
        if (configured != null && !configured.isBlank()) {
            Path root = Path.of(configured).toAbsolutePath().normalize();
            if (looksLikeRepoRoot(root)) {
                return root;
            }
            throw new AssertionError("turbo.repo.root does not look like the repository root: " + root);
        }

        LinkedHashSet<Path> starts = new LinkedHashSet<>();
        starts.add(Path.of("").toAbsolutePath().normalize());
        try {
            Path codeSource = Path.of(TestPaths.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            starts.add(codeSource.toAbsolutePath().normalize());
        } catch (URISyntaxException ignored) {
            // The working-directory candidates below remain available.
        }

        List<Path> candidates = new ArrayList<>();
        for (Path start : starts) {
            for (Path dir = Files.isDirectory(start) ? start : start.getParent(); dir != null; dir = dir.getParent()) {
                candidates.add(dir);
                if (looksLikeRepoRoot(dir)) {
                    return dir;
                }
            }
        }
        throw new AssertionError("could not locate repository root from candidates: " + candidates);
    }

    public static Path fixture(String fileName) {
        Path path = repoRoot().resolve("clients/java-panama/src/test/resources/fixtures").resolve(fileName);
        if (!Files.isRegularFile(path)) {
            throw new AssertionError("missing test fixture: " + path);
        }
        return path;
    }

    public static Path tinyBertBundle() {
        Path path = repoRoot().resolve("testdata/tiny-bert-bundle");
        if (!Files.isDirectory(path)) {
            throw new AssertionError("missing tiny BERT bundle: " + path);
        }
        return path;
    }

    private static boolean looksLikeRepoRoot(Path dir) {
        return Files.isDirectory(dir.resolve("clients/java-panama/src/test/resources/fixtures"))
                && Files.isDirectory(dir.resolve("testdata/tiny-bert-bundle"))
                && Files.isRegularFile(dir.resolve("clients/java-panama/pom.xml"));
    }
}
