package ai.pipestream.turboembed.conformance;

import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/** The sealed bundles in the repository's {@code testdata}, found without a workstation path. */
public final class Testdata {
    private Testdata() {}

    /** The repository root: {@code turbo.repo.root} when set, else the nearest ancestor that looks like it. */
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
            starts.add(Path.of(Testdata.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                    .toAbsolutePath()
                    .normalize());
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

    /** The sealed small BERT bundle with no heads. */
    public static Path tinyBertBundle() {
        return bundle("tiny-bert-bundle");
    }

    /** The sealed small BERT bundle with seeded part-of-speech, entity, dependency and lemma heads. */
    public static Path headsBundle() {
        return bundle("tiny-bert-heads-bundle");
    }

    /** A path under {@code testdata} that no bundle is at. */
    public static Path missingBundle() {
        return repoRoot().resolve("testdata").resolve("does-not-exist");
    }

    private static Path bundle(String name) {
        Path path = repoRoot().resolve("testdata").resolve(name);
        if (!Files.isDirectory(path)) {
            throw new AssertionError("missing testdata bundle: " + path);
        }
        return path;
    }

    private static boolean looksLikeRepoRoot(Path dir) {
        return Files.isDirectory(dir.resolve("testdata/tiny-bert-bundle"))
                && Files.isDirectory(dir.resolve("testdata/tiny-bert-heads-bundle"))
                && Files.isRegularFile(dir.resolve("clients/java-conformance/pom.xml"));
    }
}
