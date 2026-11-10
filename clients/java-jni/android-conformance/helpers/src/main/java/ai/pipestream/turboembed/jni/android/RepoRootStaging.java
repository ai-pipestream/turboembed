package ai.pipestream.turboembed.jni.android;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Builds a miniature repository root the conformance suite's {@code Testdata} locator accepts:
 * the two sealed bundles under {@code testdata/} and a stub
 * {@code clients/java-conformance/pom.xml}. Used by the Android instrumented harness (assets to
 * the application files directory) and by a host unit test over the real {@code testdata} tree.
 *
 * <p>This class does not load {@code libturbo} and does not talk to ART.
 */
public final class RepoRootStaging {
    /** Relative paths of every file the sealed tiny BERT bundle carries. */
    public static final List<String> TINY_BERT_FILES =
            Collections.unmodifiableList(
                    Arrays.asList(
                            "manifest.json",
                            "tokenizer.json",
                            "weights/model.safetensors",
                            "reference/reference.safetensors"));

    /** Relative paths of every file the sealed tiny BERT heads bundle carries. */
    public static final List<String> TINY_BERT_HEADS_FILES =
            Collections.unmodifiableList(
                    Arrays.asList(
                            "manifest.json",
                            "tokenizer.json",
                            "weights/model.safetensors",
                            "reference/reference.safetensors"));

    /** Relative paths of every file the sealed tiny BERT segment bundle carries. */
    public static final List<String> TINY_BERT_SEGMENT_FILES =
            Collections.unmodifiableList(
                    Arrays.asList(
                            "manifest.json",
                            "tokenizer.json",
                            "weights/model.safetensors",
                            "reference/reference.safetensors"));

    /** Marker file {@code Testdata.looksLikeRepoRoot} requires beside the bundles. */
    public static final String CONFORMANCE_POM = "clients/java-conformance/pom.xml";

    private static final String TINY_BERT = "tiny-bert-bundle";
    private static final String TINY_BERT_HEADS = "tiny-bert-heads-bundle";
    private static final String TINY_BERT_SEGMENT = "tiny-bert-segment-bundle";
    private static final String STUB_POM =
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                    + "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n"
                    + "  <modelVersion>4.0.0</modelVersion>\n"
                    + "  <groupId>ai.pipestream</groupId>\n"
                    + "  <artifactId>turboembed-conformance-android-stub</artifactId>\n"
                    + "  <version>0.0.0</version>\n"
                    + "  <packaging>pom</packaging>\n"
                    + "</project>\n";

    /** Opens one relative path under a bundle or other staged prefix. */
    @FunctionalInterface
    public interface FileSource {
        /**
         * Opens {@code relativePath} for reading.
         *
         * @param relativePath path relative to the source root (forward slashes)
         * @return the file bytes, never null
         * @throws IOException when the file is missing or unreadable
         */
        InputStream open(String relativePath) throws IOException;
    }

    private RepoRootStaging() {}

    /**
     * Writes both sealed bundles and the stub pom under {@code destinationRoot}.
     *
     * @param destinationRoot directory that becomes {@code turbo.repo.root}
     * @param source opens files relative to a root that contains {@code testdata/...}
     * @return {@code destinationRoot} normalized
     * @throws IOException on I/O failure
     */
    public static Path stage(Path destinationRoot, FileSource source) throws IOException {
        Objects.requireNonNull(destinationRoot, "destinationRoot");
        Objects.requireNonNull(source, "source");
        Path root = destinationRoot.toAbsolutePath().normalize();
        Files.createDirectories(root);
        copyBundle(root, source, TINY_BERT, TINY_BERT_FILES);
        copyBundle(root, source, TINY_BERT_HEADS, TINY_BERT_HEADS_FILES);
        copyBundle(root, source, TINY_BERT_SEGMENT, TINY_BERT_SEGMENT_FILES);
        Path pom = root.resolve(CONFORMANCE_POM);
        Files.createDirectories(pom.getParent());
        Files.write(pom, STUB_POM.getBytes(StandardCharsets.UTF_8));
        if (!looksLikeRepoRoot(root)) {
            throw new IOException("staged root is incomplete: " + root);
        }
        return root;
    }

    /**
     * Stages from a filesystem tree whose {@code testdata/} holds the sealed bundles (the
     * repository root on a developer machine or CI checkout).
     *
     * @param destinationRoot directory that becomes {@code turbo.repo.root}
     * @param repositoryRoot repository root containing {@code testdata/}
     * @return {@code destinationRoot} normalized
     * @throws IOException on I/O failure
     */
    public static Path stageFromRepository(Path destinationRoot, Path repositoryRoot) throws IOException {
        Path repo = repositoryRoot.toAbsolutePath().normalize();
        return stage(destinationRoot, relative -> {
            Path file = repo.resolve(relative);
            if (!Files.isRegularFile(file)) {
                throw new IOException("missing " + file);
            }
            return Files.newInputStream(file);
        });
    }

    /**
     * Returns whether {@code dir} matches the conformance suite's repository-root check.
     *
     * @param dir candidate root
     * @return true when both bundles and the stub pom are present
     */
    public static boolean looksLikeRepoRoot(Path dir) {
        return Files.isDirectory(dir.resolve("testdata").resolve(TINY_BERT))
                && Files.isDirectory(dir.resolve("testdata").resolve(TINY_BERT_HEADS))
                && Files.isDirectory(dir.resolve("testdata").resolve(TINY_BERT_SEGMENT))
                && Files.isRegularFile(dir.resolve(CONFORMANCE_POM));
    }

    private static void copyBundle(Path root, FileSource source, String bundle, List<String> files)
            throws IOException {
        Path bundleDir = root.resolve("testdata").resolve(bundle);
        for (String relative : files) {
            Path out = bundleDir.resolve(relative);
            Files.createDirectories(out.getParent());
            String sourcePath = "testdata/" + bundle + "/" + relative;
            try (InputStream in = source.open(sourcePath);
                    OutputStream outStream = Files.newOutputStream(out)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = in.read(buffer)) >= 0) {
                    outStream.write(buffer, 0, read);
                }
            }
            if (Files.size(out) == 0 && !relative.endsWith(".json")) {
                // Empty weights would never be a sealed bundle; keep the check narrow.
                throw new IOException("staged empty file: " + out);
            }
        }
    }

    /**
     * Copies one file, replacing an existing destination.
     *
     * @param source source file
     * @param destination destination file
     * @throws IOException on I/O failure
     */
    public static void copyFile(Path source, Path destination) throws IOException {
        Files.createDirectories(destination.getParent());
        Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
    }
}
