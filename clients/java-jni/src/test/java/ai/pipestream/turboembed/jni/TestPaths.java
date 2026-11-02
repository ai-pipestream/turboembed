package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.conformance.Testdata;
import java.nio.file.Path;

/** The repository's testdata bundles, through the conformance suite's locator. */
public final class TestPaths {
    private TestPaths() {}

    public static Path repoRoot() {
        return Testdata.repoRoot();
    }

    /** The sealed small BERT bundle in testdata with no heads. */
    public static Path tinyBertBundle() {
        return Testdata.tinyBertBundle();
    }

    /** The sealed small BERT bundle in testdata with seeded part-of-speech, entity, dependency and lemma heads. */
    public static Path headsBundle() {
        return Testdata.headsBundle();
    }
}
