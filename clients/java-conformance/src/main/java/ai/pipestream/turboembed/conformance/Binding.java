package ai.pipestream.turboembed.conformance;

import java.nio.file.Path;

/**
 * One in-process Java binding of {@code libturbo} under the conformance suite. A binding
 * adapts its own analyzer, session and exception types to the neutral ones here; the suite
 * then holds every binding to the same rows, labels, probabilities, vectors and errors.
 */
public interface Binding {
    /** A short name for messages, such as {@code "jni"} or {@code "panama"}. */
    String name();

    /**
     * Opens an analyzer.
     *
     * @param language   a BCP-47 language tag
     * @param bundlePath a bundle directory whose heads write the learned layers, or null for the
     *                   rule sentence splitter and tokenizer alone
     * @return the open analyzer
     */
    Analyzer openAnalyzer(String language, Path bundlePath);

    /** Opens an embedding or reranking session over a bundle on the CPU device. */
    Session openSession(Path bundleDir);

    /**
     * The native status code a failed call threw.
     *
     * @return the {@code TURBO_E_*} code, or -1 when {@code failure} is not this binding's
     *     native exception
     */
    int statusCode(Throwable failure);

    /** The symbolic status name a failed call threw, or null. */
    String statusName(Throwable failure);

    /** The 1-based field a failed call named, 0 for none, or -1 as {@link #statusCode}. */
    int statusField(Throwable failure);
}
