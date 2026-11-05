package ai.pipestream.turboembed.jni.android;

import org.junit.runner.Runner;

/**
 * Android instrumentation {@link Runner} that stages the sealed {@code testdata} bundles into the
 * target application's files directory before executing a JUnit Jupiter test class.
 */
public final class AndroidJupiterRunner extends JupiterTestRunner {
    /**
     * Creates a runner for {@code testClass} wired to {@link AndroidConformanceSetup#ensureReady()}.
     *
     * @param testClass concrete class carrying {@code @org.junit.jupiter.api.Test} methods
     */
    public AndroidJupiterRunner(Class<?> testClass) {
        super(testClass, AndroidConformanceSetup::ensureReady);
    }
}
