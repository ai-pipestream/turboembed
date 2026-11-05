package ai.pipestream.turboembed.jni.android;

import org.junit.runner.Runner;
import org.junit.runners.model.RunnerBuilder;

/**
 * Custom {@link RunnerBuilder} registered via {@code testInstrumentationRunnerArguments["runnerBuilder"]}
 * so {@code AndroidJUnitRunner} executes all JUnit Jupiter test classes (including package-private
 * classes such as {@code ai.pipestream.turboembed.jni.TurboJniTest}) through {@link AndroidJupiterRunner}.
 */
public final class AndroidJupiterRunnerBuilder extends RunnerBuilder {
    /** Public no-arg constructor required by {@code AndroidRunnerBuilder}. */
    public AndroidJupiterRunnerBuilder() {}

    @Override
    public Runner runnerForClass(Class<?> testClass) {
        if (!JupiterTestRunner.hasTests(testClass)) {
            return null;
        }
        return new AndroidJupiterRunner(testClass);
    }
}
