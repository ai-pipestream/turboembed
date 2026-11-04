package ai.pipestream.turboembed.jni.android;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.AnalyzerConformance;
import ai.pipestream.turboembed.jni.JniRuntime;
import org.junit.runner.RunWith;

/**
 * Runs every shared {@link AnalyzerConformance} assertion on the Android runtime (ART) against the
 * {@code libturbo.so} packaged in {@code jniLibs} and the bundles unpacked from {@code assets}.
 */
@RunWith(AndroidJupiterRunner.class)
public final class AndroidAnalyzerConformanceTest extends AnalyzerConformance {
    static {
        AndroidConformanceSetup.ensureReady();
    }

    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
