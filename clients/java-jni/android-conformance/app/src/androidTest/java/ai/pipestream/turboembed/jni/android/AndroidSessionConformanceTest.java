package ai.pipestream.turboembed.jni.android;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.SessionConformance;
import ai.pipestream.turboembed.jni.JniRuntime;
import org.junit.runner.RunWith;

/** Runs every shared {@link SessionConformance} assertion on ART. */
@RunWith(AndroidJupiterRunner.class)
public final class AndroidSessionConformanceTest extends SessionConformance {
    static {
        AndroidConformanceSetup.ensureReady();
    }

    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
