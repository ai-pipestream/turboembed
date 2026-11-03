package ai.pipestream.turboembed.jni.android;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.SessionConformance;
import ai.pipestream.turboembed.jni.JniRuntime;
import org.junit.jupiter.api.extension.ExtendWith;

/** Session conformance on ART through the JNI binding and {@code System.loadLibrary("turbo")}. */
@ExtendWith(AndroidConformanceSetup.class)
final class AndroidSessionConformanceTest extends SessionConformance {
    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
