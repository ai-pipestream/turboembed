package ai.pipestream.turboembed.jni.android;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.AnalyzerConformance;
import ai.pipestream.turboembed.jni.JniRuntime;
import org.junit.jupiter.api.extension.ExtendWith;

/** Analyzer conformance on ART through the JNI binding and {@code System.loadLibrary("turbo")}. */
@ExtendWith(AndroidConformanceSetup.class)
final class AndroidAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
