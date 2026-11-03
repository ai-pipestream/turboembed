package ai.pipestream.turboembed.jni.android;

import ai.pipestream.turboembed.conformance.AnalyzerConformance;
import ai.pipestream.turboembed.conformance.Binding;
import ai.pipestream.turboembed.jni.JniBinding;
import org.junit.jupiter.api.extension.ExtendWith;

/** Analyzer conformance on ART through the JNI binding and {@code System.loadLibrary("turbo")}. */
@ExtendWith(AndroidConformanceSetup.class)
final class AndroidAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected Binding binding() {
        return JniBinding.INSTANCE;
    }
}
