package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.AnalyzerConformance;

/** The shared analyzer conformance suite over this binding. */
final class JniAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
