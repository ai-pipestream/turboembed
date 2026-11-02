package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.conformance.AnalyzerConformance;
import ai.pipestream.turboembed.conformance.Binding;

/** The shared analyzer conformance suite over this binding. */
final class JniAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected Binding binding() {
        return JniBinding.INSTANCE;
    }
}
