package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.conformance.AnalyzerConformance;
import ai.pipestream.turboembed.conformance.Binding;

/** The shared analyzer conformance suite over this binding. */
final class PanamaAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected Binding binding() {
        return PanamaBinding.INSTANCE;
    }
}
