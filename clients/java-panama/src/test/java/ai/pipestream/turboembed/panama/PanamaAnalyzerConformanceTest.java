package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.AnalyzerConformance;

/** The shared analyzer conformance suite over this binding. */
final class PanamaAnalyzerConformanceTest extends AnalyzerConformance {
    @Override
    protected TurboRuntime binding() {
        return new PanamaRuntime();
    }
}
