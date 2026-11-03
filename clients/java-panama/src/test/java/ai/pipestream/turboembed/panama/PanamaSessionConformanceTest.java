package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.SessionConformance;

/** The shared session conformance suite over this binding. */
final class PanamaSessionConformanceTest extends SessionConformance {
    @Override
    protected TurboRuntime binding() {
        return new PanamaRuntime();
    }
}
