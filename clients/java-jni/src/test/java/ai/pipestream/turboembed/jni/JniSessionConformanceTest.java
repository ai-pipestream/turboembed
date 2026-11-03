package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.conformance.SessionConformance;

/** The shared session conformance suite over this binding. */
final class JniSessionConformanceTest extends SessionConformance {
    @Override
    protected TurboRuntime binding() {
        return new JniRuntime();
    }
}
