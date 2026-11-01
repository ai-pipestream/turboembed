package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.conformance.Binding;
import ai.pipestream.turboembed.conformance.SessionConformance;

/** The shared session conformance suite over this binding. */
final class JniSessionConformanceTest extends SessionConformance {
    @Override
    protected Binding binding() {
        return JniBinding.INSTANCE;
    }
}
