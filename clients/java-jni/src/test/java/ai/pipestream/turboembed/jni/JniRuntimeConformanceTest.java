package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboBridge;
import ai.pipestream.turboembed.conformance.RuntimeConformance;

final class JniRuntimeConformanceTest extends RuntimeConformance {
    @Override
    protected TurboBridge bridge() { return TurboBridge.JNI; }
}
