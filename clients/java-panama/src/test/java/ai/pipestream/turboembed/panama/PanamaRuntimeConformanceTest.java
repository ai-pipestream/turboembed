package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.TurboBridge;
import ai.pipestream.turboembed.conformance.RuntimeConformance;

final class PanamaRuntimeConformanceTest extends RuntimeConformance {
    @Override
    protected TurboBridge bridge() { return TurboBridge.PANAMA; }
}
