package ai.pipestream.turboembed.jni;

import ai.pipestream.turboembed.TurboBridge;
import ai.pipestream.turboembed.TurboRuntime;
import ai.pipestream.turboembed.TurboSessionOptions;
import java.nio.file.Path;

/** Factory for the Jni implementation of the shared Java API. */
public final class JniRuntime implements TurboRuntime {
    @Override
    public TurboBridge kind() { return TurboBridge.JNI; }

    @Override
    public TurboAnalyzer createAnalyzer(ai.pipestream.turboembed.TurboAnalyzer.Config config) {
        return TurboAnalyzer.create(config);
    }

    @Override
    public TurboSession openSession(Path bundleDir, TurboSessionOptions options) {
        return TurboSession.open(bundleDir, options);
    }
}
