package ai.pipestream.turboembed.panama;

import ai.pipestream.turboembed.TurboBridge;
import ai.pipestream.turboembed.TurboRuntime;
import java.nio.file.Path;

/** Factory for the Panama implementation of the shared Java API. */
public final class PanamaRuntime implements TurboRuntime {
    @Override
    public TurboBridge kind() { return TurboBridge.PANAMA; }

    @Override
    public TurboAnalyzer createAnalyzer(ai.pipestream.turboembed.TurboAnalyzer.Config config) {
        return TurboAnalyzer.create(config);
    }

    @Override
    public TurboSession openSession(Path bundleDir) { return TurboSession.open(bundleDir); }
}
