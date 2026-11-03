package ai.pipestream.turboembed;

import java.nio.file.Path;

/** A factory for one native bridge, without a dependency on either implementation. */
public interface TurboRuntime {
    /** The chosen transport. Obtaining a factory does not load the native library. */
    TurboBridge kind();

    /** Creates an analyzer. The caller owns and closes it. */
    TurboAnalyzer createAnalyzer(TurboAnalyzer.Config config);

    /** Opens an analyzer for a language and optional bundle. */
    default TurboAnalyzer openAnalyzer(String language, Path bundlePath) {
        return createAnalyzer(new TurboAnalyzer.Config(language, bundlePath));
    }

    /** Opens a CPU embedding or reranking session. The caller owns and closes it. */
    TurboSession openSession(Path bundleDir);

    /**
     * Selects Panama on Java 25+ with the foreign API, JNI otherwise or on Android.
     * The {@code turbo.bridge} property accepts {@code auto}, {@code jni}, or {@code panama}.
     * A missing selected implementation is an error, not a silent transport change.
     */
    static TurboRuntime load() {
        return RuntimeLoader.load(RuntimeLoader.selected(System.getProperty("turbo.bridge", "auto")));
    }

    /** Loads exactly the requested implementation, overriding the system property. */
    static TurboRuntime load(TurboBridge bridge) {
        return RuntimeLoader.load(bridge);
    }
}
