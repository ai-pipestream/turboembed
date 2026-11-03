package ai.pipestream.turboembed.jni.android;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import ai.pipestream.turboembed.jni.TurboLibrary;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Stages the sealed testdata bundles from APK assets into the application files directory,
 * points {@code turbo.repo.root} at that layout, and loads {@code libturbo} through
 * {@link System#loadLibrary(String)} (the Android {@code jniLibs} path). No
 * {@code turbo.library.path} or {@code turbo.library.dir} is set, so a host-JVM fallback cannot
 * satisfy the load.
 */
public final class AndroidConformanceSetup implements BeforeAllCallback {
    private static final Object LOCK = new Object();
    private static boolean ready;

    @Override
    public void beforeAll(ExtensionContext context) throws Exception {
        ensureReady();
    }

    /** Stages once per process for every instrumented suite that registers this extension. */
    public static void ensureReady() throws Exception {
        synchronized (LOCK) {
            if (ready) {
                return;
            }
            Context app = InstrumentationRegistry.getInstrumentation().getTargetContext();
            Path destination = app.getFilesDir().toPath().resolve("turbo-repo");
            RepoRootStaging.stage(destination, new AssetFileSource(app.getAssets()));
            System.setProperty("turbo.repo.root", destination.toString());
            // Refuse a host-style library path so System.loadLibrary("turbo") is what runs.
            System.clearProperty("turbo.library.path");
            System.clearProperty("turbo.library.dir");
            TurboLibrary.load();
            ready = true;
        }
    }
}
