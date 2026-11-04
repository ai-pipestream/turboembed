package ai.pipestream.turboembed.jni.android;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Unpacks the APK's {@code testdata/} assets into the application files directory once per process
 * and sets {@code turbo.repo.root} so {@code Testdata.repoRoot()} and {@code TestPaths.repoRoot()}
 * resolve on Android without a host checkout.
 */
final class AndroidConformanceSetup {
    private static final AtomicReference<Path> STAGED_ROOT = new AtomicReference<>();

    private AndroidConformanceSetup() {}

    static Path ensureReady() {
        Path existing = STAGED_ROOT.get();
        if (existing != null) {
            return existing;
        }
        synchronized (AndroidConformanceSetup.class) {
            existing = STAGED_ROOT.get();
            if (existing != null) {
                return existing;
            }
            Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
            Path dest = target.getFilesDir().toPath().resolve("turboembed-repo");
            try {
                Path staged =
                        RepoRootStaging.stage(dest, new AssetFileSource(target.getAssets()));
                System.setProperty("turbo.repo.root", staged.toString());
                STAGED_ROOT.set(staged);
                return staged;
            } catch (IOException ex) {
                throw new IllegalStateException("failed to stage testdata into " + dest, ex);
            }
        }
    }
}
