package ai.pipestream.turboembed.jni.android;

import android.content.res.AssetManager;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/** {@link RepoRootStaging.FileSource} over the instrumentation APK's asset tree. */
final class AssetFileSource implements RepoRootStaging.FileSource {
    private final AssetManager assets;

    AssetFileSource(AssetManager assets) {
        this.assets = Objects.requireNonNull(assets, "assets");
    }

    @Override
    public InputStream open(String relativePath) throws IOException {
        return assets.open(relativePath);
    }
}
