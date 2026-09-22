// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

/**
 * An embedding engine over the JNI shim: a bundle on the best device, with
 * a session of {@code maxBatch} rows. Not thread-safe; one engine per
 * thread or guard it. Loads {@code libturbo_jni.so}, which links
 * {@code libturbo.so}; both ship in the APK's {@code jniLibs}.
 */
public final class TurboEngine implements AutoCloseable {
    static {
        System.loadLibrary("turbo_jni");
    }

    private long handle;
    private final int dim;

    public TurboEngine(String bundlePath, String providerLib, int maxBatch) {
        handle = TurboJni.open(bundlePath, providerLib, maxBatch);
        if (handle == 0) {
            throw new IllegalStateException("engine did not open and no exception was thrown");
        }
        dim = TurboJni.dim(handle);
    }

    public String describe() {
        return TurboJni.describe(handle());
    }

    public int dim() {
        return dim;
    }

    /** One row of {@code dim()} floats per text, in order. */
    public float[][] embed(String... texts) {
        float[] flat = TurboJni.embed(handle(), texts);
        float[][] rows = new float[texts.length][];
        for (int i = 0; i < texts.length; i++) {
            rows[i] = new float[dim];
            System.arraycopy(flat, i * dim, rows[i], 0, dim);
        }
        return rows;
    }

    public static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    private long handle() {
        if (handle == 0) {
            throw new IllegalStateException("engine is closed");
        }
        return handle;
    }

    @Override
    public void close() {
        if (handle != 0) {
            TurboJni.close(handle);
            handle = 0;
        }
    }
}
