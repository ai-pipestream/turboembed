// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

/**
 * The JNI surface of {@code libturbo_jni.so} (jni/turbo_jni.c). Handles are
 * opaque longs; use {@link TurboEngine} rather than these directly.
 *
 * <p>Text crosses as UTF-8 {@code byte[]}, not {@code String}: JNI hands C
 * modified UTF-8, which the library rejects as invalid UTF-8 for any
 * supplementary character or embedded NUL.
 */
final class TurboJni {
    private TurboJni() {}

    static native long open(byte[] bundle, byte[] providerLib, int maxBatch);

    static native String describe(long handle);

    static native int dim(long handle);

    static native float[] embed(long handle, byte[][] texts);

    static native void close(long handle);
}
