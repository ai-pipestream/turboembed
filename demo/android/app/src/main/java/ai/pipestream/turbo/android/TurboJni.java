// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

/**
 * The JNI surface of {@code libturbo_jni.so} (jni/turbo_jni.c). Handles are
 * opaque longs; use {@link TurboEngine} rather than these directly.
 */
final class TurboJni {
    private TurboJni() {}

    static native long open(String bundle, String providerLib, int maxBatch);

    static native String describe(long handle);

    static native int dim(long handle);

    static native float[] embed(long handle, String[] texts);

    static native void close(long handle);
}
