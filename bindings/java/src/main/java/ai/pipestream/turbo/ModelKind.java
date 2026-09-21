package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Model kind from the bundle manifest. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum ModelKind {
    EMBEDDING(TurboNative.TURBO_MODEL_EMBEDDING()),
    RERANKER(TurboNative.TURBO_MODEL_RERANKER()),
    CLASSIFIER(TurboNative.TURBO_MODEL_CLASSIFIER()),
    TOKEN_CLASSIFIER(TurboNative.TURBO_MODEL_TOKEN_CLASSIFIER()),
    GENERATIVE(TurboNative.TURBO_MODEL_GENERATIVE()),
    GENERIC(TurboNative.TURBO_MODEL_GENERIC());

    private final int value;

    ModelKind(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static ModelKind of(int v) {
        for (ModelKind k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("model kind value " + v + " is not a known constant");
    }
}
