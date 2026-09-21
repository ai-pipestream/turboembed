package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Task a session or capability cell refers to. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum Task {
    EMBED(TurboNative.TURBO_TASK_EMBED()),
    RERANK(TurboNative.TURBO_TASK_RERANK()),
    CLASSIFY(TurboNative.TURBO_TASK_CLASSIFY()),
    TOKEN_CLASSIFY(TurboNative.TURBO_TASK_TOKEN_CLASSIFY()),
    GENERATE(TurboNative.TURBO_TASK_GENERATE()),
    TOKENIZE(TurboNative.TURBO_TASK_TOKENIZE()),
    RUN(TurboNative.TURBO_TASK_RUN()),
    CHUNK(TurboNative.TURBO_TASK_CHUNK());

    private final int value;

    Task(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static Task of(int v) {
        for (Task k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("task value " + v + " is not a known constant");
    }
}
