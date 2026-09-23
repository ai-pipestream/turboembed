package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.TurboNative;

/**
 * Prompt prefix role from the bundle contract. Each constant carries the {@code uint32_t} value the ABI defines;
 * {@link #of(int)} refuses a value the library did not define.
 */
public enum PromptRole {
    NONE(TurboNative.TURBO_PROMPT_NONE()),
    QUERY(TurboNative.TURBO_PROMPT_QUERY()),
    DOCUMENT(TurboNative.TURBO_PROMPT_DOCUMENT());

    private final int value;

    PromptRole(int value) {
        this.value = value;
    }

    /** The ABI constant. */
    public int value() {
        return value;
    }

    /** The constant for an ABI value. */
    public static PromptRole of(int v) {
        for (PromptRole k : values()) {
            if (k.value == v) {
                return k;
            }
        }
        throw new IllegalArgumentException("prompt role value " + v + " is not a known constant");
    }
}
