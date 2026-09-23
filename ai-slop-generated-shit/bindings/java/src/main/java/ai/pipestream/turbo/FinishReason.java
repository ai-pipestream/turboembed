package ai.pipestream.turbo;

/** Why a generation stopped ({@code TURBO_FINISH_*}). */
public enum FinishReason {
    NONE(0),
    EOS(1),
    STOP(2),
    LENGTH(3),
    CANCELLED(4);

    private final int value;

    FinishReason(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    static FinishReason of(int v) {
        for (FinishReason r : values()) {
            if (r.value == v) {
                return r;
            }
        }
        throw new IllegalStateException("unknown finish reason " + v);
    }
}
