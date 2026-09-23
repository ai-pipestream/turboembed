package ai.pipestream.turbo;

/**
 * A failed libturbo call: the graded status code, the 1-based index of the
 * offending descriptor field (0 when the error is not about one field), and
 * the library's message. Codes are the {@code TURBO_E_*} constants.
 */
public final class TurboException extends RuntimeException {
    private final int code;
    private final int field;

    public TurboException(int code, int field, String message) {
        super(Native.statusName(code) + (field != 0 ? " (field " + field + ")" : "") + ": " + message);
        this.code = code;
        this.field = field;
    }

    /** Status code ({@code TURBO_E_*}). */
    public int code() {
        return code;
    }

    /** 1-based descriptor field index, or 0. */
    public int field() {
        return field;
    }

    /** Symbolic name of the status code, for example {@code TURBO_E_BUSY}. */
    public String statusName() {
        return Native.statusName(code);
    }
}
