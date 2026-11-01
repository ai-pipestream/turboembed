package ai.pipestream.turboembed.jni;

import java.util.Objects;

/**
 * Unchecked exception thrown when a {@code libturbo} C ABI call returns a non-zero status code.
 */
public final class TurboException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private final int code;
    private final String statusName;
    private final int field;

    /**
     * Constructs a {@code TurboException} from a C ABI error result.
     *
     * @param code       status code returned by the C ABI
     * @param statusName symbolic constant name from {@code turbo_status_name}
     * @param field      1-based option field index, or 0 if not field-specific
     * @param message    UTF-8 diagnostic message from {@code turbo_error}
     */
    public TurboException(int code, String statusName, int field, String message) {
        super(formatMessage(code, statusName, field, message));
        this.code = code;
        this.statusName = Objects.requireNonNull(statusName, "statusName");
        this.field = field;
    }

    private static String formatMessage(int code, String statusName, int field, String message) {
        String detail = (message == null || message.isEmpty()) ? "no detail" : message;
        if (field > 0) {
            return statusName + " (" + code + ", field=" + field + "): " + detail;
        }
        return statusName + " (" + code + "): " + detail;
    }

    /**
     * Returns the numeric {@code TURBO_E_*} status code.
     *
     * @return numeric status code
     */
    public int code() {
        return code;
    }

    /**
     * Returns the symbolic constant name (for example {@code "TURBO_E_INVALID_ARGUMENT"}).
     *
     * @return status constant name
     */
    public String statusName() {
        return statusName;
    }

    /**
     * Returns the 1-based struct field index associated with the error, or {@code 0}.
     *
     * @return 1-based field index or 0
     */
    public int field() {
        return field;
    }
}
