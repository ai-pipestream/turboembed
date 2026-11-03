package ai.pipestream.turboembed.jni;

/**
 * Native-entry-point compatibility name. Catch the shared API exception instead.
 * @deprecated use {@link ai.pipestream.turboembed.TurboException}
 */
@Deprecated(since = "0.2", forRemoval = false)
public final class TurboException extends ai.pipestream.turboembed.TurboException {
    private static final long serialVersionUID = 1L;

    /** Preserves the constructor called by the existing JNI library. */
    public TurboException(int code, String statusName, int field, String message) {
        super(code, statusName, field, message);
    }
}
