// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/** Failure reported by the native TurboEmbed API. */
public final class NativeException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private final int code;

    public NativeException(int code, String message) {
        super(message);
        this.code = code;
    }

    public int code() {
        return code;
    }
}
