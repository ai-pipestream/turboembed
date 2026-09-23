// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

/** A libturbo failure surfaced through the JNI shim: what was called, the status, the field, the message. */
public final class TurboException extends RuntimeException {
    private final int code;
    private final String statusName;
    private final int field;

    public TurboException(String what, int code, String statusName, int field, String message) {
        super(what + ": " + statusName + (field != 0 ? " (field " + field + ")" : "") + ": " + message);
        this.code = code;
        this.statusName = statusName;
        this.field = field;
    }

    public int code() {
        return code;
    }

    public String statusName() {
        return statusName;
    }

    public int field() {
        return field;
    }
}
