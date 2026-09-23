// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.TurboException;
import org.springframework.http.HttpStatus;

/**
 * How a libturbo status code becomes an HTTP status. The rule is who can fix
 * it: a malformed request is 400, a request the model's contract cannot hold
 * is 422, an option this device does not implement is 501, a full queue is
 * 503, and everything else is the server's fault and is 500. The status name
 * and the 1-based field index always travel with the body, so a caller can act
 * on the refusal instead of guessing.
 */
public final class TurboStatus {
    private TurboStatus() {}

    /** The HTTP status for a libturbo refusal. */
    public static HttpStatus of(TurboException e) {
        return switch (e.statusName()) {
            case "TURBO_E_INVALID_ARGUMENT", "TURBO_E_INVALID_UTF8", "TURBO_E_INVALID_ENUM", "TURBO_E_INVALID_SHAPE",
                    "TURBO_E_INVALID_STRUCT_SIZE" ->
                HttpStatus.BAD_REQUEST;
            case "TURBO_E_CAPACITY" -> HttpStatus.UNPROCESSABLE_ENTITY;
            case "TURBO_E_UNSUPPORTED", "TURBO_E_UNSUPPORTED_OPTION", "TURBO_E_UNSUPPORTED_TASK",
                    "TURBO_E_UNSUPPORTED_DTYPE", "TURBO_E_UNSUPPORTED_PLACEMENT", "TURBO_E_UNSUPPORTED_MODALITY",
                    "TURBO_E_NOT_IMPLEMENTED" ->
                HttpStatus.NOT_IMPLEMENTED;
            case "TURBO_E_BUSY", "TURBO_E_OVERLOADED", "TURBO_E_DEVICE_UNAVAILABLE" -> HttpStatus.SERVICE_UNAVAILABLE;
            case "TURBO_E_BUNDLE_NOT_FOUND", "TURBO_E_BUNDLE_INVALID", "TURBO_E_BUNDLE_INTEGRITY",
                    "TURBO_E_BUNDLE_NO_ARTIFACT" ->
                HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.INTERNAL_SERVER_ERROR;
        };
    }
}
