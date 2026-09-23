// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import io.swagger.v3.oas.annotations.media.Schema;

/**
 * The body of every refusal under {@code /api/v1}. {@code status} and
 * {@code field} are present only when libturbo itself refused the call:
 * {@code status} is the {@code TURBO_E_*} name and {@code field} is the
 * 1-based index of the descriptor field the library named.
 */
@Schema(name = "ApiError", description = "A refusal, with the library's status name and field index when libturbo refused")
public record ApiError(
        @Schema(description = "What was wrong, in the library's own words when it refused",
                example = "TURBO_E_UNSUPPORTED_OPTION (field 6): mock device 1 does not honor option `pooling`")
        String error,
        @Schema(description = "The TURBO_E_* status name, when libturbo refused", example = "TURBO_E_UNSUPPORTED_OPTION")
        String status,
        @Schema(description = "1-based descriptor field index the library named, when it named one", example = "6")
        Integer field,
        @Schema(description = "The request path that was refused", example = "/api/v1/embed")
        String path) {}
