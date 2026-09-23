// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.Aggregation;
import ai.pipestream.turbo.ClassifyOptions;
import ai.pipestream.turbo.Truncate;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 * Per-call classification options ({@code turbo_classify_options}).
 * {@code aggregation} is gated by {@code TURBO_CAP_OPT_AGGREGATION} and
 * {@code rawScores} by {@code TURBO_CAP_OPT_RAW_SCORES}.
 */
@Schema(name = "ClassifyOptions", description = "Per-call classification options; anything omitted is the bundle's contract")
public record ClassifyOptionsDto(
        @Schema(description = "Truncation policy", example = "RIGHT") Truncate truncate,
        @Schema(description = "Token budget per text; 0 is the bundle's max_seq", example = "64") Integer maxTokens,
        @Schema(description = "How token labels are aggregated into spans (token classification only)", example = "SIMPLE")
        Aggregation aggregation,
        @Schema(description = "Return raw logits instead of the bundle's activation", example = "false") Boolean rawScores) {

    /** The defaults: the bundle contract. */
    public static ClassifyOptionsDto defaults() {
        return new ClassifyOptionsDto(null, null, null, null);
    }

    /** The binding's option record. */
    public ClassifyOptions toTurbo() {
        return new ClassifyOptions(
                truncate == null ? Truncate.MODEL : truncate,
                maxTokens == null ? 0 : maxTokens,
                aggregation == null ? Aggregation.MODEL : aggregation,
                rawScores != null && rawScores);
    }
}
