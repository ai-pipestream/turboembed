// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.RerankOptions;
import ai.pipestream.turbo.Truncate;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 * Per-call rerank options ({@code turbo_rerank_options}). {@code topN} and
 * {@code returnSorted} are gated by {@code TURBO_CAP_OPT_TOP_N} and
 * {@code rawScores} by {@code TURBO_CAP_OPT_RAW_SCORES}; a device without the
 * bit refuses the call naming the field rather than ignoring it.
 */
@Schema(name = "RerankOptions", description = "Per-call rerank options; anything omitted is the bundle's contract")
public record RerankOptionsDto(
        @Schema(description = "Truncation policy", example = "RIGHT") Truncate truncate,
        @Schema(description = "Token budget per pair; 0 is the bundle's max_seq", example = "64") Integer maxTokens,
        @Schema(description = "Keep only the best n documents in the ranking; 0 keeps all", example = "3") Integer topN,
        @Schema(description = "Also return the document indexes ordered best first", example = "true") Boolean returnSorted,
        @Schema(description = "Return raw logits instead of the bundle's activation", example = "false") Boolean rawScores) {

    /** The defaults: the bundle contract, scores in input order. */
    public static RerankOptionsDto defaults() {
        return new RerankOptionsDto(null, null, null, null, null);
    }

    /** The binding's option record. */
    public RerankOptions toTurbo() {
        return new RerankOptions(
                truncate == null ? Truncate.MODEL : truncate,
                maxTokens == null ? 0 : maxTokens,
                topN == null ? 0 : topN,
                returnSorted != null && returnSorted,
                rawScores != null && rawScores);
    }
}
