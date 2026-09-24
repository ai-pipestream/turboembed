// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.EmbedOptions;
import ai.pipestream.turbo.Normalize;
import ai.pipestream.turbo.OutputDType;
import ai.pipestream.turbo.Pooling;
import ai.pipestream.turbo.PromptRole;
import ai.pipestream.turbo.Truncate;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 * Per-call embedding options ({@code turbo_embed_options}). Every field left
 * out means the bundle's own contract. A field the device does not honor is
 * refused with {@code TURBO_E_UNSUPPORTED_OPTION} and the 1-based field index,
 * which this server returns as 501 with both in the body; it is never ignored
 * or clamped.
 */
@Schema(name = "EmbedOptions", description = "Per-call embedding options; anything omitted is the bundle's contract")
public record EmbedOptionsDto(
        @Schema(description = "Truncation policy", example = "RIGHT") Truncate truncate,
        @Schema(description = "Token budget per text; 0 is the bundle's max_seq", example = "128") Integer maxTokens,
        @Schema(description = "Prompt prefix role the bundle declares", example = "QUERY") PromptRole promptRole,
        @Schema(description = "Normalization of the output vectors", example = "L2") Normalize normalize,
        @Schema(description = "Pooling over the hidden states", example = "MEAN") Pooling pooling,
        @Schema(description = "Truncated output dimension; must be one of the bundle's truncate_dims", example = "4")
        Integer outputDim,
        @Schema(description = "Requested output element type", example = "F32") OutputDType outputDtype) {

    /** The defaults: the bundle contract for everything. */
    public static EmbedOptionsDto defaults() {
        return new EmbedOptionsDto(null, null, null, null, null, null, null);
    }

    /** The binding's option record, with the bundle contract for every field left out. */
    public EmbedOptions toTurbo() {
        return new EmbedOptions(
                truncate == null ? Truncate.MODEL : truncate,
                maxTokens == null ? 0 : maxTokens,
                promptRole == null ? PromptRole.NONE : promptRole,
                normalize == null ? Normalize.MODEL : normalize,
                pooling == null ? Pooling.MODEL : pooling,
                outputDim == null ? 0 : outputDim,
                outputDtype == null ? OutputDType.MODEL : outputDtype);
    }
}
