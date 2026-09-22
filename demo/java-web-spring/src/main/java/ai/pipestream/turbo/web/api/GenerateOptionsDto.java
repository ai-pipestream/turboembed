// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.GenerateDesc;
import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import java.util.Map;

/**
 * Generation parameters ({@code turbo_generate_desc}). Each one is gated by a
 * {@code TURBO_CAP_OPT_GEN_*} bit: a device without the bit refuses the call
 * with {@code TURBO_E_UNSUPPORTED_OPTION} and the field index instead of
 * sampling differently from what was asked.
 */
@Schema(name = "GenerateOptions", description = "Generation parameters; anything omitted is the model's default")
public record GenerateOptionsDto(
        @Schema(description = "New tokens to generate at most", example = "128") Integer maxTokens,
        @Schema(description = "New tokens to generate before EOS is allowed", example = "8") Integer minTokens,
        @Schema(description = "Sampling temperature; 0 is greedy", example = "0.7") Float temperature,
        @Schema(description = "Nucleus sampling mass; 0 or 1 is off", example = "0.95") Float topP,
        @Schema(description = "Top-k sampling; 0 is off", example = "40") Integer topK,
        @Schema(description = "Min-p sampling; 0 is off", example = "0.05") Float minP,
        @Schema(description = "Sampling seed; omitted means the model's own", example = "1234") Long seed,
        @Schema(description = "Stop strings", example = "[\"\\n\\n\"]") List<String> stop,
        @Schema(description = "Stop token ids", example = "[2]") List<Integer> stopTokens,
        @Schema(description = "Logprobs to return per token; 0 is none", example = "1") Integer logprobs,
        @Schema(description = "Include the prompt in the generated text", example = "false") Boolean echo) {

    /** The defaults: the model's own for everything. */
    public static GenerateOptionsDto defaults() {
        return new GenerateOptionsDto(null, null, null, null, null, null, null, null, null, null, null);
    }

    /** The binding's descriptor, with the model's default for every field left out. */
    public GenerateDesc toTurbo() {
        int[] tokens = new int[stopTokens == null ? 0 : stopTokens.size()];
        for (int i = 0; i < tokens.length; i++) {
            Integer id = stopTokens.get(i);
            if (id == null) {
                throw new IllegalArgumentException("options.stop_tokens[" + i + "] is null");
            }
            tokens[i] = id;
        }
        List<String> stops = stop == null ? List.of() : List.copyOf(stop);
        for (int i = 0; i < stops.size(); i++) {
            if (stops.get(i) == null || stops.get(i).isEmpty()) {
                throw new IllegalArgumentException("options.stop[" + i + "] is empty");
            }
        }
        return new GenerateDesc(
                maxTokens == null ? 0 : maxTokens,
                minTokens == null ? 0 : minTokens,
                0,
                temperature == null ? 0f : temperature,
                topK == null ? 0 : topK,
                topP == null ? 0f : topP,
                minP == null ? 0f : minP,
                0f,
                0f,
                0f,
                seed,
                stops,
                tokens,
                Map.of(),
                logprobs == null ? 0 : logprobs,
                echo != null && echo);
    }
}
