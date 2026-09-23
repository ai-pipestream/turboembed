// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.TokenizerInfo;
import ai.pipestream.turbo.web.LoadedModel;
import ai.pipestream.turbo.web.TurboService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import java.util.ArrayList;
import java.util.List;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * The tokenizer a bundle declares, on its own: text to ids and back.
 *
 * <p>The tokenizer comes from the bundle's {@code tokenizer.json} entry, which
 * is hash-verified on load, so the ids here are the ids the model sees. A
 * bundle that declares no {@code tokenizer.json} (the mock bundles do not)
 * fails with the library's own {@code TURBO_E_BUNDLE_INVALID} unless the model
 * is configured with a {@code tokenizer-bundle} that carries one.
 */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Tokenize", description = "The bundle's own tokenizer: text to ids and back")
public class TokenizeController {
    /** Texts to encode. */
    @Schema(name = "TokenizeRequest", description = "Texts to encode with a model's tokenizer")
    public record TokenizeRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first model loaded",
                    example = "mock-embedding")
            String model,
            @Schema(description = "Texts to encode", example = "[\"a brown dog runs through the grass\"]")
            @NotEmpty(message = "texts must not be empty") List<String> texts,
            @Schema(description = "Add the tokenizer's special tokens; defaults to true", example = "true")
            Boolean addSpecialTokens) {}

    /** One encoded text. */
    @Schema(name = "TokenizedText", description = "One text's ids, attention mask and decoded pieces")
    public record TokenizedText(int index, String text, int count, List<Integer> ids, List<Integer> mask,
            List<String> tokens) {}

    /** The ids, with the tokenizer's own identity. */
    @Schema(name = "TokenizeResponse", description = "One encoded row per text, with the tokenizer's identity")
    public record TokenizeResponse(String model, String tokenizerBundle, TokenizerInfo tokenizer,
            List<TokenizedText> results, TurboService.Timings timings) {}

    /** Id rows to decode. */
    @Schema(name = "DetokenizeRequest", description = "Id rows to decode back to text")
    public record DetokenizeRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first model loaded",
                    example = "mock-embedding")
            String model,
            @Schema(description = "One row of token ids per text", example = "[[101,1037,2829,3899,102]]")
            @NotEmpty(message = "ids must not be empty") List<List<Integer>> ids,
            @Schema(description = "Drop the special tokens from the decoded text; defaults to true", example = "true")
            Boolean skipSpecialTokens) {}

    /** The decoded texts. */
    @Schema(name = "DetokenizeResponse", description = "One decoded text per id row")
    public record DetokenizeResponse(String model, String tokenizerBundle, List<String> texts,
            TurboService.Timings timings) {}

    private static final String TOKENIZE_EXAMPLE = """
            {"model":"mock-embedding","tokenizer_bundle":"testdata/bundles/minilm-tokenizer",\
            "tokenizer":{"vocab_size":30522,"max_seq":256,"specials_per_sequence":2,"pad_id":0,"kind":"wordpiece"},\
            "results":[{"index":0,"text":"a brown dog","count":5,"ids":[101,1037,2829,3899,102],\
            "mask":[1,1,1,1,1],"tokens":["[CLS]","a","brown","dog","[SEP]"]}],\
            "timings":{"write_ms":0.01,"run_ms":0.43,"read_ms":0.22,"total_ms":0.66}}""";

    private final TurboService turbo;

    public TokenizeController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/tokenize")
    @Operation(summary = "Encode texts with a model's tokenizer",
            description = "Returns each text's token ids, its attention mask, the decoded piece of every id and "
                    + "the live token count. Rows are encoded up to the tokenizer's max_seq and truncated there by "
                    + "the bundle's own truncation policy. Byte offsets are not returned: the Java binding's "
                    + "Tokenizer.encode does not pass an offsets buffer to turbo_tokenizer_encode, so this server "
                    + "has none to report and does not invent any.")
    @ApiResponse(responseCode = "200", description = "The ids",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = TokenizeResponse.class),
                    examples = @ExampleObject(name = "one sentence through the MiniLM tokenizer", value = TOKENIZE_EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Empty batch or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "404", description = "No model is served under that name",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "422", description = "A text exceeds the tokenizer's budget with truncation off",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "500", description = "The bundle declares no tokenizer.json (TURBO_E_BUNDLE_INVALID)",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public TokenizeResponse tokenize(@Valid @RequestBody TokenizeRequest req) {
        LoadedModel m = model(req.model());
        List<String> texts = List.copyOf(req.texts());
        boolean specials = req.addSpecialTokens() == null || req.addSpecialTokens();
        TurboService.TokenizeResult r = turbo.tokenize(m, texts, specials);
        List<TokenizedText> results = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            TurboService.TokenizeRow row = r.rows().get(i);
            results.add(new TokenizedText(i, texts.get(i), row.count(), boxed(row.ids()), boxed(row.mask()),
                    row.tokens()));
        }
        return new TokenizeResponse(r.model(), r.tokenizerBundle(), r.tokenizer(), results, r.timings());
    }

    @PostMapping("/detokenize")
    @Operation(summary = "Decode id rows back to text",
            description = "The inverse of /api/v1/tokenize, through the same tokenizer. An id outside the "
                    + "vocabulary is the tokenizer's own refusal, not a silently dropped token.")
    @ApiResponse(responseCode = "200", description = "The decoded texts",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = DetokenizeResponse.class),
                    examples = @ExampleObject(name = "one row",
                            value = "{\"model\":\"mock-embedding\",\"texts\":[\"a brown dog\"],"
                                    + "\"timings\":{\"total_ms\":0.11}}")))
    @ApiResponse(responseCode = "400", description = "Empty ids or a null id",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "404", description = "No model is served under that name",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public DetokenizeResponse detokenize(@Valid @RequestBody DetokenizeRequest req) {
        LoadedModel m = model(req.model());
        List<int[]> rows = new ArrayList<>(req.ids().size());
        for (int r = 0; r < req.ids().size(); r++) {
            List<Integer> row = req.ids().get(r);
            if (row == null || row.isEmpty()) {
                throw new IllegalArgumentException("ids[" + r + "] is empty");
            }
            int[] out = new int[row.size()];
            for (int i = 0; i < out.length; i++) {
                Integer id = row.get(i);
                if (id == null) {
                    throw new IllegalArgumentException("ids[" + r + "][" + i + "] is null");
                }
                out[i] = id;
            }
            rows.add(out);
        }
        boolean skip = req.skipSpecialTokens() == null || req.skipSpecialTokens();
        TurboService.DetokenizeResult r = turbo.detokenize(m, rows, skip);
        return new DetokenizeResponse(r.model(), r.tokenizerBundle(), r.texts(), r.timings());
    }

    /** The named model, or the first one loaded; every model has a tokenizer bundle. */
    private LoadedModel model(String name) {
        if (name != null && !name.isBlank()) {
            return turbo.model(name);
        }
        List<LoadedModel> loaded = turbo.loaded();
        if (loaded.isEmpty()) {
            throw new TurboService.TaskNotServed("this server has no loaded model to tokenize with");
        }
        return loaded.get(0);
    }

    private static List<Integer> boxed(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int v : values) {
            out.add(v);
        }
        return out;
    }
}
