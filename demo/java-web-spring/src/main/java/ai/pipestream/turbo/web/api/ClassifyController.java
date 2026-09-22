// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.Task;
import ai.pipestream.turbo.web.LoadedModel;
import ai.pipestream.turbo.web.TurboService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.ArrayList;
import java.util.List;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** Sequence classification and token classification, on bundles that declare labels. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Classify", description = "Sequence labels and token spans, on bundles that declare a label set")
public class ClassifyController {
    /** Texts to label and the per-call options. */
    @Schema(name = "ClassifyRequest", description = "Texts to label on a classifier or token classifier")
    public record ClassifyRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first model of that task",
                    example = "mock-classifier")
            String model,
            @Schema(description = "One text per row; at most the model's max_batch",
                    example = "[\"the service was excellent\",\"the parcel never arrived\"]")
            @NotEmpty(message = "texts must not be empty")
            List<@NotBlank(message = "texts must not contain a blank string") String> texts,
            @Valid ClassifyOptionsDto options) {}

    /** One label and its score. */
    @Schema(name = "LabelScore", description = "One label of the bundle's label set, with its score")
    public record LabelScore(String label, float score) {}

    /** One input text's labels, best first. */
    @Schema(name = "ClassifyRow", description = "The labels of one input text, ordered best first")
    public record ClassifyRow(int index, String text, String top, List<LabelScore> scores) {}

    /** Scores for every text, with the label set the bundle declares. */
    @Schema(name = "ClassifyResponse", description = "One labelled row per text")
    public record ClassifyResponse(String model, TurboService.DeviceRef device, List<String> labels,
            List<ClassifyRow> results, String placement, TurboService.Timings timings) {}

    /** One aggregated span of one input text. */
    @Schema(name = "TokenSpan", description = "One aggregated span: byte offsets into its input text, the label and the score")
    public record TokenSpan(int row, long byteStart, long byteEnd, String text, String label, int labelIndex, float score) {}

    /** Spans for every text, with the raw score tensor's shape. */
    @Schema(name = "TokenClassifyResponse", description = "The spans the model found, per input text")
    public record TokenClassifyResponse(String model, TurboService.DeviceRef device, List<String> labels,
            List<TokenSpan> spans, List<Long> scoreShape, String placement, TurboService.Timings timings) {}

    private static final String CLASSIFY_EXAMPLE = """
            {"model":"mock-classifier","device":{"index":1,"name":"Mock accelerator","provider_id":"mock","ordinal":1},\
            "labels":["negative","neutral","positive"],"results":[{"index":0,"text":"the service was excellent",\
            "top":"positive","scores":[{"label":"positive","score":0.71},{"label":"neutral","score":0.2},\
            {"label":"negative","score":0.09}]}],"placement":"HOST",\
            "timings":{"write_ms":0.08,"run_ms":0.19,"read_ms":0.02,"total_ms":0.29}}""";

    private static final String TOKEN_EXAMPLE = """
            {"model":"mock-token_classifier","labels":["O","PER","LOC"],"spans":[{"row":0,"byte_start":0,\
            "byte_end":4,"text":"Ada","label":"PER","label_index":1,"score":0.9}],"score_shape":[1,16,3],\
            "placement":"HOST","timings":{"write_ms":0.07,"run_ms":0.21,"read_ms":0.03,"total_ms":0.31}}""";

    private final TurboService turbo;

    public ClassifyController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/classify")
    @Operation(summary = "Label a batch of texts",
            description = "Runs a classifier bundle and returns the bundle's own labels with their scores, best "
                    + "first. raw_scores returns the logits instead of the bundle's activation, and is refused "
                    + "naming the field on a device without TURBO_CAP_OPT_RAW_SCORES.")
    @ApiResponse(responseCode = "200", description = "The labels",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ClassifyResponse.class),
                    examples = @ExampleObject(name = "one sentence on the mock classifier", value = CLASSIFY_EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Empty batch, a batch above max_batch, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "409", description = "No loaded model performs CLASSIFY",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "501", description = "The device does not implement an option that was set",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public ClassifyResponse classify(@Valid @RequestBody ClassifyRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.CLASSIFY);
        List<String> texts = List.copyOf(req.texts());
        ClassifyOptionsDto opts = req.options() == null ? ClassifyOptionsDto.defaults() : req.options();
        TurboService.ClassifyResult r = turbo.classify(m, texts, opts.toTurbo());
        List<ClassifyRow> rows = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            float[] scores = r.scores()[i];
            List<LabelScore> labelled = new ArrayList<>(scores.length);
            for (int l = 0; l < scores.length; l++) {
                labelled.add(new LabelScore(l < r.labels().size() ? r.labels().get(l) : "label_" + l, scores[l]));
            }
            List<LabelScore> ordered = labelled.stream()
                    .sorted((a, b) -> Float.compare(b.score(), a.score()))
                    .toList();
            rows.add(new ClassifyRow(i, texts.get(i), ordered.isEmpty() ? null : ordered.get(0).label(), ordered));
        }
        return new ClassifyResponse(r.model(), r.device(), r.labels(), rows, r.placement(), r.timings());
    }

    @PostMapping("/token-classify")
    @Operation(summary = "Find labelled spans in a batch of texts",
            description = "Runs a token classifier bundle and returns the spans the provider aggregated, with "
                    + "byte offsets into the input text they came from. aggregation picks how token labels become "
                    + "spans and is refused naming the field on a device without TURBO_CAP_OPT_AGGREGATION.")
    @ApiResponse(responseCode = "200", description = "The spans",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = TokenClassifyResponse.class),
                    examples = @ExampleObject(name = "one sentence on the mock token classifier", value = TOKEN_EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Empty batch, a batch above max_batch, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "409", description = "No loaded model performs TOKEN_CLASSIFY",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "501", description = "The device does not implement an option that was set",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public TokenClassifyResponse tokenClassify(@Valid @RequestBody ClassifyRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.TOKEN_CLASSIFY);
        List<String> texts = List.copyOf(req.texts());
        ClassifyOptionsDto opts = req.options() == null ? ClassifyOptionsDto.defaults() : req.options();
        TurboService.TokenClassifyResult r = turbo.tokenClassify(m, texts, opts.toTurbo());
        List<TokenSpan> spans = r.spans().stream()
                .map(s -> new TokenSpan(s.row(), s.byteStart(), s.byteEnd(), s.text(), s.label(), s.labelIndex(), s.score()))
                .toList();
        List<Long> shape = new ArrayList<>(r.scoreShape().length);
        for (long d : r.scoreShape()) {
            shape.add(d);
        }
        return new TokenClassifyResponse(r.model(), r.device(), r.labels(), spans, shape, r.placement(), r.timings());
    }
}
