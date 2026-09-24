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
import java.util.List;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** Embedding: vectors for a batch of texts, and the cosine matrix over them. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Embed", description = "Vectors for a batch of texts, and their cosine similarities")
public class EmbedController {
    /** Texts to embed, the model to use, and the per-call options. */
    @Schema(name = "EmbedRequest", description = "Texts to embed on an embedding model")
    public record EmbedRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first embedding model loaded",
                    example = "mock-embedding")
            String model,
            @Schema(description = "One text per vector; at most the model's max_batch",
                    example = "[\"a brown dog runs through the grass\",\"a dog is running on the lawn\"]")
            @NotEmpty(message = "texts must not be empty")
            List<@NotBlank(message = "texts must not contain a blank string") String> texts,
            @Valid EmbedOptionsDto options) {}

    /** Vectors, where they were produced, and how long each phase took. */
    @Schema(name = "EmbedResponse", description = "One vector per text, with the device and the timings")
    public record EmbedResponse(String model, TurboService.DeviceRef device, int dim, int count, List<String> texts,
            float[][] vectors, String placement, TurboService.Timings timings) {}

    /** Vectors plus the full cosine matrix the heat map draws. */
    @Schema(name = "SimilarityResponse", description = "The cosine matrix over the embedded texts")
    public record SimilarityResponse(String model, TurboService.DeviceRef device, int dim, int count,
            List<String> texts, float[][] vectors, double[][] similarity, String placement,
            TurboService.Timings timings) {}

    private static final String EMBED_EXAMPLE = """
            {"model":"mock-embedding","device":{"index":1,"name":"Mock accelerator","kind":"ACCEL",\
            "provider_id":"mock","ordinal":1,"runtime_version":"mock"},"dim":8,"count":2,\
            "texts":["a brown dog runs through the grass","a dog is running on the lawn"],\
            "vectors":[[0.35,-0.12,0.44,0.02,-0.31,0.51,-0.18,0.53],[0.31,-0.09,0.47,0.06,-0.28,0.55,-0.14,0.50]],\
            "placement":"HOST","timings":{"write_ms":0.12,"run_ms":0.31,"read_ms":0.04,"total_ms":0.47}}""";

    private final TurboService turbo;

    public EmbedController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/embed")
    @Operation(summary = "Embed a batch of texts",
            description = "Writes the texts into a session of the named model and runs it. Options are honored "
                    + "exactly: a device that does not implement one refuses the call with its TURBO_E_* code and "
                    + "the 1-based field index, which this server returns as 501. A batch wider than the model's "
                    + "max_batch is 400 before any work is done.")
    @ApiResponse(responseCode = "200", description = "The vectors",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = EmbedResponse.class),
                    examples = @ExampleObject(name = "two sentences on the mock bundle", value = EMBED_EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Empty batch, a batch above max_batch, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "422", description = "The request exceeds the model's contract (TURBO_E_CAPACITY)",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "501", description = "The device does not implement an option that was set",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "503", description = "Every session of the model is busy",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public EmbedResponse embed(@Valid @RequestBody EmbedRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.EMBED);
        List<String> texts = List.copyOf(req.texts());
        TurboService.EmbedResult r = turbo.embed(m, texts, options(req.options()).toTurbo());
        return new EmbedResponse(r.model(), r.device(), r.dim(), texts.size(), texts, r.vectors(), r.placement(),
                r.timings());
    }

    @PostMapping("/similarity")
    @Operation(summary = "Embed a batch of texts and return their cosine matrix",
            description = "The same call as /api/v1/embed plus the n by n cosine matrix over the vectors it "
                    + "produced. The matrix is computed on this server from the vectors the device returned, so a "
                    + "diagonal of 1.000 and symmetry are checks on the vectors, not on the arithmetic here.")
    @ApiResponse(responseCode = "200", description = "The vectors and the matrix",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = SimilarityResponse.class),
                    examples = @ExampleObject(name = "two sentences on the mock bundle",
                            value = "{\"model\":\"mock-embedding\",\"dim\":8,\"count\":2,"
                                    + "\"texts\":[\"a brown dog runs through the grass\",\"a dog is running on the lawn\"],"
                                    + "\"similarity\":[[1.0,0.83],[0.83,1.0]],\"placement\":\"HOST\"}")))
    @ApiResponse(responseCode = "400", description = "Empty batch, a batch above max_batch, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public SimilarityResponse similarity(@Valid @RequestBody EmbedRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.EMBED);
        List<String> texts = List.copyOf(req.texts());
        TurboService.EmbedResult r = turbo.embed(m, texts, options(req.options()).toTurbo());
        return new SimilarityResponse(r.model(), r.device(), r.dim(), texts.size(), texts, r.vectors(),
                TurboService.similarity(r.vectors()), r.placement(), r.timings());
    }

    private static EmbedOptionsDto options(EmbedOptionsDto o) {
        return o == null ? EmbedOptionsDto.defaults() : o;
    }
}
