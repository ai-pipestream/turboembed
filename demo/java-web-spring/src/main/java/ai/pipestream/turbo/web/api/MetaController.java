// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.web.TurboService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.ArraySchema;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** What this server is, what hardware it found, and what it loaded. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Server", description = "Liveness, the device survey, and the loaded models")
public class MetaController {
    private final TurboService turbo;

    public MetaController(TurboService turbo) {
        this.turbo = turbo;
    }

    @GetMapping("/health")
    @Operation(summary = "Liveness and the shape of the server",
            description = "Answers as soon as the runtime is up and every configured bundle has loaded. "
                    + "A server that could not load a bundle never starts, so a 200 here means every listed model is ready.")
    @ApiResponse(responseCode = "200", description = "The server is up",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = TurboService.HealthReport.class),
                    examples = @ExampleObject(name = "mock bundles",
                            value = "{\"status\":\"ok\",\"abi_version\":2,\"device_count\":2,"
                                    + "\"models\":[\"mock-embedding\",\"mock-generative\"]}")))
    public TurboService.HealthReport health() {
        return turbo.health();
    }

    @GetMapping("/devices")
    @Operation(summary = "Every device of every loaded provider",
            description = "The discover-style survey: one entry per device with its provider, runtime and driver "
                    + "versions, the TURBO_CAP_* option bits it honors, and the full task-by-modality capability "
                    + "matrix with each cell's status, compute dtype and measured precision floor. A cell marked "
                    + "SUPPORTED carries a receipt; PLANNED and EXPERIMENTAL are honest, non-blocking states.")
    @ApiResponse(responseCode = "200", description = "The survey",
            content = @Content(mediaType = "application/json", array = @ArraySchema(
                    schema = @Schema(implementation = TurboService.DeviceReport.class)),
                    examples = @ExampleObject(name = "the mock accelerator",
                            value = "[{\"index\":1,\"name\":\"Mock accelerator\",\"kind\":\"ACCEL\",\"ordinal\":1,"
                                    + "\"vendor\":\"Pipestream\",\"provider_id\":\"mock\",\"runtime_version\":\"mock\","
                                    + "\"caps\":72198606848002,\"features\":[\"HOST_PTR_IMPORT\",\"DYNAMIC_SHAPE\"],"
                                    + "\"capabilities\":[{\"task\":\"EMBED\",\"modality\":\"TEXT\",\"status\":\"SUPPORTED\","
                                    + "\"dtype\":\"F32\",\"cosine_floor\":1.0,\"deterministic\":true}],"
                                    + "\"models\":[\"mock-embedding\"]}]")))
    public List<TurboService.DeviceReport> devices() {
        return turbo.devices();
    }

    @GetMapping("/models")
    @Operation(summary = "Every loaded model and its contract",
            description = "One entry per loaded model with the whole turbo_model_info contract: task, kind, "
                    + "dimension, pooling, normalization, sequence and batch limits, compute dtype, per-stage "
                    + "placement, label set, tokenizer hash and prompt prefixes. These names are what the model "
                    + "field of a request and the {name} of a /v2 path refer to.")
    @ApiResponse(responseCode = "200", description = "The loaded models",
            content = @Content(mediaType = "application/json", array = @ArraySchema(
                    schema = @Schema(implementation = TurboService.ModelReport.class)),
                    examples = @ExampleObject(name = "the mock embedding bundle",
                            value = "[{\"name\":\"mock-embedding\",\"task\":\"EMBED\",\"kind\":\"EMBEDDING\","
                                    + "\"modality\":\"TEXT\",\"dim\":8,\"labels\":[],\"pooling\":\"MEAN\","
                                    + "\"normalize\":\"L2\",\"max_seq\":16,\"max_batch\":8,\"dtype\":\"F32\","
                                    + "\"fully_accelerated\":false,\"stage_placement\":{\"tokenize\":\"host\","
                                    + "\"encode\":\"host\",\"pool\":\"host\",\"normalize\":\"host\","
                                    + "\"postprocess\":\"unused\"},\"model_id\":\"turbo/mock-embedding\","
                                    + "\"prefix_query\":\"query:\",\"prefix_document\":\"passage:\"}]")))
    public List<TurboService.ModelReport> models() {
        return turbo.models();
    }

    @GetMapping("/models/{name}")
    @Operation(summary = "One loaded model's contract",
            description = "The same entry GET /api/v1/models returns, for a single name. An unknown name is 404.")
    @ApiResponse(responseCode = "200", description = "The model's contract")
    @ApiResponse(responseCode = "404", description = "No model is served under that name",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public TurboService.ModelReport model(
            @Parameter(description = "The name GET /api/v1/models lists", example = "mock-embedding")
            @PathVariable("name") String name) {
        return TurboService.report(turbo.model(name));
    }
}
