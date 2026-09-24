// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.oip;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import java.util.Map;

/**
 * The JSON objects of the Open Inference Protocol version 2, HTTP/REST
 * binding, as KServe defines them (Server Metadata, Model Metadata, Inference
 * Request and Inference Response JSON Objects):
 * https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/
 *
 * <p>Field names are the protocol's own, so this surface is portable to any
 * OIP client. The richer {@code /api/v1} surface is where libturbo's own
 * options, capability bits and timings live.
 */
public final class Oip {
    private Oip() {}

    /** Server Metadata Response JSON Object. */
    @Schema(name = "OipServerMetadata", description = "Open Inference Protocol server metadata")
    public record ServerMetadata(
            @Schema(example = "turbo") String name,
            @Schema(example = "2.0.0-alpha.0") String version,
            @Schema(example = "[\"model_repository\"]") List<String> extensions) {}

    /** One tensor of a Model Metadata Response JSON Object. */
    @Schema(name = "OipTensorMetadata", description = "One input or output tensor a model accepts or produces")
    public record TensorMetadata(
            @Schema(example = "text") String name,
            @Schema(description = "Tensor Data Type from the protocol's table", example = "BYTES") String datatype,
            @Schema(description = "Shape; -1 is a variable extent", example = "[-1]") List<Long> shape) {}

    /** Model Metadata Response JSON Object. */
    @Schema(name = "OipModelMetadata", description = "Open Inference Protocol model metadata")
    public record ModelMetadata(
            @Schema(example = "mock-embedding") String name,
            @Schema(example = "[\"1\"]") List<String> versions,
            @Schema(description = "The provider that runs this model", example = "mock") String platform,
            List<TensorMetadata> inputs,
            List<TensorMetadata> outputs) {}

    /** One tensor of an Inference Request JSON Object. */
    @Schema(name = "OipInferInput", description = "One input tensor of an inference request")
    public record InferInput(
            @Schema(example = "text") String name,
            @Schema(example = "[2]") List<Long> shape,
            @Schema(description = "BYTES for text, FP32 for vectors, INT32 for ids", example = "BYTES") String datatype,
            @Schema(description = "The tensor contents, flattened row-major") List<Object> data,
            Map<String, Object> parameters) {}

    /** One entry of the optional {@code outputs} list of an Inference Request JSON Object. */
    @Schema(name = "OipRequestedOutput", description = "An output the client wants back")
    public record RequestedOutput(
            @Schema(example = "embeddings") String name,
            Map<String, Object> parameters) {}

    /** Inference Request JSON Object. */
    @Schema(name = "OipInferRequest", description = "Open Inference Protocol inference request")
    public record InferRequest(
            @Schema(description = "Client-chosen id, echoed in the response", example = "req-1") String id,
            Map<String, Object> parameters,
            List<InferInput> inputs,
            List<RequestedOutput> outputs) {}

    /** One tensor of an Inference Response JSON Object. */
    @Schema(name = "OipInferOutput", description = "One output tensor of an inference response")
    public record InferOutput(
            @Schema(example = "embeddings") String name,
            @Schema(example = "[2,8]") List<Long> shape,
            @Schema(example = "FP32") String datatype,
            List<Object> data) {}

    /** Inference Response JSON Object. */
    @Schema(name = "OipInferResponse", description = "Open Inference Protocol inference response")
    public record InferResponse(
            @Schema(example = "mock-embedding") String modelName,
            @Schema(example = "1") String modelVersion,
            @Schema(example = "req-1") String id,
            Map<String, Object> parameters,
            List<InferOutput> outputs) {}

    /** Server Live and Server Ready Response JSON Objects, which both name the field {@code live}. */
    @Schema(name = "OipServerHealth", description = "The protocol's liveness and readiness body")
    public record ServerHealth(
            @Schema(example = "true") boolean live,
            @Schema(description = "Also true, for clients that read `ready` on /v2/health/ready", example = "true")
            boolean ready) {}

    /** Model Ready Response JSON Object. */
    @Schema(name = "OipModelHealth", description = "The protocol's model readiness body")
    public record ModelHealth(
            @Schema(example = "mock-embedding") String name,
            @Schema(example = "true") boolean ready) {}

    /** The protocol's error body: a single {@code error} string. */
    @Schema(name = "OipError", description = "The protocol's error body")
    public record Error(@Schema(example = "input `text` must be datatype BYTES, not FP32") String error) {}

    /** The version every model of this server is served under. */
    public static final String VERSION = "1";
}
