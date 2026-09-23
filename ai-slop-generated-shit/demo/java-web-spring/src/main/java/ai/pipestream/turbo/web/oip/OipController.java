// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.oip;

import ai.pipestream.turbo.Aggregation;
import ai.pipestream.turbo.ClassifyOptions;
import ai.pipestream.turbo.EmbedOptions;
import ai.pipestream.turbo.GenerateDesc;
import ai.pipestream.turbo.Message;
import ai.pipestream.turbo.ModelInfo;
import ai.pipestream.turbo.Normalize;
import ai.pipestream.turbo.OutputDType;
import ai.pipestream.turbo.Pooling;
import ai.pipestream.turbo.PromptRole;
import ai.pipestream.turbo.RerankOptions;
import ai.pipestream.turbo.Truncate;
import ai.pipestream.turbo.web.LoadedModel;
import ai.pipestream.turbo.web.TurboService;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * The KServe Open Inference Protocol version 2, HTTP/REST binding.
 *
 * <p>Implements the six endpoints of the protocol's HTTP/REST section: Server
 * Metadata, Server Live, Server Ready, Model Metadata, Model Ready and
 * Inference, with the version-qualified path forms the specification defines.
 * There is no gRPC here; a separate Rust server is planned for that.
 *
 * <p>The mapping from the protocol's tensors to libturbo's tasks is one entry
 * per model kind and is listed in this app's README. Anything the protocol
 * cannot say, such as which capability bit refused an option, is on
 * {@code /api/v1} instead.
 */
@RestController
@Tag(name = "Open Inference Protocol v2",
        description = "The KServe OIP v2 REST surface: server and model metadata, readiness, and inference")
public class OipController {
    private static final String LIVE_READY =
            "200 with the protocol's probe object; the answer is also in the status code alone";

    private final TurboService turbo;
    private final ObjectMapper json;
    private final String serverName;
    private final String serverVersion;

    public OipController(TurboService turbo, ObjectMapper json,
            @Value("${turbo.oip.server-name:turbo}") String serverName,
            @Value("${turbo.oip.server-version:2.0.0-alpha.0}") String serverVersion) {
        this.turbo = turbo;
        this.json = json;
        this.serverName = serverName;
        this.serverVersion = serverVersion;
    }

    @GetMapping("/v2")
    @Operation(summary = "Server Metadata",
            description = "The protocol's Server Metadata endpoint. `extensions` is empty: this server implements "
                    + "the core protocol and no optional extension.")
    @ApiResponse(responseCode = "200", description = "The server's name, version and extensions",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.ServerMetadata.class),
                    examples = @ExampleObject(value = "{\"name\":\"turbo\",\"version\":\"2.0.0-alpha.0\",\"extensions\":[]}")))
    public Oip.ServerMetadata server() {
        return new Oip.ServerMetadata(serverName, serverVersion, List.of());
    }

    @GetMapping("/v2/health/live")
    @Operation(summary = "Server Live", description = "200 once the HTTP server is accepting requests.")
    @ApiResponse(responseCode = "200", description = LIVE_READY,
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.ServerHealth.class),
                    examples = @ExampleObject(value = "{\"live\":true,\"ready\":true}")))
    public Oip.ServerHealth live() {
        return new Oip.ServerHealth(true, true);
    }

    @GetMapping("/v2/health/ready")
    @Operation(summary = "Server Ready",
            description = "200 once every configured bundle has loaded. A bundle that fails to load fails startup, "
                    + "so this server is never up and unready. The specification's Server Ready Response JSON Object "
                    + "names its only field `live`; `ready` carries the same answer for clients that read that name.")
    @ApiResponse(responseCode = "200", description = LIVE_READY,
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.ServerHealth.class),
                    examples = @ExampleObject(value = "{\"live\":true,\"ready\":true}")))
    public Oip.ServerHealth ready() {
        turbo.health();
        return new Oip.ServerHealth(true, true);
    }

    @GetMapping({"/v2/models/{name}", "/v2/models/{name}/versions/{version}"})
    @Operation(summary = "Model Metadata",
            description = "The model's inputs and outputs with their datatypes and shapes, where -1 is a variable "
                    + "extent. `platform` is the provider that runs the model. The full libturbo contract, "
                    + "including pooling, normalization and the stage placements, is on GET /api/v1/models.")
    @ApiResponse(responseCode = "200", description = "The model's metadata",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.ModelMetadata.class),
                    examples = @ExampleObject(name = "an embedding model",
                            value = "{\"name\":\"mock-embedding\",\"versions\":[\"1\"],\"platform\":\"mock\","
                                    + "\"inputs\":[{\"name\":\"text\",\"datatype\":\"BYTES\",\"shape\":[-1]}],"
                                    + "\"outputs\":[{\"name\":\"embeddings\",\"datatype\":\"FP32\",\"shape\":[-1,8]}]}")))
    @ApiResponse(responseCode = "404", description = "No model is served under that name or version",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.Error.class)))
    public Oip.ModelMetadata metadata(
            @Parameter(description = "The name GET /api/v1/models lists", example = "mock-embedding")
            @PathVariable("name") String name,
            @Parameter(description = "Model version; this server serves version 1", example = "1")
            @PathVariable(value = "version", required = false) String version) {
        LoadedModel m = model(name, version);
        ModelInfo i = m.info();
        return new Oip.ModelMetadata(m.name(), List.of(Oip.VERSION), i.providerId(), inputs(i), outputs(i));
    }

    @GetMapping({"/v2/models/{name}/ready", "/v2/models/{name}/versions/{version}/ready"})
    @Operation(summary = "Model Ready", description = "200 when the named model is loaded and can take a request.")
    @ApiResponse(responseCode = "200", description = LIVE_READY,
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.ModelHealth.class),
                    examples = @ExampleObject(value = "{\"name\":\"mock-embedding\",\"ready\":true}")))
    @ApiResponse(responseCode = "404", description = "No model is served under that name or version",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.Error.class)))
    public Oip.ModelHealth modelReady(
            @Parameter(description = "The name GET /api/v1/models lists", example = "mock-embedding")
            @PathVariable("name") String name,
            @Parameter(description = "Model version; this server serves version 1", example = "1")
            @PathVariable(value = "version", required = false) String version) {
        return new Oip.ModelHealth(model(name, version).name(), true);
    }

    @PostMapping({"/v2/models/{name}/infer", "/v2/models/{name}/versions/{version}/infer"})
    @Operation(summary = "Inference",
            description = """
                    Runs the model named in the path. Which tensors a request carries depends on the model's kind, \
                    which GET /v2/models/{name} reports: an embedding model takes `text` BYTES and answers \
                    `embeddings` FP32; a reranker takes `query` and `documents` BYTES and answers `scores` FP32 and \
                    `sorted` INT32; a classifier takes `text` BYTES and answers `scores` FP32 and `labels` BYTES; a \
                    generative model takes `prompt` BYTES or `messages` BYTES (one JSON object per turn) and answers \
                    `text` BYTES with the finish reason and token counts in the response parameters. Per-call \
                    options travel in the request's `parameters` object, are honored exactly, and are refused by \
                    name when the device does not implement them.""")
    @ApiResponse(responseCode = "200", description = "The inference response",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.InferResponse.class),
                    examples = @ExampleObject(name = "two texts through an embedding model",
                            value = "{\"model_name\":\"mock-embedding\",\"model_version\":\"1\",\"id\":\"req-1\","
                                    + "\"parameters\":{\"placement\":\"HOST\",\"total_ms\":0.47},"
                                    + "\"outputs\":[{\"name\":\"embeddings\",\"shape\":[2,8],\"datatype\":\"FP32\","
                                    + "\"data\":[0.35,-0.12,0.44,0.02,-0.31,0.51,-0.18,0.53,0.31,-0.09,0.47,0.06,"
                                    + "-0.28,0.55,-0.14,0.50]}]}")))
    @ApiResponse(responseCode = "400", description = "The request does not match the model's tensors",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.Error.class)))
    @ApiResponse(responseCode = "404", description = "No model is served under that name or version",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.Error.class)))
    @ApiResponse(responseCode = "500", description = "The server or the device failed",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = Oip.Error.class)))
    public Oip.InferResponse infer(
            @Parameter(description = "The name GET /api/v1/models lists", example = "mock-embedding")
            @PathVariable("name") String name,
            @Parameter(description = "Model version; this server serves version 1", example = "1")
            @PathVariable(value = "version", required = false) String version,
            @RequestBody Oip.InferRequest request) {
        LoadedModel m = model(name, version);
        if (request == null || request.inputs() == null || request.inputs().isEmpty()) {
            throw new IllegalArgumentException("an inference request needs at least one entry in `inputs`");
        }
        Map<String, Object> parameters = request.parameters();
        Inferred produced = switch (m.info().kind()) {
            case EMBEDDING -> embed(m, request, parameters);
            case RERANKER -> rerank(m, request, parameters);
            case CLASSIFIER -> classify(m, request, parameters);
            case TOKEN_CLASSIFIER -> tokenClassify(m, request, parameters);
            case GENERATIVE -> generate(m, request, parameters);
            case GENERIC -> throw new IllegalStateException(
                    "model `" + m.name() + "` is GENERIC, which this server does not load");
        };
        return new Oip.InferResponse(m.name(), Oip.VERSION, request.id(), produced.parameters(),
                requested(request, produced.outputs()));
    }

    /** What one task produced: its tensors and the response-level parameters that describe the run. */
    private record Inferred(List<Oip.InferOutput> outputs, Map<String, Object> parameters) {}

    private Inferred embed(LoadedModel m, Oip.InferRequest request, Map<String, Object> parameters) {
        List<String> texts = texts(input(request, "text"), "text");
        EmbedOptions options = new EmbedOptions(
                or(Params.constant(Truncate.class, parameters, "truncate"), Truncate.MODEL),
                or(Params.integer(parameters, "max_tokens"), 0),
                or(Params.constant(PromptRole.class, parameters, "prompt_role"), PromptRole.NONE),
                or(Params.constant(Normalize.class, parameters, "normalize"), Normalize.MODEL),
                or(Params.constant(Pooling.class, parameters, "pooling"), Pooling.MODEL),
                or(Params.integer(parameters, "output_dim"), 0),
                or(Params.constant(OutputDType.class, parameters, "output_dtype"), OutputDType.MODEL));
        TurboService.EmbedResult r = turbo.embed(m, texts, options);
        List<Object> data = new ArrayList<>(texts.size() * r.dim());
        for (float[] v : r.vectors()) {
            for (float x : v) {
                data.add(x);
            }
        }
        return new Inferred(
                List.of(new Oip.InferOutput("embeddings", List.of((long) texts.size(), (long) r.dim()), "FP32", data)),
                ran(r.placement(), r.timings().totalMs()));
    }

    private Inferred rerank(LoadedModel m, Oip.InferRequest request, Map<String, Object> parameters) {
        List<String> query = texts(input(request, "query"), "query");
        if (query.size() != 1) {
            throw new IllegalArgumentException("input `query` must carry exactly one string, not " + query.size());
        }
        List<String> documents = texts(input(request, "documents"), "documents");
        RerankOptions options = new RerankOptions(
                or(Params.constant(Truncate.class, parameters, "truncate"), Truncate.MODEL),
                or(Params.integer(parameters, "max_tokens"), 0),
                or(Params.integer(parameters, "top_n"), 0),
                or(Params.flag(parameters, "return_sorted"), false),
                or(Params.flag(parameters, "raw_scores"), false));
        TurboService.RerankResult r = turbo.rerank(m, query.get(0), documents, options);
        List<Oip.InferOutput> out = new ArrayList<>(2);
        List<Object> scores = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i++) {
            scores.add(r.scores()[i]);
        }
        out.add(new Oip.InferOutput("scores", List.of((long) documents.size()), "FP32", scores));
        if (r.sorted() != null) {
            List<Object> sorted = new ArrayList<>(r.sorted().length);
            for (int index : r.sorted()) {
                sorted.add(index);
            }
            out.add(new Oip.InferOutput("sorted", List.of((long) r.sorted().length), "INT32", sorted));
        }
        return new Inferred(out, ran(r.placement(), r.timings().totalMs()));
    }

    private Inferred classify(LoadedModel m, Oip.InferRequest request, Map<String, Object> parameters) {
        List<String> texts = texts(input(request, "text"), "text");
        TurboService.ClassifyResult r = turbo.classify(m, texts, classifyOptions(parameters));
        int width = r.scores().length == 0 ? 0 : r.scores()[0].length;
        List<Object> data = new ArrayList<>(texts.size() * width);
        for (float[] row : r.scores()) {
            for (float x : row) {
                data.add(x);
            }
        }
        return new Inferred(List.of(
                new Oip.InferOutput("scores", List.of((long) texts.size(), (long) width), "FP32", data),
                new Oip.InferOutput("labels", List.of((long) r.labels().size()), "BYTES", List.copyOf(r.labels()))),
                ran(r.placement(), r.timings().totalMs()));
    }

    private Inferred tokenClassify(LoadedModel m, Oip.InferRequest request, Map<String, Object> parameters) {
        List<String> texts = texts(input(request, "text"), "text");
        TurboService.TokenClassifyResult r = turbo.tokenClassify(m, texts, classifyOptions(parameters));
        List<Long> shape = new ArrayList<>(r.scoreShape().length);
        for (long d : r.scoreShape()) {
            shape.add(d);
        }
        List<Object> data = new ArrayList<>(r.scores().length);
        for (float x : r.scores()) {
            data.add(x);
        }
        return new Inferred(List.of(
                new Oip.InferOutput("scores", shape, "FP32", data),
                new Oip.InferOutput("labels", List.of((long) r.labels().size()), "BYTES", List.copyOf(r.labels()))),
                ran(r.placement(), r.timings().totalMs()));
    }

    private ClassifyOptions classifyOptions(Map<String, Object> parameters) {
        return new ClassifyOptions(
                or(Params.constant(Truncate.class, parameters, "truncate"), Truncate.MODEL),
                or(Params.integer(parameters, "max_tokens"), 0),
                or(Params.constant(Aggregation.class, parameters, "aggregation"), Aggregation.MODEL),
                or(Params.flag(parameters, "raw_scores"), false));
    }

    private Inferred generate(LoadedModel m, Oip.InferRequest request, Map<String, Object> parameters) {
        Oip.InferInput prompt = find(request, "prompt");
        Oip.InferInput chat = find(request, "messages");
        if ((prompt == null) == (chat == null)) {
            throw new IllegalArgumentException(
                    "a generative model takes input `prompt` or input `messages`, not both and not neither");
        }
        List<Message> messages;
        if (prompt != null) {
            List<String> one = texts(prompt, "prompt");
            if (one.size() != 1) {
                throw new IllegalArgumentException("input `prompt` must carry exactly one string, not " + one.size());
            }
            messages = List.of(Message.user(one.get(0)));
        } else {
            List<String> turns = texts(chat, "messages");
            messages = new ArrayList<>(turns.size());
            for (int i = 0; i < turns.size(); i++) {
                messages.add(turn(turns.get(i), i));
            }
        }
        List<String> stop = Params.strings(parameters, "stop");
        GenerateDesc desc = new GenerateDesc(
                or(Params.integer(parameters, "max_tokens"), 0),
                or(Params.integer(parameters, "min_tokens"), 0),
                0,
                or(Params.number(parameters, "temperature"), 0f),
                or(Params.integer(parameters, "top_k"), 0),
                or(Params.number(parameters, "top_p"), 0f),
                or(Params.number(parameters, "min_p"), 0f),
                0f,
                0f,
                0f,
                Params.longValue(parameters, "seed"),
                stop == null ? List.of() : stop,
                new int[0],
                Map.of(),
                or(Params.integer(parameters, "logprobs"), 0),
                or(Params.flag(parameters, "echo"), false));
        TurboService.GenerateResult r = turbo.generate(m, messages, desc, chunk -> true);
        Map<String, Object> described = new LinkedHashMap<>();
        described.put("finish_reason", r.finishReason());
        described.put("prompt_tokens", r.promptTokens());
        described.put("generated_tokens", r.generatedTokens());
        described.put("total_ms", r.timings().totalMs());
        return new Inferred(List.of(new Oip.InferOutput("text", List.of(1L), "BYTES", List.of(r.text()))), described);
    }

    /** The response parameters that describe a session run. */
    private static Map<String, Object> ran(String placement, double totalMs) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("placement", placement);
        out.put("total_ms", totalMs);
        return out;
    }

    private Message turn(String raw, int index) {
        try {
            Map<?, ?> parsed = json.readValue(raw, Map.class);
            Object role = parsed.get("role");
            Object content = parsed.get("content");
            if (!(role instanceof String r) || r.isBlank() || !(content instanceof String c)) {
                throw new IllegalArgumentException("input `messages[" + index
                        + "]` must be a JSON object with a non-empty string `role` and a string `content`");
            }
            return new Message(r, c);
        } catch (com.fasterxml.jackson.core.JacksonException e) {
            throw new IllegalArgumentException("input `messages[" + index + "]` is not a JSON object: " + e.getOriginalMessage());
        }
    }

    /** Keep only the outputs the request asked for, in the order it asked for them. */
    private static List<Oip.InferOutput> requested(Oip.InferRequest request, List<Oip.InferOutput> produced) {
        if (request.outputs() == null || request.outputs().isEmpty()) {
            return produced;
        }
        List<Oip.InferOutput> out = new ArrayList<>(request.outputs().size());
        for (Oip.RequestedOutput wanted : request.outputs()) {
            if (wanted == null || wanted.name() == null) {
                throw new IllegalArgumentException("every entry of `outputs` needs a `name`");
            }
            Oip.InferOutput match = produced.stream().filter(o -> o.name().equals(wanted.name())).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("this model produces no output named `"
                            + wanted.name() + "`; it produces "
                            + produced.stream().map(Oip.InferOutput::name).toList()));
            out.add(match);
        }
        return out;
    }

    private LoadedModel model(String name, String version) {
        if (version != null && !version.equals(Oip.VERSION)) {
            throw new TurboService.ModelNotFound(
                    "model `" + name + "` has no version `" + version + "`; this server serves version " + Oip.VERSION);
        }
        return turbo.model(name);
    }

    private static Oip.InferInput find(Oip.InferRequest request, String name) {
        for (Oip.InferInput in : request.inputs()) {
            if (in != null && name.equals(in.name())) {
                return in;
            }
        }
        return null;
    }

    private static Oip.InferInput input(Oip.InferRequest request, String name) {
        Oip.InferInput in = find(request, name);
        if (in == null) {
            List<String> given = request.inputs().stream().map(i -> i == null ? "null" : i.name()).toList();
            throw new IllegalArgumentException("this model needs an input named `" + name + "`; the request carries " + given);
        }
        return in;
    }

    /** The strings of a BYTES tensor, checked against its declared datatype and shape. */
    private static List<String> texts(Oip.InferInput in, String name) {
        if (!"BYTES".equals(in.datatype())) {
            throw new IllegalArgumentException(
                    "input `" + name + "` must be datatype BYTES, not " + in.datatype());
        }
        if (in.data() == null) {
            throw new IllegalArgumentException("input `" + name + "` carries no `data`");
        }
        List<String> out = new ArrayList<>(in.data().size());
        for (int i = 0; i < in.data().size(); i++) {
            Object e = in.data().get(i);
            if (!(e instanceof String s)) {
                throw new IllegalArgumentException("input `" + name + "` is BYTES, so data[" + i
                        + "] must be a string, not " + (e == null ? "null" : e.getClass().getSimpleName()));
            }
            out.add(s);
        }
        long declared = elements(in.shape());
        if (declared >= 0 && declared != out.size()) {
            throw new IllegalArgumentException("input `" + name + "` declares shape " + in.shape() + " ("
                    + declared + " elements) but carries " + out.size());
        }
        return out;
    }

    private static long elements(List<Long> shape) {
        if (shape == null || shape.isEmpty()) {
            return -1;
        }
        long n = 1;
        for (Long d : shape) {
            if (d == null || d < 0) {
                return -1;
            }
            n *= d;
        }
        return n;
    }

    private static <T> T or(T value, T fallback) {
        return value == null ? fallback : value;
    }

    /** The input tensors a model of this kind accepts. */
    static List<Oip.TensorMetadata> inputs(ModelInfo i) {
        return switch (i.kind()) {
            case EMBEDDING, CLASSIFIER, TOKEN_CLASSIFIER ->
                List.of(new Oip.TensorMetadata("text", "BYTES", List.of(-1L)));
            case RERANKER -> List.of(
                    new Oip.TensorMetadata("query", "BYTES", List.of(1L)),
                    new Oip.TensorMetadata("documents", "BYTES", List.of(-1L)));
            case GENERATIVE -> List.of(
                    new Oip.TensorMetadata("prompt", "BYTES", List.of(1L)),
                    new Oip.TensorMetadata("messages", "BYTES", List.of(-1L)));
            case GENERIC -> List.of();
        };
    }

    /** The output tensors a model of this kind produces. */
    static List<Oip.TensorMetadata> outputs(ModelInfo i) {
        long labels = i.labels().size();
        return switch (i.kind()) {
            case EMBEDDING -> List.of(new Oip.TensorMetadata("embeddings", "FP32", List.of(-1L, (long) i.dim())));
            case RERANKER -> List.of(
                    new Oip.TensorMetadata("scores", "FP32", List.of(-1L)),
                    new Oip.TensorMetadata("sorted", "INT32", List.of(-1L)));
            case CLASSIFIER -> List.of(
                    new Oip.TensorMetadata("scores", "FP32", List.of(-1L, labels)),
                    new Oip.TensorMetadata("labels", "BYTES", List.of(labels)));
            case TOKEN_CLASSIFIER -> List.of(
                    new Oip.TensorMetadata("scores", "FP32", List.of(-1L, (long) i.maxSeq(), labels)),
                    new Oip.TensorMetadata("labels", "BYTES", List.of(labels)));
            case GENERATIVE -> List.of(new Oip.TensorMetadata("text", "BYTES", List.of(1L)));
            case GENERIC -> List.of();
        };
    }
}
