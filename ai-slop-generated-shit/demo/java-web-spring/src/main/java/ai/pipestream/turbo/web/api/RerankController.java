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

/** Reranking: one relevance score per document against a query. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Rerank", description = "Relevance of documents to a query, on a cross-encoder bundle")
public class RerankController {
    /** A query, the documents to score against it, and the per-call options. */
    @Schema(name = "RerankRequest", description = "A query and the documents to score against it")
    public record RerankRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first reranker loaded",
                    example = "mock-reranker")
            String model,
            @Schema(description = "The query every document is scored against", example = "how fast is the gpu")
            @NotBlank(message = "query must not be blank") String query,
            @Schema(description = "Documents to score; at most the model's max_batch",
                    example = "[\"the gpu runs at 2.5 GHz\",\"the recipe needs two eggs\"]")
            @NotEmpty(message = "documents must not be empty")
            List<@NotBlank(message = "documents must not contain a blank string") String> documents,
            @Valid RerankOptionsDto options) {}

    /** One hit: the document, its input index, its score, and its rank when one was asked for. */
    @Schema(name = "RerankHit", description = "One scored document")
    public record RerankHit(int index, String document, float score, Integer rank) {}

    /** Scores in input order, plus the ranking when {@code return_sorted} or {@code top_n} was set. */
    @Schema(name = "RerankResponse", description = "Scores in input order, with the ranking when one was asked for")
    public record RerankResponse(String model, TurboService.DeviceRef device, String query, List<RerankHit> results,
            List<Integer> sorted, String placement, TurboService.Timings timings) {}

    private static final String EXAMPLE = """
            {"model":"mock-reranker","device":{"index":1,"name":"Mock accelerator","provider_id":"mock","ordinal":1},\
            "query":"how fast is the gpu","results":[{"index":0,"document":"the gpu runs at 2.5 GHz","score":0.88,\
            "rank":0},{"index":1,"document":"the recipe needs two eggs","score":0.12,"rank":1}],"sorted":[0,1],\
            "placement":"HOST","timings":{"write_ms":0.09,"run_ms":0.22,"read_ms":0.03,"total_ms":0.34}}""";

    private final TurboService turbo;

    public RerankController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/rerank")
    @Operation(summary = "Score documents against a query",
            description = "Writes (query, document) pairs into a session of the named reranker and runs it. "
                    + "Scores come back in input order; set return_sorted or top_n for the ranking, both of which "
                    + "the device must advertise as TURBO_CAP_OPT_TOP_N or the call is refused naming the field. "
                    + "raw_scores returns the logits instead of the bundle's activation.")
    @ApiResponse(responseCode = "200", description = "The scores",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = RerankResponse.class),
                    examples = @ExampleObject(name = "two documents on the mock reranker", value = EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Blank query, empty documents, top_n above the batch, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "404", description = "No model is served under that name",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "409", description = "No loaded model performs RERANK",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "501", description = "The device does not implement an option that was set",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public RerankResponse rerank(@Valid @RequestBody RerankRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.RERANK);
        List<String> documents = List.copyOf(req.documents());
        RerankOptionsDto opts = req.options() == null ? RerankOptionsDto.defaults() : req.options();
        TurboService.RerankResult r = turbo.rerank(m, req.query(), documents, opts.toTurbo());
        Integer[] rank = new Integer[documents.size()];
        List<Integer> sorted = null;
        if (r.sorted() != null) {
            sorted = new ArrayList<>(r.sorted().length);
            for (int position = 0; position < r.sorted().length; position++) {
                int document = r.sorted()[position];
                sorted.add(document);
                if (document >= 0 && document < rank.length) {
                    rank[document] = position;
                }
            }
        }
        List<RerankHit> hits = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i++) {
            hits.add(new RerankHit(i, documents.get(i), r.scores()[i], rank[i]));
        }
        return new RerankResponse(r.model(), r.device(), req.query(), hits, sorted, r.placement(), r.timings());
    }
}
