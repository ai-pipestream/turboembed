// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.web.BenchmarkReceipts;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** The committed benchmark receipts, as the Benchmarks panel draws them. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Benchmarks", description = "The committed receipts: libturbo against the same runtime driven directly")
public class BenchmarkController {
    private final BenchmarkReceipts receipts;

    public BenchmarkController(BenchmarkReceipts receipts) {
        this.receipts = receipts;
    }

    @GetMapping("/benchmarks")
    @Operation(summary = "Every committed benchmark receipt, by kind",
            description = "Reads the directory named by turbo.receipts and answers with three groups. "
                    + "comparisons is one entry per compare receipt: the device, the libturbo provider and its "
                    + "runtime, the runtime the reference program drove directly, the bundle, the dates and "
                    + "commits of both sides, every matched cell with its ratio (libturbo throughput as a "
                    + "fraction of native, so above 1.0 is faster than the raw runtime), and the verdict, which "
                    + "is SUPPORTED when every cell reaches the floor and nothing is unmatched. turbo is one "
                    + "entry per libturbo receipt and native one per direct-native receipt, each with the embed "
                    + "cells, the rerank cell and the generation cell the receipt carries. Nothing is computed "
                    + "here: a figure a receipt does not carry is absent rather than defaulted.")
    @ApiResponse(responseCode = "200", description = "The receipts in the directory",
            content = @Content(mediaType = "application/json",
                    schema = @Schema(implementation = BenchmarkReceipts.BenchmarkReport.class),
                    examples = @ExampleObject(name = "one comparison and its libturbo side",
                            value = "{\"directory\":\"/src/turbo/testdata/receipts/turbo/bench\","
                                    + "\"receipt_count\":31,"
                                    + "\"comparisons\":[{\"file\":\"compare-cuda-rtx4080-embed-2026-09-22.json\","
                                    + "\"verdict\":\"SUPPORTED\",\"floor\":0.95,\"task\":\"embed\","
                                    + "\"device\":\"NVIDIA GeForce RTX 4080 SUPER (sm_89)\",\"device_kind\":\"Gpu\","
                                    + "\"provider\":\"cuda\",\"native_provider\":\"onnxruntime-cuda\","
                                    + "\"model_id\":\"sentence-transformers/all-MiniLM-L6-v2\","
                                    + "\"date\":\"2026-09-22\",\"cells\":[{\"cell\":\"embed 8x128\","
                                    + "\"measure\":\"prepared tokens p50 ms\",\"turbo\":1.452284,"
                                    + "\"native\":3.08509,\"ratio\":2.1243021337424364,\"within_floor\":true}],"
                                    + "\"unmatched\":[]}],"
                                    + "\"turbo\":[{\"file\":\"cuda-rtx4080-embed-2026-09-22.json\","
                                    + "\"kind\":\"benchmark\",\"task\":\"embed\",\"date\":\"2026-09-22\","
                                    + "\"provider\":\"cuda\",\"device\":\"NVIDIA GeForce RTX 4080 SUPER (sm_89)\","
                                    + "\"embed\":[{\"batch\":8,\"seq\":128,\"live_tokens_per_row\":21.0,"
                                    + "\"token_count_source\":\"tokenizer\",\"text\":{\"p50_ms\":1.5,"
                                    + "\"rows_per_s\":5449.0,\"tokens_per_s\":114429.0,\"iters\":30}}]}],"
                                    + "\"native\":[]}")))
    @ApiResponse(responseCode = "404",
            description = "turbo.receipts names a directory that is not there, or one that holds no receipt",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class),
                    examples = @ExampleObject(name = "no such directory",
                            value = "{\"error\":\"no benchmark receipts: /src/turbo/nowhere is not a directory; "
                                    + "point turbo.receipts at the tree's testdata/receipts/turbo/bench\","
                                    + "\"path\":\"/api/v1/benchmarks\"}")))
    @ApiResponse(responseCode = "500", description = "A file in the directory is not a receipt this server can read",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public BenchmarkReceipts.BenchmarkReport benchmarks() {
        return receipts.report();
    }
}
