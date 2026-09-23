// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.test.context.TestPropertySource;

/**
 * {@code GET /api/v1/benchmarks} over a frozen copy of receipts in
 * {@code src/test/resources/receipts}: the values asserted below are those
 * files' values, so a benchmark re-run in the tree (which replaces the
 * receipts under {@code testdata/receipts/turbo/bench}) does not move them.
 * The copies are receipts that were committed at some point, with the
 * machine named by its architecture label.
 *
 * <p>The application's default {@code turbo.receipts} still points at the
 * tree's directory; {@link BenchmarkReceiptsDirectoryTest} covers that path.
 */
@TestPropertySource(properties = "turbo.receipts=src/test/resources/receipts")
class BenchmarkApiTest extends ApiTestBase {
    private static final ObjectMapper JSON = new ObjectMapper();

    private JsonNode report() throws Exception {
        String body = mvc.perform(get("/api/v1/benchmarks"))
                .andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        return JSON.readTree(body);
    }

    private static JsonNode named(JsonNode list, String file) {
        for (JsonNode entry : list) {
            if (file.equals(entry.path("file").asText())) {
                return entry;
            }
        }
        throw new AssertionError("no entry for " + file + " in " + list);
    }

    private static JsonNode cell(JsonNode comparison, String name, String measure) {
        for (JsonNode c : comparison.path("cells")) {
            if (name.equals(c.path("cell").asText()) && measure.equals(c.path("measure").asText())) {
                return c;
            }
        }
        throw new AssertionError("no cell " + name + " measured as " + measure + " in " + comparison.path("file"));
    }

    @Test
    void everyReceiptInTheDirectoryIsGroupedByItsKind() throws Exception {
        JsonNode report = report();
        assertTrue(report.path("directory").asText().endsWith("src/test/resources/receipts"),
                "the report names the directory it read: " + report.path("directory").asText());
        int comparisons = report.path("comparisons").size();
        int turbo = report.path("turbo").size();
        int natives = report.path("native").size();
        assertTrue(comparisons > 0, "the fixtures carry compare receipts");
        assertTrue(turbo > 0, "the fixtures carry libturbo receipts");
        assertTrue(natives > 0, "the committed tree carries direct-native receipts");
        assertEquals(report.path("receipt_count").asInt(), comparisons + turbo + natives,
                "every file read is in exactly one group");
        for (JsonNode c : report.path("comparisons")) {
            assertTrue(c.path("file").asText().startsWith("compare-"), "a comparison comes from a compare-* file");
            assertFalse(c.path("verdict").asText().isBlank(), "every comparison carries a verdict");
        }
        for (JsonNode r : report.path("turbo")) {
            assertEquals("benchmark", r.path("kind").asText(), r.path("file").asText());
        }
        for (JsonNode r : report.path("native")) {
            assertEquals("native", r.path("kind").asText(), r.path("file").asText());
        }
    }

    @Test
    void aComparisonCarriesBothSidesTheBundleAndEveryCell() throws Exception {
        JsonNode c = named(report().path("comparisons"), "compare-cuda-rtx4080-embed-2026-09-22.json");
        assertEquals("SUPPORTED", c.path("verdict").asText());
        assertEquals(0.95, c.path("floor").asDouble(), 1e-9);
        assertEquals("embed", c.path("task").asText());
        assertEquals("NVIDIA GeForce RTX 4080 SUPER (sm_89)", c.path("device").asText());
        assertEquals("Gpu", c.path("device_kind").asText());
        assertEquals("cuda", c.path("provider").asText());
        assertEquals("13.2", c.path("driver").asText());
        assertEquals("onnxruntime-cuda", c.path("native_provider").asText());
        assertTrue(c.path("native_runtime").asText().startsWith("ONNX Runtime"), c.path("native_runtime").asText());
        assertEquals("sentence-transformers/all-MiniLM-L6-v2", c.path("model_id").asText());
        assertEquals("2026-09-22", c.path("date").asText());
        assertFalse(c.path("commit").asText().isBlank(), "a receipt without a commit is not a measurement");
        assertEquals(9, c.path("cells").size());
        assertEquals(0, c.path("unmatched").size());

        JsonNode best = cell(c, "embed 8x128", "prepared tokens p50 ms");
        assertEquals(1.452284, best.path("turbo").asDouble(), 1e-9);
        assertEquals(3.08509, best.path("native").asDouble(), 1e-9);
        assertEquals(2.1243021337424364, best.path("ratio").asDouble(), 1e-9);
        assertTrue(best.path("within_floor").asBoolean());
    }

    @Test
    void aComparisonWithCellsUnderTheFloorIsExperimentalAndSaysWhich() throws Exception {
        JsonNode c = named(report().path("comparisons"), "compare-openvino-rtx4080-cpu-embed-2026-09-22.json");
        assertEquals("EXPERIMENTAL", c.path("verdict").asText());
        assertEquals("openvino", c.path("provider").asText());
        assertEquals("openvino", c.path("native_provider").asText());
        int under = 0;
        for (JsonNode cell : c.path("cells")) {
            if (!cell.path("within_floor").asBoolean()) {
                under++;
                assertTrue(cell.path("ratio").asDouble() < c.path("floor").asDouble(),
                        cell.path("cell").asText() + " is marked under the floor but is not");
            }
        }
        assertEquals(4, under, "four cells of the CPU comparison are under 0.95");
    }

    @Test
    void everyVerdictFollowsFromTheCellsAndTheFloor() throws Exception {
        for (JsonNode c : report().path("comparisons")) {
            String file = c.path("file").asText();
            double floor = c.path("floor").asDouble();
            boolean all = true;
            for (JsonNode cell : c.path("cells")) {
                boolean within = cell.path("ratio").asDouble() >= floor;
                assertEquals(within, cell.path("within_floor").asBoolean(),
                        file + ": " + cell.path("cell").asText() + " " + cell.path("measure").asText());
                all &= within;
            }
            boolean supported = all && c.path("unmatched").isEmpty();
            assertEquals(supported ? "SUPPORTED" : "EXPERIMENTAL", c.path("verdict").asText(), file);
        }
    }

    @Test
    void aLibturboEmbedReceiptCarriesP50RowsAndTokensPerCell() throws Exception {
        JsonNode run = named(report().path("turbo"), "cuda-rtx4080-embed-2026-09-22.json");
        assertEquals("embed", run.path("task").asText());
        assertEquals("rtx4080", run.path("hostname").asText());
        assertEquals("cuda", run.path("provider").asText());
        assertEquals(0, run.path("ordinal").asInt());
        assertEquals("sentence-transformers/all-MiniLM-L6-v2", run.path("model_id").asText());
        assertEquals(9, run.path("embed").size());
        JsonNode first = run.path("embed").get(0);
        assertEquals(1, first.path("batch").asInt());
        assertEquals(32, first.path("seq").asInt());
        assertEquals(21.0, first.path("live_tokens_per_row").asDouble(), 1e-9);
        assertEquals("tokenizer", first.path("token_count_source").asText());
        assertEquals(0.526034, first.path("text").path("p50_ms").asDouble(), 1e-9);
        assertEquals(1748.0357904501304, first.path("text").path("rows_per_s").asDouble(), 1e-9);
        assertEquals(36708.75159945274, first.path("text").path("tokens_per_s").asDouble(), 1e-9);
        assertEquals(30, first.path("text").path("iters").asInt());
        assertEquals(0.5147529999999999, first.path("prepared_tokens").path("p50_ms").asDouble(), 1e-9);
        assertTrue(run.path("rerank").isMissingNode(), "an embedding receipt carries no rerank cell");
        assertTrue(run.path("generate").isMissingNode(), "an embedding receipt carries no generation cell");
    }

    @Test
    void theRerankReceiptCarriesItsSingleCell() throws Exception {
        JsonNode run = named(report().path("turbo"), "cuda-rtx4080-rerank-2026-09-21.json");
        assertEquals("rerank", run.path("task").asText());
        assertEquals("cross-encoder/ms-marco-MiniLM-L-12-v2", run.path("model_id").asText());
        assertEquals(0, run.path("embed").size());
        assertEquals(16, run.path("rerank").path("docs").asInt());
        assertEquals(128, run.path("rerank").path("seq").asInt());
        assertEquals(0.887377, run.path("rerank").path("text").path("p50_ms").asDouble(), 1e-9);
        assertEquals(1.9988380000000001, run.path("rerank").path("text").path("p99_ms").asDouble(), 1e-9);
        assertEquals(16359.31155654487, run.path("rerank").path("text").path("rows_per_s").asDouble(), 1e-9);
    }

    @Test
    void theGenerationReceiptCarriesTtftTheDecodeRateAndTheTotal() throws Exception {
        JsonNode run = named(report().path("turbo"), "ggml-rtx4080-gpu-generate-2026-09-22.json");
        assertEquals("generate", run.path("task").asText());
        assertEquals("ggml", run.path("provider").asText());
        JsonNode g = run.path("generate");
        assertEquals(128, g.path("new_tokens_requested").asInt());
        assertEquals(128.0, g.path("generated_tokens_mean").asDouble(), 1e-9);
        assertEquals(19, g.path("prompt_tokens").asInt());
        assertEquals(5.111717, g.path("time_to_first_token_ms_p50").asDouble(), 1e-9);
        assertEquals(626.0293006162336, g.path("decode_tokens_per_s_p50").asDouble(), 1e-9);
        assertEquals(208.761781, g.path("total_ms_p50").asDouble(), 1e-9);
        assertEquals(10, g.path("iters").asInt());
        assertEquals(10, g.path("finish_reasons").size());
    }

    @Test
    void aFigureTheReceiptDoesNotCarryStaysAbsent() throws Exception {
        JsonNode report = report();
        // Thirty samples are too few for a p99, so the receipt records none and
        // the endpoint does not fill one in.
        JsonNode cuda = named(report.path("turbo"), "cuda-rtx4080-embed-2026-09-22.json");
        assertTrue(cuda.path("embed").get(0).path("text").path("p99_ms").isMissingNode(),
                "a p99 needs a hundred samples; this cell has thirty");
        // The ggml provider reports no driver version, so there is no driver field.
        JsonNode ggml = named(report.path("turbo"), "ggml-rtx4080-gpu-generate-2026-09-22.json");
        assertTrue(ggml.path("driver").isMissingNode(), "the ggml receipt records an empty driver version");
        // A receipt written before token_count_source existed does not gain one.
        JsonNode hailo = named(report.path("turbo"), "hailo-pi5-hailo8-embed-2026-09-21.json");
        assertTrue(hailo.path("embed").get(0).path("token_count_source").isMissingNode(),
                "this receipt predates token_count_source");
    }
}
