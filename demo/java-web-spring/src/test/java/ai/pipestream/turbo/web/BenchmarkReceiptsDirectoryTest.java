// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import ai.pipestream.turbo.web.api.ApiExceptionHandler;
import ai.pipestream.turbo.web.api.BenchmarkController;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import org.hamcrest.Matchers;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * {@code GET /api/v1/benchmarks} pointed somewhere with no receipts. A
 * directory that is not there, and one that holds none, are both refusals
 * naming the path: the panel says the receipts are missing rather than
 * drawing an empty table.
 *
 * <p>These cases build the controller directly with the refusal handler the
 * app registers, so each one can point {@code turbo.receipts} somewhere else
 * without a second server.
 */
class BenchmarkReceiptsDirectoryTest {
    private static MockMvc serverReading(String receipts) {
        TurboProperties props = new TurboProperties();
        props.setReceipts(receipts);
        BenchmarkController controller = new BenchmarkController(new BenchmarkReceipts(new ObjectMapper(), props));
        return MockMvcBuilders.standaloneSetup(controller).setControllerAdvice(new ApiExceptionHandler()).build();
    }

    @Test
    void aDirectoryThatIsNotThereIsRefusedNamingThePath(@TempDir Path tmp) throws Exception {
        Path missing = tmp.resolve("no-such-receipts");
        serverReading(missing.toString())
                .perform(get("/api/v1/benchmarks"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", Matchers.containsString("no benchmark receipts")))
                .andExpect(jsonPath("$.error", Matchers.containsString(missing.toString())))
                .andExpect(jsonPath("$.error", Matchers.containsString("turbo.receipts")))
                .andExpect(jsonPath("$.path").value("/api/v1/benchmarks"));
    }

    @Test
    void aDirectoryWithNoReceiptInItIsRefusedRatherThanAnsweredEmpty(@TempDir Path tmp) throws Exception {
        serverReading(tmp.toString())
                .perform(get("/api/v1/benchmarks"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", Matchers.containsString("holds no .json file")))
                .andExpect(jsonPath("$.error", Matchers.containsString(tmp.toString())))
                .andExpect(jsonPath("$.comparisons").doesNotExist());
    }
}
