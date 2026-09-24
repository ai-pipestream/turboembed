// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;

/** The static page and the OpenAPI document. */
class ServerSurfaceTest extends ApiTestBase {
    @Test
    void thePageIsServed() throws Exception {
        mvc.perform(get("/index.html"))
                .andExpect(status().isOk())
                .andExpect(content().string(containsString("Turbo inference server")));
        mvc.perform(get("/app.js")).andExpect(status().isOk());
        mvc.perform(get("/app.css")).andExpect(status().isOk());
    }

    @Test
    void theOpenApiDocumentCoversEveryEndpoint() throws Exception {
        mvc.perform(get("/v3/api-docs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.info.title").value("Turbo inference server"))
                .andExpect(jsonPath("$.paths['/api/v1/health']").exists())
                .andExpect(jsonPath("$.paths['/api/v1/devices']").exists())
                .andExpect(jsonPath("$.paths['/api/v1/models']").exists())
                .andExpect(jsonPath("$.paths['/api/v1/models/{name}']").exists())
                .andExpect(jsonPath("$.paths['/api/v1/benchmarks'].get").exists())
                .andExpect(jsonPath("$.paths['/api/v1/embed'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/similarity'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/rerank'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/classify'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/token-classify'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/tokenize'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/detokenize'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/generate'].post").exists())
                .andExpect(jsonPath("$.paths['/api/v1/generate/stream'].post").exists())
                .andExpect(jsonPath("$.paths['/v2'].get").exists())
                .andExpect(jsonPath("$.paths['/v2/health/live'].get").exists())
                .andExpect(jsonPath("$.paths['/v2/health/ready'].get").exists())
                .andExpect(jsonPath("$.paths['/v2/models/{name}'].get").exists())
                .andExpect(jsonPath("$.paths['/v2/models/{name}/ready'].get").exists())
                .andExpect(jsonPath("$.paths['/v2/models/{name}/infer'].post").exists())
                .andExpect(jsonPath("$.paths['/v2/models/{name}/versions/{version}/infer'].post").exists())
                // Every endpoint carries at least one documented example.
                .andExpect(jsonPath("$.paths['/api/v1/embed'].post.responses.200.content['application/json'].examples")
                        .exists())
                .andExpect(jsonPath("$.components.schemas.EmbedRequest").exists())
                .andExpect(jsonPath("$.components.schemas.OipInferRequest").exists())
                .andExpect(jsonPath("$.components.schemas.ApiError").exists())
                .andExpect(jsonPath("$.components.schemas.BenchmarkReport").exists());
    }

    @Test
    void swaggerUiIsServed() throws Exception {
        mvc.perform(get("/swagger-ui/index.html")).andExpect(status().isOk());
    }
}
