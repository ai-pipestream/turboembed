// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/** {@code POST /api/v1/rerank} on the mock reranker, whose score is query token overlap. */
class RerankApiTest extends ApiTestBase {
    private static final String DOCUMENTS =
            "[\"the accelerator is fast\",\"a pot of soup on the stove\",\"the accelerator is fast and quiet\"]";

    private static MockHttpServletRequestBuilder rerank(String body) {
        return post("/api/v1/rerank").contentType(MediaType.APPLICATION_JSON).content(body);
    }

    @Test
    void scoresComeBackInInputOrderWithTheDeviceAndTheTimings() throws Exception {
        mvc.perform(rerank("{\"query\":\"the accelerator is fast\",\"documents\":" + DOCUMENTS + "}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("rerank"))
                .andExpect(jsonPath("$.query").value("the accelerator is fast"))
                .andExpect(jsonPath("$.results", hasSize(3)))
                .andExpect(jsonPath("$.results[0].index").value(0))
                .andExpect(jsonPath("$.results[1].document").value("a pot of soup on the stove"))
                .andExpect(jsonPath("$.results[0].rank").doesNotExist())
                .andExpect(jsonPath("$.sorted").doesNotExist())
                .andExpect(jsonPath("$.placement").value("HOST"))
                .andExpect(jsonPath("$.device.provider_id").value("mock"))
                .andExpect(jsonPath("$.timings.run_ms").exists());
    }

    @Test
    void returnSortedAddsTheRankingBestFirst() throws Exception {
        mvc.perform(rerank("{\"query\":\"the accelerator is fast\",\"documents\":" + DOCUMENTS
                        + ",\"options\":{\"return_sorted\":true}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sorted", hasSize(3)))
                // Every document overlapping the query outranks the soup.
                .andExpect(jsonPath("$.sorted[2]").value(1))
                .andExpect(jsonPath("$.results[1].rank").value(2));
    }

    @Test
    void topNKeepsOnlyTheBestDocuments() throws Exception {
        mvc.perform(rerank("{\"query\":\"the accelerator is fast\",\"documents\":" + DOCUMENTS
                        + ",\"options\":{\"top_n\":2,\"return_sorted\":true}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results", hasSize(3)))
                .andExpect(jsonPath("$.sorted", hasSize(2)));
    }

    @Test
    void rawScoresReturnsTheLogitsRatherThanTheBundlesActivation() throws Exception {
        String activated = mvc.perform(rerank("{\"query\":\"the accelerator is fast\",\"documents\":" + DOCUMENTS + "}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        String raw = mvc.perform(rerank("{\"query\":\"the accelerator is fast\",\"documents\":" + DOCUMENTS
                        + ",\"options\":{\"raw_scores\":true}}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        org.junit.jupiter.api.Assertions.assertNotEquals(activated, raw,
                "raw_scores must change the numbers, not be ignored");
    }

    @Test
    void aBlankQueryOrAnEmptyDocumentListIsRefusedWithTheReason() throws Exception {
        mvc.perform(rerank("{\"query\":\"  \",\"documents\":[\"a\"]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("query must not be blank"));
        mvc.perform(rerank("{\"query\":\"q\",\"documents\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("documents must not be empty"));
    }

    @Test
    void topNAboveTheDocumentCountIsRefusedBeforeAnyNativeCall() throws Exception {
        mvc.perform(rerank("{\"query\":\"q\",\"documents\":[\"a\",\"b\"],\"options\":{\"top_n\":5}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("top_n is 5")));
    }

    @Test
    void noRerankerUnderThatNameIs404() throws Exception {
        mvc.perform(rerank("{\"model\":\"nope\",\"query\":\"q\",\"documents\":[\"a\"]}"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("no model named `nope`")));
    }
}
