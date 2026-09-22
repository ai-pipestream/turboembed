// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;

/** {@code POST /api/v1/classify} and {@code /token-classify} on the mock label bundles. */
class ClassifyApiTest extends ApiTestBase {
    @Test
    void classifyReturnsTheBundlesLabelsBestFirst() throws Exception {
        mvc.perform(post("/api/v1/classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"the service was excellent\",\"the parcel never arrived\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("classify"))
                .andExpect(jsonPath("$.labels", hasSize(3)))
                .andExpect(jsonPath("$.labels[0]").value("negative"))
                .andExpect(jsonPath("$.results", hasSize(2)))
                .andExpect(jsonPath("$.results[0].scores", hasSize(3)))
                .andExpect(jsonPath("$.results[0].top").exists())
                .andExpect(jsonPath("$.results[0].index").value(0))
                .andExpect(jsonPath("$.placement").value("HOST"))
                .andExpect(jsonPath("$.timings.total_ms", greaterThanOrEqualTo(0.0)));
    }

    @Test
    void classifyOrdersEveryRowByScore() throws Exception {
        String body = mvc.perform(post("/api/v1/classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"the service was excellent\"]}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        com.fasterxml.jackson.databind.JsonNode scores = new com.fasterxml.jackson.databind.ObjectMapper()
                .readTree(body).get("results").get(0).get("scores");
        for (int i = 1; i < scores.size(); i++) {
            org.junit.jupiter.api.Assertions.assertTrue(
                    scores.get(i - 1).get("score").asDouble() >= scores.get(i).get("score").asDouble(),
                    "scores are not ordered best first: " + scores);
        }
    }

    @Test
    void tokenClassifyReturnsSpansWithByteOffsetsIntoTheirInput() throws Exception {
        mvc.perform(post("/api/v1/token-classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"Ada Lovelace worked in London\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("ner"))
                .andExpect(jsonPath("$.labels", hasSize(3)))
                .andExpect(jsonPath("$.spans[0].row").value(0))
                .andExpect(jsonPath("$.spans[0].label").exists())
                .andExpect(jsonPath("$.spans[0].text").exists())
                .andExpect(jsonPath("$.spans[0].byte_end", greaterThanOrEqualTo(1)))
                // The raw tensor is [batch, max_seq, labels].
                .andExpect(jsonPath("$.score_shape", hasSize(3)))
                .andExpect(jsonPath("$.score_shape[0]").value(1))
                .andExpect(jsonPath("$.score_shape[1]").value(16))
                .andExpect(jsonPath("$.score_shape[2]").value(3));
    }

    @Test
    void everySpanTextIsTheSliceItsOffsetsName() throws Exception {
        String text = "Ada Lovelace worked in London";
        String body = mvc.perform(post("/api/v1/token-classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"" + text + "\"]}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        com.fasterxml.jackson.databind.JsonNode spans =
                new com.fasterxml.jackson.databind.ObjectMapper().readTree(body).get("spans");
        org.junit.jupiter.api.Assertions.assertTrue(spans.size() > 0, "the mock found no spans: " + body);
        byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        for (com.fasterxml.jackson.databind.JsonNode span : spans) {
            int from = span.get("byte_start").asInt();
            int to = span.get("byte_end").asInt();
            String expected = new String(bytes, from, to - from, java.nio.charset.StandardCharsets.UTF_8);
            org.junit.jupiter.api.Assertions.assertEquals(expected, span.get("text").asText(),
                    "a span's text is not the slice its byte offsets name");
        }
    }

    @Test
    void aClassifierIsNotATokenClassifier() throws Exception {
        mvc.perform(post("/api/v1/token-classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"classify\",\"texts\":[\"a\"]}"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.error", containsString("performs CLASSIFY")));
    }

    @Test
    void anEmptyBatchOrAnUnknownAggregationIsRefusedWithTheReason() throws Exception {
        mvc.perform(post("/api/v1/classify").contentType(MediaType.APPLICATION_JSON).content("{\"texts\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("texts must not be empty"));
        mvc.perform(post("/api/v1/token-classify").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"a\"],\"options\":{\"aggregation\":\"clever\"}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("aggregation")))
                .andExpect(jsonPath("$.error", containsString("MODEL, NONE, SIMPLE, FIRST, MAX")));
    }
}
