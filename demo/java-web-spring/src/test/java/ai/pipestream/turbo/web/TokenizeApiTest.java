// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;

/**
 * {@code POST /api/v1/tokenize} and {@code /detokenize}. The embedding model is
 * configured with the committed MiniLM tokenizer bundle, because the mock
 * bundles declare a {@code mock} tokenizer with no {@code tokenizer.json}.
 */
class TokenizeApiTest extends ApiTestBase {
    @Test
    void tokenizeReturnsIdsMaskPiecesAndCounts() throws Exception {
        mvc.perform(post("/api/v1/tokenize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"a brown dog runs\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value(EMBED_MODEL))
                .andExpect(jsonPath("$.tokenizer_bundle", containsString("minilm-tokenizer")))
                .andExpect(jsonPath("$.tokenizer.kind").value("wordpiece"))
                .andExpect(jsonPath("$.tokenizer.vocab_size").value(30522))
                .andExpect(jsonPath("$.results", hasSize(1)))
                .andExpect(jsonPath("$.results[0].count").value(6))
                .andExpect(jsonPath("$.results[0].ids", hasSize(6)))
                .andExpect(jsonPath("$.results[0].mask", hasSize(6)))
                .andExpect(jsonPath("$.results[0].tokens[0]").value("[CLS]"))
                .andExpect(jsonPath("$.results[0].tokens[5]").value("[SEP]"))
                .andExpect(jsonPath("$.results[0].tokens[1]").value("a"))
                .andExpect(jsonPath("$.timings.total_ms", greaterThan(0.0)));
    }

    @Test
    void specialTokensCanBeLeftOut() throws Exception {
        mvc.perform(post("/api/v1/tokenize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"a brown dog runs\"],\"add_special_tokens\":false}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].count").value(4))
                .andExpect(jsonPath("$.results[0].tokens[0]").value("a"));
    }

    @Test
    void detokenizeIsTheInverse() throws Exception {
        mvc.perform(post("/api/v1/detokenize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"ids\":[[101,1037,2829,3899,102]]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.texts", hasSize(1)))
                .andExpect(jsonPath("$.texts[0]").value("a brown dog"))
                .andExpect(jsonPath("$.tokenizer_bundle", containsString("minilm-tokenizer")));
    }

    @Test
    void aBundleWithNoTokenizerJsonFailsWithTheLibrarysOwnRefusal() throws Exception {
        mvc.perform(post("/api/v1/tokenize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"rerank\",\"texts\":[\"a\"]}"))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.status").value("TURBO_E_BUNDLE_INVALID"))
                .andExpect(jsonPath("$.error", containsString("no `tokenizer.json` file entry")));
    }

    @Test
    void emptyInputIsRefusedWithTheReason() throws Exception {
        mvc.perform(post("/api/v1/tokenize").contentType(MediaType.APPLICATION_JSON).content("{\"texts\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("texts must not be empty"));
        mvc.perform(post("/api/v1/detokenize").contentType(MediaType.APPLICATION_JSON).content("{\"ids\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("ids must not be empty"));
        mvc.perform(post("/api/v1/detokenize").contentType(MediaType.APPLICATION_JSON).content("{\"ids\":[[]]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("ids[0] is empty"));
    }

    @Test
    void anUnknownModelNameIs404() throws Exception {
        mvc.perform(post("/api/v1/tokenize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"nope\",\"texts\":[\"a\"]}"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("no model named `nope`")));
    }
}
