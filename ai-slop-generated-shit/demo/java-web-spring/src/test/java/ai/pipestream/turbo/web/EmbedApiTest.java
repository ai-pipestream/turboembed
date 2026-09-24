// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.lessThan;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/** {@code POST /api/v1/embed} and {@code /similarity}, including every refusal. */
class EmbedApiTest extends ApiTestBase {
    private static MockHttpServletRequestBuilder embed(String body) {
        return post("/api/v1/embed").contentType(MediaType.APPLICATION_JSON).content(body);
    }

    @Test
    void embedReturnsUnitVectorsWithTheDeviceAndTheTimings() throws Exception {
        mvc.perform(embed("{\"texts\":[\"a brown dog runs through the grass\",\"the stock market closed higher\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value(EMBED_MODEL))
                .andExpect(jsonPath("$.dim").value(8))
                .andExpect(jsonPath("$.count").value(2))
                .andExpect(jsonPath("$.vectors", hasSize(2)))
                .andExpect(jsonPath("$.vectors[0]", hasSize(8)))
                .andExpect(jsonPath("$.placement").value("HOST"))
                .andExpect(jsonPath("$.device.name").value("Mock accelerator"))
                .andExpect(jsonPath("$.device.provider_id").value("mock"))
                .andExpect(jsonPath("$.timings.total_ms", greaterThanOrEqualTo(0.0)));
    }

    @Test
    void similarityIsSymmetricWithAUnitDiagonal() throws Exception {
        mvc.perform(post("/api/v1/similarity").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"a brown dog runs through the grass\",\"the stock market closed higher\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.similarity", hasSize(2)))
                .andExpect(jsonPath("$.similarity[0][0]", closeTo(1.0, 1e-4)))
                .andExpect(jsonPath("$.similarity[1][1]", closeTo(1.0, 1e-4)))
                .andExpect(jsonPath("$.similarity[0][1]", lessThan(1.0)))
                .andExpect(jsonPath("$.texts", hasSize(2)));
    }

    @Test
    void theBundlesTruncatedDimensionIsHonored() throws Exception {
        // The mock bundle's contract lists truncate_dims [4], and the device
        // advertises TURBO_CAP_OPT_OUTPUT_DIM, so 4 is served exactly.
        mvc.perform(embed("{\"texts\":[\"a brown dog\"],\"options\":{\"output_dim\":4}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.dim").value(4))
                .andExpect(jsonPath("$.vectors[0]", hasSize(4)));
    }

    @Test
    void aDimensionTheBundleDoesNotOfferIsRefusedNamingTheField() throws Exception {
        mvc.perform(embed("{\"texts\":[\"a brown dog\"],\"options\":{\"output_dim\":5}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value("TURBO_E_INVALID_ARGUMENT"))
                .andExpect(jsonPath("$.field").value(7))
                .andExpect(jsonPath("$.error", containsString("truncate_dims")));
    }

    @Test
    void anOptionTheDeviceDoesNotImplementIs501WithTheCodeAndTheField() throws Exception {
        // The mock advertises no TURBO_CAP_OPT_POOLING_OVERRIDE, and the
        // bundle pools with MEAN, so CLS is refused rather than substituted.
        mvc.perform(embed("{\"texts\":[\"a brown dog\"],\"options\":{\"pooling\":\"CLS\"}}"))
                .andExpect(status().isNotImplemented())
                .andExpect(jsonPath("$.status").value("TURBO_E_UNSUPPORTED_OPTION"))
                .andExpect(jsonPath("$.field").value(6))
                .andExpect(jsonPath("$.error", containsString("pooling")))
                .andExpect(jsonPath("$.path").value("/api/v1/embed"));
    }

    @Test
    void aRequestBeyondTheModelsSequenceLimitIs422() throws Exception {
        mvc.perform(embed("{\"texts\":[\"a brown dog\"],\"options\":{\"max_tokens\":99}}"))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.status").value("TURBO_E_CAPACITY"))
                .andExpect(jsonPath("$.field").value(3))
                .andExpect(jsonPath("$.error", containsString("max_seq 16")));
    }

    @Test
    void anUnknownEnumConstantIs400NamingTheFieldAndWhatItAccepts() throws Exception {
        mvc.perform(embed("{\"texts\":[\"a\"],\"options\":{\"pooling\":\"banana\"}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("pooling")))
                .andExpect(jsonPath("$.error", containsString("MODEL, MEAN, CLS, LAST")))
                .andExpect(jsonPath("$.error", containsString("banana")))
                .andExpect(jsonPath("$.status").doesNotExist());
    }

    @Test
    void anEmptyOrOversizedBatchIsRefusedBeforeAnyNativeCall() throws Exception {
        mvc.perform(embed("{\"texts\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("texts must not be empty"));
        mvc.perform(embed("{\"texts\":[\"a\",\"  \"]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("texts must not contain a blank string"));
        mvc.perform(embed("{\"texts\":[\"1\",\"2\",\"3\",\"4\",\"5\"]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("batch of 4")));
    }

    @Test
    void namingAModelOfAnotherTaskIs409AndAnUnknownNameIs404() throws Exception {
        mvc.perform(embed("{\"model\":\"rerank\",\"texts\":[\"a\"]}"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.error", containsString("performs RERANK")));
        mvc.perform(embed("{\"model\":\"nope\",\"texts\":[\"a\"]}"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("no model named `nope`")));
    }

    @Test
    void theSameTextsEmbedToTheSameVectors() throws Exception {
        String body = "{\"texts\":[\"a brown dog runs through the grass\"]}";
        String first = mvc.perform(embed(body)).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        String second = mvc.perform(embed(body)).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        // The timings differ, the vectors must not.
        org.junit.jupiter.api.Assertions.assertEquals(vectorsOf(first), vectorsOf(second),
                "the mock provider is documented as deterministic");
    }

    private static String vectorsOf(String body) {
        int start = body.indexOf("\"vectors\"");
        int end = body.indexOf("\"placement\"");
        return body.substring(start, end);
    }
}
