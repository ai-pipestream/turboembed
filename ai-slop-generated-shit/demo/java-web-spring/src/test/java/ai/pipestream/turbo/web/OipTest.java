// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/**
 * The KServe Open Inference Protocol version 2 HTTP/REST surface: the six
 * endpoints of the protocol's HTTP/REST section and the inference mapping for
 * each model kind.
 */
class OipTest extends ApiTestBase {
    private static MockHttpServletRequestBuilder infer(String model, String body) {
        return post("/v2/models/" + model + "/infer").contentType(MediaType.APPLICATION_JSON).content(body);
    }

    @Test
    void serverMetadataNamesTheServerAndItsExtensions() throws Exception {
        mvc.perform(get("/v2"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("turbo"))
                .andExpect(jsonPath("$.version").exists())
                .andExpect(jsonPath("$.extensions", hasSize(0)));
    }

    @Test
    void liveAndReadyCarryTheProtocolsProbeObjects() throws Exception {
        mvc.perform(get("/v2/health/live"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.live").value(true));
        mvc.perform(get("/v2/health/ready"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.live").value(true))
                .andExpect(jsonPath("$.ready").value(true));
        mvc.perform(get("/v2/models/" + EMBED_MODEL + "/ready"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value(EMBED_MODEL))
                .andExpect(jsonPath("$.ready").value(true));
        mvc.perform(get("/v2/models/" + EMBED_MODEL + "/versions/1/ready"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.ready").value(true));
        mvc.perform(get("/v2/models/nope/ready")).andExpect(status().isNotFound());
    }

    @Test
    void modelMetadataDeclaresTheTensorsOfEachKind() throws Exception {
        mvc.perform(get("/v2/models/" + EMBED_MODEL))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value(EMBED_MODEL))
                .andExpect(jsonPath("$.versions", hasSize(1)))
                .andExpect(jsonPath("$.versions[0]").value("1"))
                .andExpect(jsonPath("$.platform").value("mock"))
                .andExpect(jsonPath("$.inputs[0].name").value("text"))
                .andExpect(jsonPath("$.inputs[0].datatype").value("BYTES"))
                .andExpect(jsonPath("$.inputs[0].shape[0]").value(-1))
                .andExpect(jsonPath("$.outputs[0].name").value("embeddings"))
                .andExpect(jsonPath("$.outputs[0].datatype").value("FP32"))
                .andExpect(jsonPath("$.outputs[0].shape[1]").value(8));

        mvc.perform(get("/v2/models/rerank/versions/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.inputs[0].name").value("query"))
                .andExpect(jsonPath("$.inputs[1].name").value("documents"))
                .andExpect(jsonPath("$.outputs[0].name").value("scores"))
                .andExpect(jsonPath("$.outputs[1].name").value("sorted"))
                .andExpect(jsonPath("$.outputs[1].datatype").value("INT32"));

        mvc.perform(get("/v2/models/" + GENERATE_MODEL))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.inputs[0].name").value("prompt"))
                .andExpect(jsonPath("$.inputs[1].name").value("messages"))
                .andExpect(jsonPath("$.outputs[0].name").value("text"))
                .andExpect(jsonPath("$.outputs[0].datatype").value("BYTES"));

        mvc.perform(get("/v2/models/ner"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs[0].shape", hasSize(3)))
                .andExpect(jsonPath("$.outputs[1].name").value("labels"));
    }

    @Test
    void embedInferReturnsOneFlattenedFp32Tensor() throws Exception {
        mvc.perform(infer(EMBED_MODEL, "{\"id\":\"req-1\",\"inputs\":[{\"name\":\"text\",\"shape\":[2],"
                        + "\"datatype\":\"BYTES\",\"data\":[\"a brown dog\",\"the stock market\"]}]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model_name").value(EMBED_MODEL))
                .andExpect(jsonPath("$.model_version").value("1"))
                .andExpect(jsonPath("$.id").value("req-1"))
                .andExpect(jsonPath("$.parameters.placement").value("HOST"))
                .andExpect(jsonPath("$.outputs", hasSize(1)))
                .andExpect(jsonPath("$.outputs[0].name").value("embeddings"))
                .andExpect(jsonPath("$.outputs[0].datatype").value("FP32"))
                .andExpect(jsonPath("$.outputs[0].shape[0]").value(2))
                .andExpect(jsonPath("$.outputs[0].shape[1]").value(8))
                .andExpect(jsonPath("$.outputs[0].data", hasSize(16)));
    }

    @Test
    void rerankInferReturnsScoresAndTheRanking() throws Exception {
        mvc.perform(infer("rerank", "{\"inputs\":[{\"name\":\"query\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"the accelerator is fast\"]},{\"name\":\"documents\",\"shape\":[2],"
                        + "\"datatype\":\"BYTES\",\"data\":[\"the accelerator is fast\",\"a pot of soup\"]}],"
                        + "\"parameters\":{\"return_sorted\":true}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs", hasSize(2)))
                .andExpect(jsonPath("$.outputs[0].name").value("scores"))
                .andExpect(jsonPath("$.outputs[0].datatype").value("FP32"))
                .andExpect(jsonPath("$.outputs[0].data", hasSize(2)))
                .andExpect(jsonPath("$.outputs[1].name").value("sorted"))
                .andExpect(jsonPath("$.outputs[1].datatype").value("INT32"))
                .andExpect(jsonPath("$.outputs[1].data[0]").value(0));
    }

    @Test
    void classifyInferReturnsScoresAndTheLabelSet() throws Exception {
        mvc.perform(infer("classify", "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"the service was excellent\"]}]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs[0].name").value("scores"))
                .andExpect(jsonPath("$.outputs[0].shape[1]").value(3))
                .andExpect(jsonPath("$.outputs[1].name").value("labels"))
                .andExpect(jsonPath("$.outputs[1].datatype").value("BYTES"))
                .andExpect(jsonPath("$.outputs[1].data[0]").value("negative"));
    }

    @Test
    void tokenClassifyInferReturnsThePerTokenTensor() throws Exception {
        mvc.perform(infer("ner", "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"Ada Lovelace worked in London\"]}]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs[0].name").value("scores"))
                .andExpect(jsonPath("$.outputs[0].shape", hasSize(3)))
                .andExpect(jsonPath("$.outputs[0].data", hasSize(48)))
                .andExpect(jsonPath("$.outputs[1].data", hasSize(3)));
    }

    @Test
    void generateInferTakesAPromptOrAChatAndCarriesTheFinishReason() throws Exception {
        mvc.perform(infer(GENERATE_MODEL, "{\"inputs\":[{\"name\":\"prompt\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"summarize this\"]}],\"parameters\":{\"max_tokens\":4}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs[0].name").value("text"))
                .andExpect(jsonPath("$.outputs[0].datatype").value("BYTES"))
                .andExpect(jsonPath("$.outputs[0].data", hasSize(1)))
                .andExpect(jsonPath("$.outputs[0].data[0]", containsString("tok")))
                .andExpect(jsonPath("$.parameters.finish_reason").value("LENGTH"))
                .andExpect(jsonPath("$.parameters.generated_tokens").value(4));

        mvc.perform(infer(GENERATE_MODEL, "{\"inputs\":[{\"name\":\"messages\",\"shape\":[2],\"datatype\":\"BYTES\","
                        + "\"data\":[\"{\\\"role\\\":\\\"system\\\",\\\"content\\\":\\\"be terse\\\"}\","
                        + "\"{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"hello\\\"}\"]}],"
                        + "\"parameters\":{\"max_tokens\":2}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.parameters.generated_tokens").value(2));
    }

    @Test
    void requestedOutputsNarrowTheResponse() throws Exception {
        mvc.perform(infer("classify", "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"excellent\"]}],\"outputs\":[{\"name\":\"labels\"}]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.outputs", hasSize(1)))
                .andExpect(jsonPath("$.outputs[0].name").value("labels"));
    }

    @Test
    void protocolErrorsCarryOnlyAnErrorString() throws Exception {
        mvc.perform(get("/v2/models/does-not-exist"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("no model named `does-not-exist`")))
                .andExpect(jsonPath("$.status").doesNotExist())
                .andExpect(jsonPath("$.path").doesNotExist());

        mvc.perform(get("/v2/models/" + EMBED_MODEL + "/versions/7"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("has no version `7`")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"FP32\","
                        + "\"data\":[1.0]}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("input `text` must be datatype BYTES, not FP32"));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"wrong\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("needs an input named `text`")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[3],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("declares shape")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("at least one entry in `inputs`")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}],\"outputs\":[{\"name\":\"nope\"}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("produces no output named `nope`")));
    }

    @Test
    void anOptionTheDeviceDoesNotImplementIsAProtocolErrorNamingTheField() throws Exception {
        // The protocol has no field for a status code or a field index, so both
        // travel inside the message rather than being dropped.
        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}],\"parameters\":{\"pooling\":\"CLS\"}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("TURBO_E_UNSUPPORTED_OPTION")))
                .andExpect(jsonPath("$.error", containsString("field 6")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}],\"parameters\":{\"pooling\":\"banana\"}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("must be one of [MODEL, MEAN, CLS, LAST]")));

        mvc.perform(infer(EMBED_MODEL, "{\"inputs\":[{\"name\":\"text\",\"shape\":[1],\"datatype\":\"BYTES\","
                        + "\"data\":[\"a\"]}],\"parameters\":{\"max_tokens\":\"lots\"}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("`max_tokens` must be an integer")));
    }
}
