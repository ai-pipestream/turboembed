// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;

/** {@code GET /api/v1/health}, {@code /devices} and {@code /models}. */
class MetaApiTest extends ApiTestBase {
    @Test
    void healthListsEveryLoadedModel() throws Exception {
        mvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ok"))
                .andExpect(jsonPath("$.abi_version", greaterThan(0)))
                .andExpect(jsonPath("$.device_count", greaterThan(0)))
                .andExpect(jsonPath("$.models", hasSize(5)))
                .andExpect(jsonPath("$.models", hasItem(EMBED_MODEL)))
                .andExpect(jsonPath("$.models", hasItem(GENERATE_MODEL)))
                .andExpect(jsonPath("$.models", hasItem("rerank")))
                .andExpect(jsonPath("$.models", hasItem("classify")))
                .andExpect(jsonPath("$.models", hasItem("ner")));
    }

    @Test
    void devicesCarryTheirFeatureBitsAndTheCapabilityMatrix() throws Exception {
        // The mock provider offers a CPU at ordinal 0 and an accelerator at 1;
        // AUTO never picks the CPU, so every model here runs on index 1.
        mvc.perform(get("/api/v1/devices"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[1].name").value("Mock accelerator"))
                .andExpect(jsonPath("$[1].kind").value("ACCEL"))
                .andExpect(jsonPath("$[1].provider_id").value("mock"))
                .andExpect(jsonPath("$[1].runtime_version").value("mock"))
                .andExpect(jsonPath("$[1].features", hasItem("DETERMINISTIC")))
                .andExpect(jsonPath("$[1].features", hasItem("OPT_TOP_N")))
                // The mock does not advertise a pooling override, which is why
                // an embed with pooling=CLS is refused naming field 6.
                .andExpect(jsonPath("$[1].features", org.hamcrest.Matchers.not(hasItem("OPT_POOLING_OVERRIDE"))))
                // Every task crossed with every modality, offered or not.
                .andExpect(jsonPath("$[1].capabilities", hasSize(32)))
                .andExpect(jsonPath("$[1].capabilities[?(@.task == 'EMBED' && @.modality == 'TEXT')].status")
                        .value(hasItem("SUPPORTED")))
                .andExpect(jsonPath("$[1].capabilities[?(@.task == 'EMBED' && @.modality == 'AUDIO')].status")
                        .value(hasItem("UNSUPPORTED")))
                .andExpect(jsonPath("$[1].capabilities[?(@.task == 'CHUNK' && @.modality == 'TEXT')].status")
                        .value(hasItem("UNSUPPORTED")))
                .andExpect(jsonPath("$[1].models", hasItem(EMBED_MODEL)));
    }

    @Test
    void modelsCarryTheWholeBundleContract() throws Exception {
        mvc.perform(get("/api/v1/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(5)))
                .andExpect(jsonPath("$[0].name").value(EMBED_MODEL))
                .andExpect(jsonPath("$[0].task").value("EMBED"))
                .andExpect(jsonPath("$[0].kind").value("EMBEDDING"))
                .andExpect(jsonPath("$[0].modality").value("TEXT"))
                .andExpect(jsonPath("$[0].dim").value(8))
                .andExpect(jsonPath("$[0].pooling").value("MEAN"))
                .andExpect(jsonPath("$[0].normalize").value("L2"))
                .andExpect(jsonPath("$[0].max_seq").value(16))
                .andExpect(jsonPath("$[0].max_batch").value(4))
                .andExpect(jsonPath("$[0].dtype").value("F32"))
                .andExpect(jsonPath("$[0].fully_accelerated").value(false))
                .andExpect(jsonPath("$[0].stage_placement.tokenize").value("host"))
                .andExpect(jsonPath("$[0].stage_placement.postprocess").value("unused"))
                .andExpect(jsonPath("$[0].prefix_query").value("query:"))
                .andExpect(jsonPath("$[0].prefix_document").value("passage:"))
                .andExpect(jsonPath("$[0].model_id").value("turbo/mock-embedding"))
                .andExpect(jsonPath("$[0].tokenizer_bundle", containsString("minilm-tokenizer")))
                .andExpect(jsonPath("$[0].device.provider_id").value("mock"))
                .andExpect(jsonPath("$[3].name").value("classify"))
                .andExpect(jsonPath("$[3].labels", hasSize(3)))
                .andExpect(jsonPath("$[3].labels[2]").value("positive"));
    }

    @Test
    void oneModelIsServedOnItsOwnPath() throws Exception {
        mvc.perform(get("/api/v1/models/ner"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("ner"))
                .andExpect(jsonPath("$.task").value("TOKEN_CLASSIFY"))
                .andExpect(jsonPath("$.labels[1]").value("PER"));
    }

    @Test
    void anUnknownModelNameIs404WithTheNamesThatDoExist() throws Exception {
        mvc.perform(get("/api/v1/models/does-not-exist"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error", containsString("no model named `does-not-exist`")))
                .andExpect(jsonPath("$.error", containsString(EMBED_MODEL)))
                .andExpect(jsonPath("$.path").value("/api/v1/models/does-not-exist"));
    }
}
