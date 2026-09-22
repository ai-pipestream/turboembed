// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

/** The HTTP contract, against the mock bundle (surefire passes turbo.bundle; no hardware). */
@SpringBootTest(properties = {"turbo.max-batch=4", "turbo.sessions=1"})
@AutoConfigureMockMvc
class EmbedControllerTest {
    @Autowired
    private MockMvc mvc;

    @Test
    void infoNamesTheDeviceAndModel() throws Exception {
        mvc.perform(get("/api/info"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.providerId").value("mock"))
                .andExpect(jsonPath("$.deviceKind").value("ACCEL"))
                .andExpect(jsonPath("$.dim").value(8))
                .andExpect(jsonPath("$.maxBatch").value(4));
    }

    @Test
    void embedReturnsUnitVectorsAndASymmetricMatrix() throws Exception {
        mvc.perform(post("/api/embed").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"a brown dog runs through the grass\",\"  \",\"the stock market closed higher\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.texts", hasSize(2)))
                .andExpect(jsonPath("$.dim").value(8))
                .andExpect(jsonPath("$.vectors[0]", hasSize(8)))
                .andExpect(jsonPath("$.similarity[0][0]", closeTo(1.0, 1e-4)))
                .andExpect(jsonPath("$.similarity[1][1]", closeTo(1.0, 1e-4)))
                .andExpect(jsonPath("$.similarity[0][1]").value(org.hamcrest.Matchers.lessThan(1.0)));
    }

    @Test
    void emptyAndOversizedRequestsAreRefusedWithTheReason() throws Exception {
        mvc.perform(post("/api/embed").contentType(MediaType.APPLICATION_JSON).content("{\"texts\":[]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("no texts"));
        mvc.perform(post("/api/embed").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"texts\":[\"1\",\"2\",\"3\",\"4\",\"5\"]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("batch is 4")));
    }

    @Test
    void thePageIsServed() throws Exception {
        mvc.perform(get("/index.html")).andExpect(status().isOk());
    }
}
