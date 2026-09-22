// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/** The streaming summary contract against the mock generative bundle (surefire passes turbo.generate-bundle). */
@SpringBootTest(properties = {"turbo.max-batch=4", "turbo.sessions=1", "turbo.generations=1"})
@AutoConfigureMockMvc
class SummarizeControllerTest {
    @Autowired
    private MockMvc mvc;

    @Test
    void summaryStreamsChunksAndEndsWithAFinishReason() throws Exception {
        MvcResult started = mvc.perform(post("/api/summarize").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\":\"The quick brown fox jumps over the lazy dog. It does so every single day.\",\"maxNewTokens\":12}"))
                .andExpect(request().asyncStarted())
                .andReturn();
        // An SseEmitter sets its async result on complete(); wait for it explicitly
        // (the emitter has no timeout of its own, so MockMvc would not).
        started.getAsyncResult(10_000);
        String body = mvc.perform(asyncDispatch(started))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(MediaType.TEXT_EVENT_STREAM))
                .andExpect(content().string(containsString("event:chunk")))
                .andExpect(content().string(containsString("event:done")))
                .andExpect(content().string(containsString("\"finish\":\"LENGTH\"")))
                .andReturn().getResponse().getContentAsString();
        assertTrue(body.indexOf("event:chunk") < body.indexOf("event:done"), "chunks come before done: " + body);
        assertTrue(!body.contains("event:error"), "no error event: " + body);
    }

    @Test
    void emptyTextIsRefusedBeforeAnyStream() throws Exception {
        mvc.perform(post("/api/summarize").contentType(MediaType.APPLICATION_JSON).content("{\"text\":\"   \"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("no text"));
    }

    @Test
    void infoNamesTheGenerativeModel() throws Exception {
        mvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get("/api/info"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.generate.providerId").value("mock"))
                .andExpect(jsonPath("$.generate.modelId").value(containsString("mock")));
    }
}
