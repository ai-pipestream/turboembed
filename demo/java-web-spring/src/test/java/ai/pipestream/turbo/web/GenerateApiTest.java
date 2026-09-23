// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/**
 * {@code POST /api/v1/generate} and {@code /generate/stream} against the mock
 * generative bundle, which emits one deterministic {@code tokNNN} piece per
 * step and stops at {@code max_tokens}.
 */
class GenerateApiTest extends ApiTestBase {
    @Autowired
    private TurboService turbo;

    private static MockHttpServletRequestBuilder generate(String body) {
        return post("/api/v1/generate").contentType(MediaType.APPLICATION_JSON).content(body);
    }

    @Test
    void generateReturnsTheWholeTextItsFinishReasonAndItsUsage() throws Exception {
        mvc.perform(generate("{\"prompt\":\"summarize the paragraph\",\"options\":{\"max_tokens\":6}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value(GENERATE_MODEL))
                .andExpect(jsonPath("$.finish_reason").value("LENGTH"))
                .andExpect(jsonPath("$.usage.generated_tokens").value(6))
                .andExpect(jsonPath("$.usage.prompt_tokens", greaterThan(0)))
                .andExpect(jsonPath("$.usage.total_tokens", greaterThan(6)))
                .andExpect(jsonPath("$.text", containsString("tok")))
                .andExpect(jsonPath("$.device.provider_id").value("mock"))
                .andExpect(jsonPath("$.timings.tokens_per_second", greaterThan(0.0)));
    }

    @Test
    void aChatGoesThroughTheBundlesChatTemplate() throws Exception {
        // The system turn lengthens the prompt, which is how the template is
        // observable from here: the same user turn costs more tokens with it.
        String alone = mvc.perform(generate(
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"options\":{\"max_tokens\":1}}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        String withSystem = mvc.perform(generate("{\"messages\":[{\"role\":\"system\",\"content\":\"be terse\"},"
                        + "{\"role\":\"user\",\"content\":\"hello\"}],\"options\":{\"max_tokens\":1}}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        com.fasterxml.jackson.databind.ObjectMapper json = new com.fasterxml.jackson.databind.ObjectMapper();
        int one = json.readTree(alone).get("usage").get("prompt_tokens").asInt();
        int two = json.readTree(withSystem).get("usage").get("prompt_tokens").asInt();
        org.junit.jupiter.api.Assertions.assertTrue(two > one,
                "the system turn did not reach the prompt: " + one + " then " + two);
    }

    @Test
    void samplingParametersTheDeviceHonorsChangeTheOutput() throws Exception {
        String greedy = mvc.perform(generate("{\"prompt\":\"hello\",\"options\":{\"max_tokens\":8}}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        String seeded = mvc.perform(generate(
                        "{\"prompt\":\"hello\",\"options\":{\"max_tokens\":8,\"temperature\":0.8,\"seed\":7}}"))
                .andExpect(status().isOk()).andReturn().getResponse().getContentAsString();
        com.fasterxml.jackson.databind.ObjectMapper json = new com.fasterxml.jackson.databind.ObjectMapper();
        org.junit.jupiter.api.Assertions.assertNotEquals(
                json.readTree(greedy).get("text").asText(), json.readTree(seeded).get("text").asText(),
                "seeded sampling produced the greedy text, so the seed was ignored");
    }

    @Test
    void neitherPromptNorMessagesAndBothAreRefusedWithTheReason() throws Exception {
        mvc.perform(generate("{}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("give prompt (a single user turn) or messages (a chat)"));
        mvc.perform(generate("{\"prompt\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"b\"}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("give prompt or messages, not both"));
        mvc.perform(generate("{\"messages\":[{\"role\":\"\",\"content\":\"b\"}]}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", containsString("messages[].role must not be blank")));
    }

    @Test
    void aPromptBeyondTheModelsContextIs422() throws Exception {
        String long_ = "word ".repeat(600);
        mvc.perform(generate("{\"prompt\":\"" + long_ + "\",\"options\":{\"max_tokens\":2}}"))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.status").value("TURBO_E_CAPACITY"))
                .andExpect(jsonPath("$.error", containsString("max_seq is 512")));
    }

    @Test
    void streamSendsEveryChunkBeforeOneDoneEvent() throws Exception {
        MvcResult started = mvc.perform(post("/api/v1/generate/stream").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"summarize the paragraph\",\"options\":{\"max_tokens\":12}}"))
                .andExpect(request().asyncStarted())
                .andReturn();
        // An SseEmitter sets its async result on complete(); the emitter has no
        // timeout of its own, so wait for the result explicitly.
        started.getAsyncResult(10_000);
        String body = mvc.perform(asyncDispatch(started))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(MediaType.TEXT_EVENT_STREAM))
                .andReturn().getResponse().getContentAsString();

        org.junit.jupiter.api.Assertions.assertFalse(body.contains("event:error"), "the stream carried an error: " + body);
        org.junit.jupiter.api.Assertions.assertTrue(body.indexOf("event:chunk") < body.indexOf("event:done"),
                "chunks must come before done: " + body);
        org.junit.jupiter.api.Assertions.assertEquals(1, body.split("event:done", -1).length - 1,
                "there must be exactly one done event: " + body);
        org.junit.jupiter.api.Assertions.assertEquals(12, body.split("event:chunk", -1).length - 1,
                "one chunk per generated token: " + body);
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("\"finish_reason\":\"LENGTH\""), body);
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("\"generated_tokens\":12"), body);
    }

    @Test
    void aStreamThatFailsMidwayEndsWithAnErrorEventCarryingTheCode() throws Exception {
        // The prompt is over the model's max_seq, and the refusal arrives after
        // the headers, so it cannot be a status code.
        MvcResult started = mvc.perform(post("/api/v1/generate/stream").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"prompt\":\"" + "word ".repeat(600) + "\",\"options\":{\"max_tokens\":2}}"))
                .andExpect(request().asyncStarted())
                .andReturn();
        started.getAsyncResult(10_000);
        String body = mvc.perform(asyncDispatch(started)).andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("event:error"), "expected an error event: " + body);
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("TURBO_E_CAPACITY"), body);
        org.junit.jupiter.api.Assertions.assertTrue(body.contains("\"status\":\"TURBO_E_CAPACITY\""), body);
        org.junit.jupiter.api.Assertions.assertFalse(body.contains("event:done"), body);
    }

    @Test
    void aSinkThatStopsCancelsTheGenerationOnTheDevice() {
        // This is the path a disconnected client takes: the controller's sink
        // returns false when the browser has gone, which cancels the
        // generation rather than running it to the end on a dead connection.
        // The mock model generates faster than any client can disconnect, so
        // the mechanism is exercised here instead of over HTTP.
        LoadedModel model = turbo.modelFor(GENERATE_MODEL, ai.pipestream.turbo.Task.GENERATE);
        TurboService.GenerateResult r = turbo.generate(model,
                java.util.List.of(ai.pipestream.turbo.Message.user("summarize the paragraph")),
                ai.pipestream.turbo.GenerateDesc.defaults().withMaxNewTokens(64),
                chunk -> chunk.generatedTokens() < 3);
        org.junit.jupiter.api.Assertions.assertEquals("CANCELLED", r.finishReason());
        org.junit.jupiter.api.Assertions.assertTrue(r.generatedTokens() < 64,
                "the generation ran to its budget although the sink stopped it");

        // The slot went back, so the next generation still runs to LENGTH.
        TurboService.GenerateResult after = turbo.generate(model,
                java.util.List.of(ai.pipestream.turbo.Message.user("summarize the paragraph")),
                ai.pipestream.turbo.GenerateDesc.defaults().withMaxNewTokens(4),
                chunk -> true);
        org.junit.jupiter.api.Assertions.assertEquals("LENGTH", after.finishReason());
        org.junit.jupiter.api.Assertions.assertEquals(4, after.generatedTokens());
    }

    @Test
    void aBadRequestIsRefusedBeforeTheStreamOpens() throws Exception {
        mvc.perform(post("/api/v1/generate/stream").contentType(MediaType.APPLICATION_JSON).content("{}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("give prompt (a single user turn) or messages (a chat)"));
        mvc.perform(post("/api/v1/generate/stream").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"model\":\"rerank\",\"prompt\":\"a\"}"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.error", containsString("performs RERANK")));
    }
}
