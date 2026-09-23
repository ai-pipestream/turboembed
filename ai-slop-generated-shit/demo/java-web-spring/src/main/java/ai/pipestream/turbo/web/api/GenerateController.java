// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.Chunk;
import ai.pipestream.turbo.GenerateDesc;
import ai.pipestream.turbo.Message;
import ai.pipestream.turbo.Task;
import ai.pipestream.turbo.TurboException;
import ai.pipestream.turbo.web.LoadedModel;
import ai.pipestream.turbo.web.TurboService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.annotation.PreDestroy;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/** Text generation, in one response or streamed as server-sent events. */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Generate", description = "Text generation on a generative bundle, whole or streamed")
public class GenerateController {
    /** A prompt or a chat, the model to run it on, and the sampling parameters. */
    @Schema(name = "GenerateRequest", description = "A prompt or a chat to continue")
    public record GenerateRequest(
            @Schema(description = "Model name from GET /api/v1/models; omitted means the first generative model loaded",
                    example = "mock-generative")
            String model,
            @Schema(description = "A single user turn. Give this or messages, not both.",
                    example = "Summarize the paragraph in two sentences.")
            String prompt,
            @Schema(description = "A chat, rendered through the bundle's own chat template. Give this or prompt, not both.")
            List<@Valid ChatMessageDto> messages,
            @Valid GenerateOptionsDto options) {}

    /** Tokens in and out. */
    @Schema(name = "Usage", description = "Tokens the prompt occupied and tokens generated")
    public record Usage(int promptTokens, int generatedTokens, int totalTokens) {}

    /** The whole generation. */
    @Schema(name = "GenerateResponse", description = "The generated text, why it stopped, and what it cost")
    public record GenerateResponse(String model, TurboService.DeviceRef device, String text, String finishReason,
            Usage usage, List<Float> logprobs, TurboService.GenTimings timings) {}

    private static final String EXAMPLE = """
            {"model":"mock-generative","device":{"index":1,"name":"Mock accelerator","provider_id":"mock","ordinal":1},\
            "text":"tok417 tok233 tok882","finish_reason":"LENGTH",\
            "usage":{"prompt_tokens":24,"generated_tokens":3,"total_tokens":27},"logprobs":[],\
            "timings":{"first_token_ms":0.4,"total_ms":1.9,"tokens_per_second":1578.9}}""";

    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(GenerateController.class);

    private final TurboService turbo;
    private final ExecutorService workers = Executors.newCachedThreadPool(Thread.ofPlatform().name("generate-", 0).factory());

    public GenerateController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/generate")
    @Operation(summary = "Generate text and return it whole",
            description = "Runs the generation to its finish reason and answers once. Every sampling parameter is "
                    + "gated by a TURBO_CAP_OPT_GEN_* bit: a device that does not implement one refuses the call "
                    + "with the field index rather than sampling differently. Use /api/v1/generate/stream to see "
                    + "the tokens as they are produced.")
    @ApiResponse(responseCode = "200", description = "The generated text",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = GenerateResponse.class),
                    examples = @ExampleObject(name = "three tokens from the mock generative bundle", value = EXAMPLE)))
    @ApiResponse(responseCode = "400", description = "Neither prompt nor messages, both, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "409", description = "No loaded model performs GENERATE",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "422", description = "The prompt exceeds the model's max_seq (TURBO_E_CAPACITY)",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "501", description = "The device does not implement a sampling parameter that was set",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "503", description = "Every generation slot of the model is busy",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public GenerateResponse generate(@Valid @RequestBody GenerateRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.GENERATE);
        TurboService.GenerateResult r = turbo.generate(m, messages(req), desc(req), chunk -> true);
        return new GenerateResponse(r.model(), r.device(), r.text(), r.finishReason(),
                new Usage(r.promptTokens(), r.generatedTokens(), r.promptTokens() + r.generatedTokens()),
                r.logprobs(), r.timings());
    }

    @PostMapping(value = "/generate/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    @Operation(summary = "Generate text as a stream of server-sent events",
            description = """
                    The same call as /api/v1/generate, streamed. One `chunk` event per generation step carries the \
                    new text, the new token ids and the running count; a single `done` event ends the stream with \
                    the finish reason, the token counts, the whole text and the timings. A failure after the \
                    headers are sent cannot change the status code, so it arrives as an `error` event carrying the \
                    library's status name and field index, and the stream then ends. A client that disconnects \
                    cancels the generation on the device: the next step reports CANCELLED and the slot is freed.""")
    @ApiResponse(responseCode = "200", description = "The event stream",
            content = @Content(mediaType = MediaType.TEXT_EVENT_STREAM_VALUE,
                    examples = @ExampleObject(name = "two steps then done",
                            value = """
                                    event:chunk
                                    data:{"text":"tok417 ","tokens":[417],"generated":1}

                                    event:chunk
                                    data:{"text":"tok233 ","tokens":[233],"generated":2}

                                    event:done
                                    data:{"finish_reason":"LENGTH","generated_tokens":2,"prompt_tokens":24,\
                                    "text":"tok417 tok233 ","timings":{"total_ms":1.2}}
                                    """)))
    @ApiResponse(responseCode = "400", description = "Neither prompt nor messages, both, or a malformed field",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    @ApiResponse(responseCode = "409", description = "No loaded model performs GENERATE",
            content = @Content(mediaType = "application/json", schema = @Schema(implementation = ApiError.class)))
    public SseEmitter stream(@Valid @RequestBody GenerateRequest req) {
        LoadedModel m = turbo.modelFor(req.model(), Task.GENERATE);
        List<Message> messages = messages(req);
        GenerateDesc desc = desc(req);
        SseEmitter emitter = new SseEmitter(0L);
        AtomicBoolean gone = new AtomicBoolean(false);
        emitter.onError(e -> gone.set(true));
        emitter.onTimeout(() -> gone.set(true));
        workers.submit(() -> {
            StringBuilder text = new StringBuilder();
            try {
                TurboService.GenerateResult r = turbo.generate(m, messages, desc, chunk -> {
                    text.append(chunk.text());
                    if (gone.get()) {
                        return false;
                    }
                    try {
                        emitter.send(SseEmitter.event().name("chunk").data(chunkEvent(chunk)));
                        return true;
                    } catch (IOException e) {
                        // The client went away: cancel the generation on the device.
                        gone.set(true);
                        return false;
                    }
                });
                if (gone.get()) {
                    emitter.complete();
                    return;
                }
                emitter.send(SseEmitter.event().name("done").data(doneEvent(r)));
                emitter.complete();
            } catch (IOException e) {
                // Nothing can be told to a client that is already gone.
                emitter.complete();
            } catch (TurboException e) {
                fail(emitter, e.getMessage(), e.statusName(), e.field());
            } catch (TurboService.Overloaded | TurboService.TaskNotServed | TurboService.ModelNotFound
                    | IllegalArgumentException | IllegalStateException e) {
                fail(emitter, e.getMessage(), null, null);
            } catch (RuntimeException e) {
                LOG.error("generation on model {} failed", m.name(), e);
                fail(emitter, e.getClass().getSimpleName() + ": " + e.getMessage(), null, null);
            }
        });
        return emitter;
    }

    private static Map<String, Object> chunkEvent(Chunk chunk) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("text", chunk.text());
        data.put("tokens", chunk.tokens());
        data.put("generated", chunk.generatedTokens());
        if (chunk.logprobs().length > 0) {
            data.put("logprobs", chunk.logprobs());
        }
        return data;
    }

    private static Map<String, Object> doneEvent(TurboService.GenerateResult r) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("finish_reason", r.finishReason());
        data.put("generated_tokens", r.generatedTokens());
        data.put("prompt_tokens", r.promptTokens());
        data.put("total_tokens", r.promptTokens() + r.generatedTokens());
        data.put("text", r.text());
        data.put("model", r.model());
        data.put("device", r.device());
        data.put("timings", r.timings());
        if (!r.logprobs().isEmpty()) {
            data.put("logprobs", r.logprobs());
        }
        return data;
    }

    private static void fail(SseEmitter emitter, String message, String status, Integer field) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("error", message);
        if (status != null) {
            data.put("status", status);
        }
        if (field != null && field != 0) {
            data.put("field", field);
        }
        try {
            emitter.send(SseEmitter.event().name("error").data(data));
        } catch (IOException e) {
            // The client is gone; the refusal has nowhere to go and the stream ends below.
        }
        emitter.complete();
    }

    /** The chat the request means: its messages, or its prompt as one user turn. */
    private static List<Message> messages(GenerateRequest req) {
        boolean hasPrompt = req.prompt() != null && !req.prompt().isBlank();
        boolean hasMessages = req.messages() != null && !req.messages().isEmpty();
        if (hasPrompt == hasMessages) {
            throw new IllegalArgumentException(hasPrompt
                    ? "give prompt or messages, not both"
                    : "give prompt (a single user turn) or messages (a chat)");
        }
        if (hasPrompt) {
            return List.of(Message.user(req.prompt().strip()));
        }
        List<Message> out = new ArrayList<>(req.messages().size());
        for (int i = 0; i < req.messages().size(); i++) {
            ChatMessageDto m = req.messages().get(i);
            if (m == null) {
                throw new IllegalArgumentException("messages[" + i + "] is null");
            }
            out.add(m.toTurbo());
        }
        return out;
    }

    private static GenerateDesc desc(GenerateRequest req) {
        return (req.options() == null ? GenerateOptionsDto.defaults() : req.options()).toTurbo();
    }

    @PreDestroy
    void shutdown() {
        workers.shutdownNow();
    }
}
