// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.Chunk;
import ai.pipestream.turbo.TurboException;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * {@code POST /api/summarize} streams the summary as server-sent events:
 * one {@code chunk} event per generation step ({"text": ..., "generated": n})
 * and a final {@code done} event ({"finish": "EOS"|"LENGTH"|..., "generated": n,
 * "promptTokens": n}); a failure mid-stream is an {@code error} event with
 * the library's message. Without a generative bundle the call is 409.
 */
@RestController
public class SummarizeController {
    public record SummarizeRequest(String text, int maxNewTokens) {}

    private final TurboService turbo;
    private final ExecutorService workers = Executors.newCachedThreadPool();

    public SummarizeController(TurboService turbo) {
        this.turbo = turbo;
    }

    @PostMapping("/api/summarize")
    public ResponseEntity<?> summarize(@RequestBody SummarizeRequest req) {
        if (!turbo.canGenerate()) {
            return ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of("error", "no generative bundle is configured (turbo.generate-bundle)"));
        }
        if (req.text() == null || req.text().isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "no text"));
        }
        SseEmitter emitter = new SseEmitter(0L);
        workers.submit(() -> {
            try {
                Chunk last = turbo.summarize(req.text(), req.maxNewTokens(), chunk -> {
                    try {
                        if (!chunk.text().isEmpty()) {
                            emitter.send(SseEmitter.event().name("chunk").data(Map.of("text", chunk.text(), "generated", chunk.generatedTokens())));
                        }
                        return true;
                    } catch (IOException e) {
                        return false; // the client went away: cancel the generation
                    }
                });
                emitter.send(SseEmitter.event().name("done").data(Map.of("finish", last.finishReason().name(), "generated", last.generatedTokens(), "promptTokens", last.promptTokens())));
                emitter.complete();
            } catch (IOException e) {
                emitter.complete(); // the client went away after the last chunk
            } catch (TurboException e) {
                fail(emitter, e.getMessage()); // already carries the status name and field
            } catch (TurboService.Overloaded | IllegalArgumentException | IllegalStateException e) {
                fail(emitter, e.getMessage());
            }
        });
        return ResponseEntity.ok().contentType(MediaType.TEXT_EVENT_STREAM).body(emitter);
    }

    private static void fail(SseEmitter emitter, String message) {
        try {
            emitter.send(SseEmitter.event().name("error").data(Map.of("error", message)));
        } catch (IOException ignored) {
            // the client is gone; nothing left to tell
        }
        emitter.complete();
    }
}
