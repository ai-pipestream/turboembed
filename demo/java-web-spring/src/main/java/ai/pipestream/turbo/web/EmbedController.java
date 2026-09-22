// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.TurboException;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/** The JSON API the page uses: {@code GET /api/info} and {@code POST /api/embed}. */
@RestController
public class EmbedController {
    public record EmbedRequest(List<String> texts) {}

    public record EmbedResponse(int dim, float[][] vectors, double[][] similarity, List<String> texts) {}

    private final TurboService turbo;

    public EmbedController(TurboService turbo) {
        this.turbo = turbo;
    }

    @GetMapping("/api/info")
    public TurboService.Info info() {
        return turbo.info();
    }

    @PostMapping("/api/embed")
    public EmbedResponse embed(@RequestBody EmbedRequest req) {
        List<String> texts = req.texts() == null ? List.of() : req.texts().stream().map(String::trim).filter(t -> !t.isEmpty()).toList();
        TurboService.Embedded e = turbo.embed(texts);
        return new EmbedResponse(e.dim(), e.vectors(), e.similarity(), texts);
    }

    /** libturbo refusals: options and capacity are the client's problem, the rest is the server's. */
    @ExceptionHandler(TurboException.class)
    public ResponseEntity<Map<String, Object>> turbo(TurboException e) {
        HttpStatus status = switch (e.statusName()) {
            case "TURBO_E_INVALID_ARGUMENT", "TURBO_E_INVALID_UTF8", "TURBO_E_INVALID_ENUM", "TURBO_E_UNSUPPORTED_OPTION", "TURBO_E_CAPACITY" -> HttpStatus.BAD_REQUEST;
            case "TURBO_E_BUSY", "TURBO_E_OVERLOADED" -> HttpStatus.SERVICE_UNAVAILABLE;
            default -> HttpStatus.INTERNAL_SERVER_ERROR;
        };
        return ResponseEntity.status(status).body(Map.of("error", e.getMessage(), "status", e.statusName(), "field", e.field()));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> bad(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
    }

    @ExceptionHandler(TurboService.Overloaded.class)
    public ResponseEntity<Map<String, Object>> overloaded(TurboService.Overloaded e) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(Map.of("error", e.getMessage()));
    }
}
