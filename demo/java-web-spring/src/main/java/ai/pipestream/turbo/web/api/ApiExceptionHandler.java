// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.api;

import ai.pipestream.turbo.TurboException;
import ai.pipestream.turbo.web.TurboService;
import ai.pipestream.turbo.web.TurboStatus;
import com.fasterxml.jackson.databind.exc.InvalidFormatException;
import com.fasterxml.jackson.databind.exc.MismatchedInputException;
import jakarta.servlet.http.HttpServletRequest;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Every refusal under {@code /api/v1}, with the reason intact.
 *
 * <p>A libturbo refusal keeps its {@code TURBO_E_*} name and its 1-based field
 * index in the body, and its HTTP status follows {@link TurboStatus}: a
 * malformed request is 400, a request beyond the model's contract is 422, an
 * option the device does not implement is 501, a full queue is 503. Nothing is
 * caught and dropped, and no refusal is turned into a default.
 */
@RestControllerAdvice(basePackages = "ai.pipestream.turbo.web.api")
public class ApiExceptionHandler {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(ApiExceptionHandler.class);

    /** libturbo refused: the status name and the field index travel with the message. */
    @ExceptionHandler(TurboException.class)
    public ResponseEntity<ApiError> turbo(TurboException e, HttpServletRequest request) {
        HttpStatus status = TurboStatus.of(e);
        if (status.is5xxServerError() && status != HttpStatus.NOT_IMPLEMENTED) {
            LOG.error("{} on {}", e.statusName(), request.getRequestURI(), e);
        }
        return ResponseEntity.status(status)
                .body(new ApiError(e.getMessage(), e.statusName(), e.field() == 0 ? null : e.field(),
                        request.getRequestURI()));
    }

    /** A name this server does not serve. */
    @ExceptionHandler(TurboService.ModelNotFound.class)
    public ResponseEntity<ApiError> notFound(TurboService.ModelNotFound e, HttpServletRequest request) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ApiError(e.getMessage(), null, null, request.getRequestURI()));
    }

    /** A task no loaded model performs: a configuration answer, not a request answer. */
    @ExceptionHandler(TurboService.TaskNotServed.class)
    public ResponseEntity<ApiError> notServed(TurboService.TaskNotServed e, HttpServletRequest request) {
        return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(new ApiError(e.getMessage(), null, null, request.getRequestURI()));
    }

    /** Every session or generation slot is in use; the caller may retry. */
    @ExceptionHandler(TurboService.Overloaded.class)
    public ResponseEntity<ApiError> overloaded(TurboService.Overloaded e, HttpServletRequest request) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(new ApiError(e.getMessage(), null, null, request.getRequestURI()));
    }

    /** The server got into a state it cannot serve from; the log carries the stack. */
    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<ApiError> state(IllegalStateException e, HttpServletRequest request) {
        LOG.error("request to {} failed", request.getRequestURI(), e);
        return ResponseEntity.internalServerError()
                .body(new ApiError(message(e), null, null, request.getRequestURI()));
    }

    /** A request this server checked before any native call. */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> bad(IllegalArgumentException e, HttpServletRequest request) {
        return ResponseEntity.badRequest().body(new ApiError(message(e), null, null, request.getRequestURI()));
    }

    /** Bean validation: every violated constraint, in one message. */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiError> invalid(MethodArgumentNotValidException e, HttpServletRequest request) {
        List<String> problems = new ArrayList<>();
        for (FieldError f : e.getBindingResult().getFieldErrors()) {
            problems.add(f.getDefaultMessage());
        }
        e.getBindingResult().getGlobalErrors().forEach(g -> problems.add(g.getDefaultMessage()));
        String message = problems.isEmpty() ? e.getMessage() : String.join("; ", problems);
        return ResponseEntity.badRequest().body(new ApiError(message, null, null, request.getRequestURI()));
    }

    /**
     * A body Jackson could not read. An unknown enum constant names the field
     * and lists what the field accepts, rather than reporting "bad request".
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ApiError> unreadable(HttpMessageNotReadableException e, HttpServletRequest request) {
        return ResponseEntity.badRequest()
                .body(new ApiError(explain(e), null, null, request.getRequestURI()));
    }

    private static String explain(HttpMessageNotReadableException e) {
        Throwable cause = e.getCause();
        if (cause instanceof InvalidFormatException f) {
            String path = path(f);
            Class<?> target = f.getTargetType();
            if (target != null && target.isEnum()) {
                String allowed = java.util.Arrays.stream(target.getEnumConstants())
                        .map(String::valueOf)
                        .collect(Collectors.joining(", "));
                return path + " must be one of [" + allowed + "], not " + quote(f.getValue());
            }
            return path + " must be a " + (target == null ? "different type" : target.getSimpleName()) + ", not "
                    + quote(f.getValue());
        }
        if (cause instanceof MismatchedInputException m) {
            String path = path(m);
            return path.isEmpty() ? "the request body is not the expected shape: " + m.getOriginalMessage()
                    : path + " is not the expected shape: " + m.getOriginalMessage();
        }
        return "the request body is not readable JSON: " + rootMessage(e);
    }

    private static String path(MismatchedInputException e) {
        StringBuilder sb = new StringBuilder();
        for (com.fasterxml.jackson.databind.JsonMappingException.Reference r : e.getPath()) {
            if (r.getFieldName() != null) {
                if (!sb.isEmpty()) {
                    sb.append('.');
                }
                sb.append(r.getFieldName());
            } else if (r.getIndex() >= 0) {
                sb.append('[').append(r.getIndex()).append(']');
            }
        }
        return sb.toString();
    }

    private static String quote(Object value) {
        return value == null ? "null" : "\"" + value + "\"";
    }

    private static String rootMessage(Throwable t) {
        Throwable root = t;
        while (root.getCause() != null && root.getCause() != root) {
            root = root.getCause();
        }
        return message(root);
    }

    /** An exception's message, or its type when it carries none, so a body is never empty. */
    private static String message(Throwable t) {
        return t.getMessage() == null ? t.getClass().getName() : t.getMessage();
    }
}
