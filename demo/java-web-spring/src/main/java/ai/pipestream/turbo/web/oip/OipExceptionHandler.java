// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.oip;

import ai.pipestream.turbo.TurboException;
import ai.pipestream.turbo.web.TurboService;
import ai.pipestream.turbo.web.TurboStatus;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Refusals in the protocol's own shape: a body of {@code {"error": "..."}}
 * with 400 for a request the model cannot take, 404 for a name or version this
 * server does not serve, and 500 for a server or device failure, as the Open
 * Inference Protocol version 2 HTTP/REST section specifies.
 *
 * <p>The message is the library's own, so a refusal that names a
 * {@code TURBO_E_*} code and a field index still names them here even though
 * the protocol has no field for either. The {@code /api/v1} surface returns
 * them as separate fields.
 */
@RestControllerAdvice(assignableTypes = OipController.class)
public class OipExceptionHandler {
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(OipExceptionHandler.class);

    /** libturbo refused: a caller-fixable refusal is 400, anything else is 500. */
    @ExceptionHandler(TurboException.class)
    public ResponseEntity<Oip.Error> turbo(TurboException e) {
        HttpStatus mapped = TurboStatus.of(e);
        boolean caller = mapped.is4xxClientError() || mapped == HttpStatus.NOT_IMPLEMENTED;
        if (!caller) {
            LOG.error("{} during inference", e.statusName(), e);
        }
        return ResponseEntity.status(caller ? HttpStatus.BAD_REQUEST : HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new Oip.Error(e.getMessage()));
    }

    /** A model name or version this server does not serve. */
    @ExceptionHandler(TurboService.ModelNotFound.class)
    public ResponseEntity<Oip.Error> notFound(TurboService.ModelNotFound e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new Oip.Error(e.getMessage()));
    }

    /** A request this server checked before any native call. */
    @ExceptionHandler({IllegalArgumentException.class, HttpMessageNotReadableException.class})
    public ResponseEntity<Oip.Error> bad(Exception e) {
        return ResponseEntity.badRequest().body(new Oip.Error(message(e)));
    }

    /** Every session or generation slot of the model is in use. */
    @ExceptionHandler(TurboService.Overloaded.class)
    public ResponseEntity<Oip.Error> overloaded(TurboService.Overloaded e) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(new Oip.Error(e.getMessage()));
    }

    /** The server got into a state it cannot serve from; the log carries the stack. */
    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Oip.Error> state(IllegalStateException e) {
        LOG.error("inference failed", e);
        return ResponseEntity.internalServerError().body(new Oip.Error(message(e)));
    }

    /** An exception's message, or its type when it carries none, so the error string is never empty. */
    private static String message(Throwable t) {
        return t.getMessage() == null ? t.getClass().getName() : t.getMessage();
    }
}
