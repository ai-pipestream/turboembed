package ai.pipestream.turboembed;

import java.util.Objects;

/**
 * Identity of a PII detector or scoring model: its stable {@code id} (for example
 * {@code "turbo-pii:cursor"} or {@code "acme:emp-id"}) and {@code version}.
 */
public record TurboPiiDetector(String id, String version) {
    public TurboPiiDetector {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(version, "version");
    }
}
