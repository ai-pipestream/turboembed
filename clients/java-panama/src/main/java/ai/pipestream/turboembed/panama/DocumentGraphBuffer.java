package ai.pipestream.turboembed.panama;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Holds a serialized binary {@code DocumentGraph} ({@code "TGRF"} wire format) alongside its
 * decoded spans.
 */
public final class DocumentGraphBuffer {
    /**
     * Magic header bytes ({@code "TGRF"}) at the start of every valid binary {@code DocumentGraph}.
     */
    public static final String MAGIC = "TGRF";

    private final byte[] wireBytes;
    private final List<TurboSpan> spans;

    /**
     * Constructs a {@code DocumentGraphBuffer} from raw wire bytes and decoded spans.
     *
     * @param wireBytes binary {@code DocumentGraph} payload
     * @param spans     decoded sentence, token, entity, and PII spans
     */
    public DocumentGraphBuffer(byte[] wireBytes, List<TurboSpan> spans) {
        this.wireBytes = Objects.requireNonNull(wireBytes, "wireBytes").clone();
        this.spans = List.copyOf(Objects.requireNonNull(spans, "spans"));
    }

    /**
     * Returns a defensive copy of the {@code "TGRF"} binary wire payload.
     *
     * @return serialized {@code DocumentGraph} bytes
     */
    public byte[] wireBytes() {
        return wireBytes.clone();
    }

    /**
     * Returns the number of bytes in the serialized {@code DocumentGraph}.
     *
     * @return wire byte count
     */
    public int byteLength() {
        return wireBytes.length;
    }

    /**
     * Checks whether the buffer begins with the 4-byte {@code "TGRF"} magic header.
     *
     * @return {@code true} if the header matches {@code "TGRF"}
     */
    public boolean hasValidMagic() {
        if (wireBytes.length < 4) {
            return false;
        }
        String header = new String(Arrays.copyOfRange(wireBytes, 0, 4), StandardCharsets.US_ASCII);
        return MAGIC.equals(header);
    }

    /**
     * Returns the immutable list of spans extracted from the graph.
     *
     * @return decoded spans
     */
    public List<TurboSpan> spans() {
        return spans;
    }
}
