package ai.pipestream.turboembed;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Optional;
import java.util.OptionalDouble;
import org.junit.jupiter.api.Test;

/** A copied span row never prints its value. */
final class TurboSpanDataTest {
    @Test
    void toStringWithholdsAPiiValueAndKeepsTheOtherFields() {
        String planted = "planted-alice@example.com";
        TurboSpanData row = new TurboSpanData(
                TurboLayerKind.PII,
                0,
                0,
                0,
                1,
                Optional.of(new TurboSpanData.CoordinateRange(3, 28, 3, 28, 3, 28)),
                Optional.empty(),
                Optional.empty(),
                "turbo:pii",
                "default",
                Optional.of("email"),
                Optional.of(planted),
                OptionalDouble.empty());
        String shown = row.toString();
        assertFalse(shown.contains(planted), shown);
        assertTrue(shown.contains("value=Optional[<withheld>]"), shown);
        assertTrue(shown.contains("label=Optional[email]"), shown);
        assertEquals(Optional.of(planted), row.value());
    }
}
