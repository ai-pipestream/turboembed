package ai.pipestream.turboembed;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

/** A copied span row never prints its value, and maps a PII row's numeric kind. */
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

    private static TurboSpanData pii(TurboLayerKind kind, String label, OptionalLong code) {
        return new TurboSpanData(
                kind,
                0,
                0,
                0,
                1,
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                "turbo:pii",
                "",
                Optional.of(label),
                Optional.empty(),
                OptionalDouble.empty(),
                code);
    }

    @Test
    void aPiiKindCodeNamesABuiltInKindOrACustomTableIndex() {
        TurboSpanData email = pii(TurboLayerKind.PII, "email", OptionalLong.of(5));
        assertEquals(Optional.of(TurboPiiKind.EMAIL), email.piiKind());
        assertEquals(OptionalInt.empty(), email.piiCustomKindIndex());
        assertEquals(1, email.classification(), "the tier stays in classification");

        TurboSpanData custom = pii(TurboLayerKind.PII, "employee-id", OptionalLong.of(TurboPiiKind.CUSTOM_CODE_BASE + 3));
        assertEquals(Optional.empty(), custom.piiKind());
        assertEquals(OptionalInt.of(3), custom.piiCustomKindIndex());
        for (TurboPiiKind kind : TurboPiiKind.values()) {
            assertTrue(kind.nativeCode() < TurboPiiKind.CUSTOM_CODE_BASE, kind + " collides with a custom code");
            assertEquals(Optional.of(kind), TurboPiiKind.fromCode(kind.nativeCode()));
        }

        assertThrows(IllegalStateException.class, () -> pii(TurboLayerKind.PII, "x", OptionalLong.of(0)));
        assertThrows(IllegalStateException.class, () -> pii(TurboLayerKind.PII, "x", OptionalLong.of(31)));
        assertThrows(
                IllegalStateException.class,
                () -> pii(TurboLayerKind.PII, "x", OptionalLong.of(TurboPiiKind.CUSTOM_CODE_BASE + 0x10000)));
        assertThrows(IllegalArgumentException.class, () -> pii(TurboLayerKind.ENTITIES, "x", OptionalLong.of(5)));
        TurboSpanData legacy = new TurboSpanData(
                TurboLayerKind.PII, 0, 0, 0, 1, Optional.empty(), Optional.empty(), Optional.empty(), "turbo:pii", "",
                Optional.of("email"), Optional.empty(), OptionalDouble.empty());
        assertEquals(OptionalLong.empty(), legacy.piiKindCode(), "the earlier constructor carries no code");
        assertEquals(Optional.empty(), legacy.piiKind());
    }
}
