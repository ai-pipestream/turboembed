package ai.pipestream.turboembed;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
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
        // One above the largest built-in id: ids are append-only, so this tracks the registry.
        long unknown = TurboPiiKind.values()[TurboPiiKind.values().length - 1].nativeCode() + 1L;
        assertThrows(IllegalStateException.class, () -> pii(TurboLayerKind.PII, "x", OptionalLong.of(unknown)));
        assertThrows(
                IllegalStateException.class,
                () -> pii(TurboLayerKind.PII, "x", OptionalLong.of(TurboPiiKind.CUSTOM_CODE_BASE + 0x10000)));
        assertThrows(IllegalArgumentException.class, () -> pii(TurboLayerKind.ENTITIES, "x", OptionalLong.of(5)));
        TurboSpanData legacy = new TurboSpanData(
                TurboLayerKind.PII, 0, 0, 0, 1, Optional.empty(), Optional.empty(), Optional.empty(), "turbo:pii", "",
                Optional.of("email"), Optional.empty(), OptionalDouble.empty());
        assertEquals(OptionalLong.empty(), legacy.piiKindCode(), "the earlier constructor carries no code");
        assertEquals(Optional.empty(), legacy.piiKind());
        assertEquals(Optional.empty(), legacy.piiDetector());
        assertEquals(List.of(), legacy.piiEvidence());
        assertEquals(Optional.empty(), legacy.piiScore());
    }

    @Test
    void piiEvidenceDetectorScoreAndCandidateDecodeNativeCodesAndKeepValueWithheld() {
        String planted = "4532-0151-1283-0366";
        TurboPiiDetector detector = new TurboPiiDetector("acme:emp-id", "2.1.0");
        List<TurboPiiEvidence> evidence = List.of(
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.CHECKSUM.nativeCode(), 1, 1, 0),
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.LABEL.nativeCode(), 0, 3, 3),
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.RULE.nativeCode(), 7, 0, 0),
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.DENY_HIT.nativeCode(), 4, 9, 0),
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.SHAPE.nativeCode(), 0, 0, 0));
        assertEquals(new TurboPiiEvidence.Checksum(TurboPiiEvidence.ChecksumAlgorithm.LUHN, true), evidence.get(0));
        assertEquals(new TurboPiiEvidence.Label(0, 3, 3), evidence.get(1));
        assertEquals(new TurboPiiEvidence.Rule(7), evidence.get(2));
        assertEquals(new TurboPiiEvidence.DenyHit(4, 9), evidence.get(3));
        assertEquals(new TurboPiiEvidence.Shape(), evidence.get(4));

        TurboPiiScore score = new TurboPiiScore(
                0.9375f, TurboPiiScore.Kind.PROBABILITY, new TurboPiiDetector("acme:scorer", "1.0"), true);
        TurboSpanData row = new TurboSpanData(
                TurboLayerKind.PII,
                0,
                4,
                0,
                4,
                Optional.of(new TurboSpanData.CoordinateRange(4, 23, 4, 23, 4, 23)),
                Optional.empty(),
                Optional.empty(),
                "turbo:pii",
                "default",
                Optional.of("card"),
                Optional.of(planted),
                OptionalDouble.empty(),
                OptionalLong.of(TurboPiiKind.CARD.nativeCode()),
                Optional.of(detector),
                evidence,
                Optional.of(score));
        assertFalse(row.toString().contains(planted));
        assertTrue(row.toString().contains("value=Optional[<withheld>]"));
        assertEquals(Optional.of(detector), row.piiDetector());
        assertEquals(evidence, row.piiEvidence());
        assertEquals(Optional.of(score), row.piiScore());

        TurboPiiDisposition kept = TurboPiiDisposition.fromNative(TurboPiiDisposition.Kind.KEPT.nativeCode(), 0, 0, 0);
        assertEquals(new TurboPiiDisposition.Kept(0), kept);
        TurboPiiDisposition rejected = TurboPiiDisposition.fromNative(
                TurboPiiDisposition.Kind.REJECTED_BY_RULE.nativeCode(),
                TurboPiiDisposition.RejectReason.LABEL_REQUIRED.nativeCode(),
                0,
                0);
        assertEquals(new TurboPiiDisposition.RejectedByRule(TurboPiiDisposition.RejectReason.LABEL_REQUIRED), rejected);
        TurboPiiDisposition allowed = TurboPiiDisposition.fromNative(
                TurboPiiDisposition.Kind.ALLOWED_BY_RULE.nativeCode(),
                TurboPiiDisposition.AllowScope.KIND.nativeCode(),
                TurboPiiDisposition.AllowRuleKind.PATTERN.nativeCode(),
                3);
        assertEquals(
                new TurboPiiDisposition.AllowedByRule(
                        TurboPiiDisposition.AllowScope.KIND, TurboPiiDisposition.AllowRuleKind.PATTERN, 3),
                allowed);

        TurboSpanData.CoordinateRange range = new TurboSpanData.CoordinateRange(4, 23, 4, 23, 4, 23);
        TurboPiiCandidate candidate = new TurboPiiCandidate(
                "default",
                0,
                0,
                0,
                range,
                range,
                TurboPiiKind.CARD.nativeCode(),
                "card",
                detector,
                evidence.subList(0, 2),
                Optional.of(score),
                kept);
        assertEquals("default", candidate.instance());
        assertEquals("default", candidate.piiInstance());
        assertEquals(Optional.of(TurboPiiKind.CARD), candidate.piiKind());
        assertEquals(OptionalInt.empty(), candidate.piiCustomKindIndex());

        TurboPiiCandidate crossInstance = new TurboPiiCandidate(
                "audit",
                "actual",
                0,
                0,
                0,
                range,
                range,
                TurboPiiKind.EMAIL.nativeCode(),
                "email",
                detector,
                List.of(TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.RULE.nativeCode(), 0xFFFF_FFFFL, 0, 0)),
                Optional.empty(),
                new TurboPiiDisposition.Kept(1));
        assertEquals("audit", crossInstance.instance());
        assertEquals("actual", crossInstance.piiInstance());
        assertEquals(List.of(new TurboPiiEvidence.Rule(0xFFFF_FFFFL)), crossInstance.evidence());

        assertEquals(
                new TurboPiiEvidence.DenyHit(0xFFFF_FFFFL, 1L),
                TurboPiiEvidence.fromNative(TurboPiiEvidence.Kind.DENY_HIT.nativeCode(), 0xFFFF_FFFFL, 1L, 0L));
        assertThrows(IllegalArgumentException.class, () -> new TurboPiiEvidence.Rule(-1L));
        assertThrows(IllegalArgumentException.class, () -> new TurboPiiEvidence.Rule(0x1_0000_0000L));
        assertThrows(IllegalArgumentException.class, () -> new TurboPiiEvidence.DenyHit(-1L, 0L));
        assertThrows(IllegalArgumentException.class, () -> new TurboPiiEvidence.DenyHit(0x1_0000_0000L, 0L));
        assertThrows(IllegalArgumentException.class, () -> new TurboPiiEvidence.DenyHit(0L, 0x1_0000_0000L));
    }
}
