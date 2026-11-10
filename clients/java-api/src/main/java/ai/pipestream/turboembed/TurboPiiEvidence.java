package ai.pipestream.turboembed;

import java.util.Objects;

/**
 * One structured evidence fact supporting a PII mention or candidate.
 *
 * <p>Evidence facts carry small integer identifiers and byte offsets only, never matched text or
 * context-label text.
 */
public sealed interface TurboPiiEvidence
        permits TurboPiiEvidence.Shape,
                TurboPiiEvidence.Checksum,
                TurboPiiEvidence.Label,
                TurboPiiEvidence.DenyHit,
                TurboPiiEvidence.Rule {

    /** Returns the fact's kind tag. */
    Kind kind();

    /** Native {@code TURBO_PII_EVIDENCE_*} tags. */
    enum Kind {
        SHAPE(1),
        CHECKSUM(2),
        LABEL(3),
        DENY_HIT(4),
        RULE(5);

        private final int nativeCode;

        Kind(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native {@code TURBO_PII_EVIDENCE_*} code. */
        public int nativeCode() {
            return nativeCode;
        }

        public static Kind fromNative(long nativeCode) {
            for (Kind kind : values()) {
                if (kind.nativeCode == nativeCode) {
                    return kind;
                }
            }
            throw new IllegalStateException("Unknown native PII evidence kind: " + nativeCode);
        }
    }

    /** Native {@code TURBO_PII_CHECKSUM_*} algorithm ids. */
    enum ChecksumAlgorithm {
        LUHN(1, "luhn"),
        MOD97(2, "mod97"),
        MOD11(3, "mod11"),
        MOD23(4, "mod23"),
        ES_CIF(5, "es-cif"),
        ISO7064_MOD11_10(6, "iso7064-mod11-10"),
        ICAO9303(7, "icao9303"),
        BASE58CHECK(8, "base58check"),
        BECH32(9, "bech32"),
        EIP55(10, "eip55"),
        ABA_WEIGHTS(11, "aba-weights"),
        OTHER(12, "other"),
        LA_POSTE_MOD5(13, "la-poste-mod5"),
        VIN(14, "vin"),
        UPS_MOD10(15, "ups-mod10"),
        FEDEX_MOD11(16, "fedex-mod11"),
        GS1_MOD10(17, "gs1-mod10");

        private final int nativeCode;
        private final String wireName;

        ChecksumAlgorithm(int nativeCode, String wireName) {
            this.nativeCode = nativeCode;
            this.wireName = wireName;
        }

        /** Returns the native {@code TURBO_PII_CHECKSUM_*} id. */
        public int nativeCode() {
            return nativeCode;
        }

        /** Returns the algorithm's wire name. */
        public String wireName() {
            return wireName;
        }

        public static ChecksumAlgorithm fromNative(long nativeCode) {
            for (ChecksumAlgorithm algorithm : values()) {
                if (algorithm.nativeCode == nativeCode) {
                    return algorithm;
                }
            }
            throw new IllegalStateException("Unknown native PII checksum algorithm: " + nativeCode);
        }
    }

    /** The span has the character, length and prefix shape of its kind. */
    record Shape() implements TurboPiiEvidence {
        @Override
        public Kind kind() {
            return Kind.SHAPE;
        }
    }

    /** A check digit or check word over the span was evaluated. */
    record Checksum(ChecksumAlgorithm algorithm, boolean passed) implements TurboPiiEvidence {
        public Checksum {
            Objects.requireNonNull(algorithm, "algorithm");
        }

        @Override
        public Kind kind() {
            return Kind.CHECKSUM;
        }
    }

    /**
     * A context label for the kind stands in the window around the span, at UTF-8 byte range
     * {@code [byteStart, byteEnd)} of the layer's view, matching entry {@code phraseIndex} of the
     * kind's phrase list (built-in phrases first, then extra labels).
     */
    record Label(long byteStart, long byteEnd, long phraseIndex) implements TurboPiiEvidence {
        public Label {
            if (byteStart < 0
                    || byteEnd < byteStart
                    || byteEnd > 0xFFFF_FFFFL
                    || phraseIndex < 0
                    || phraseIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException(
                        "Invalid PII label evidence: [" + byteStart + ", " + byteEnd + ") phrase " + phraseIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.LABEL;
        }
    }

    /** The span matched term {@code termIndex} of deny list {@code listIndex}. */
    record DenyHit(long listIndex, long termIndex) implements TurboPiiEvidence {
        public DenyHit {
            if (listIndex < 0 || listIndex > 0xFFFF_FFFFL || termIndex < 0 || termIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("Invalid PII deny-hit evidence: " + listIndex + ":" + termIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.DENY_HIT;
        }
    }

    /** Extractor rule {@code ruleIndex} matched. */
    record Rule(long ruleIndex) implements TurboPiiEvidence {
        public Rule {
            if (ruleIndex < 0 || ruleIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("Invalid PII rule evidence: " + ruleIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.RULE;
        }
    }

    /** Decodes one {@code turbo_pii_fact} row from its four unsigned 32-bit fields. */
    static TurboPiiEvidence fromNative(long kind, long a, long b, long c) {
        return switch (Kind.fromNative(kind)) {
            case SHAPE -> new Shape();
            case CHECKSUM -> {
                if (b != 0L && b != 1L) {
                    throw new IllegalStateException("Invalid native checksum passed flag: " + b);
                }
                yield new Checksum(ChecksumAlgorithm.fromNative(a), b == 1L);
            }
            case LABEL -> new Label(a, b, c);
            case DENY_HIT -> new DenyHit(a, b);
            case RULE -> new Rule(a);
        };
    }
}
