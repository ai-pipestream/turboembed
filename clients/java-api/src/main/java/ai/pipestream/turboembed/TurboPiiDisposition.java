package ai.pipestream.turboembed;

import java.util.Objects;

/**
 * How the PII annotator disposed of one candidate span in {@code turbo:pii-candidates}.
 */
public sealed interface TurboPiiDisposition
        permits TurboPiiDisposition.Kept,
                TurboPiiDisposition.AllowedByRule,
                TurboPiiDisposition.NestedIn,
                TurboPiiDisposition.RejectedByRule,
                TurboPiiDisposition.Repeats {

    /** Returns the disposition's tag. */
    Kind kind();

    /** Native {@code TURBO_PII_DISPOSITION_*} tags. */
    enum Kind {
        KEPT(1),
        ALLOWED_BY_RULE(2),
        NESTED_IN(3),
        REJECTED_BY_RULE(4),
        REPEATS(5);

        private final int nativeCode;

        Kind(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        public int nativeCode() {
            return nativeCode;
        }

        public static Kind fromNative(long nativeCode) {
            for (Kind kind : values()) {
                if (kind.nativeCode == nativeCode) {
                    return kind;
                }
            }
            throw new IllegalStateException("Unknown native PII disposition: " + nativeCode);
        }
    }

    /** Native {@code TURBO_PII_ALLOW_SCOPE_*} codes. */
    enum AllowScope {
        GLOBAL(1),
        KIND(2);

        private final int nativeCode;

        AllowScope(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        public int nativeCode() {
            return nativeCode;
        }

        public static AllowScope fromNative(long nativeCode) {
            for (AllowScope scope : values()) {
                if (scope.nativeCode == nativeCode) {
                    return scope;
                }
            }
            throw new IllegalStateException("Unknown native PII allow scope: " + nativeCode);
        }
    }

    /** Native {@code TURBO_PII_ALLOW_RULE_*} codes. */
    enum AllowRuleKind {
        TERM(1),
        PATTERN(2);

        private final int nativeCode;

        AllowRuleKind(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        public int nativeCode() {
            return nativeCode;
        }

        public static AllowRuleKind fromNative(long nativeCode) {
            for (AllowRuleKind ruleKind : values()) {
                if (ruleKind.nativeCode == nativeCode) {
                    return ruleKind;
                }
            }
            throw new IllegalStateException("Unknown native PII allow rule kind: " + nativeCode);
        }
    }

    /** Native {@code TURBO_PII_REJECT_*} codes. */
    enum RejectReason {
        LABEL_REQUIRED(1),
        THRESHOLD(2);

        private final int nativeCode;

        RejectReason(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        public int nativeCode() {
            return nativeCode;
        }

        public static RejectReason fromNative(long nativeCode) {
            for (RejectReason reason : values()) {
                if (reason.nativeCode == nativeCode) {
                    return reason;
                }
            }
            throw new IllegalStateException("Unknown native PII reject reason: " + nativeCode);
        }
    }

    /** Kept as item {@code mentionIndex} of the {@code turbo:pii} layer. */
    record Kept(long mentionIndex) implements TurboPiiDisposition {
        public Kept {
            if (mentionIndex < 0 || mentionIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("mentionIndex must be an unsigned 32-bit integer: " + mentionIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.KEPT;
        }
    }

    /** Suppressed by an allow rule at {@code scope} and {@code ruleIndex}. */
    record AllowedByRule(AllowScope scope, AllowRuleKind ruleKind, long ruleIndex) implements TurboPiiDisposition {
        public AllowedByRule {
            Objects.requireNonNull(scope, "scope");
            Objects.requireNonNull(ruleKind, "ruleKind");
            if (ruleIndex < 0 || ruleIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("ruleIndex must be an unsigned 32-bit integer: " + ruleIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.ALLOWED_BY_RULE;
        }
    }

    /** Suppressed because it overlapped kept mention {@code mentionIndex}. */
    record NestedIn(long mentionIndex) implements TurboPiiDisposition {
        public NestedIn {
            if (mentionIndex < 0 || mentionIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("mentionIndex must be an unsigned 32-bit integer: " + mentionIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.NESTED_IN;
        }
    }

    /** Suppressed by a detector rule ({@code reason}). */
    record RejectedByRule(RejectReason reason) implements TurboPiiDisposition {
        public RejectedByRule {
            Objects.requireNonNull(reason, "reason");
        }

        @Override
        public Kind kind() {
            return Kind.REJECTED_BY_RULE;
        }
    }

    /** Coalesced into kept mention {@code mentionIndex}. */
    record Repeats(long mentionIndex) implements TurboPiiDisposition {
        public Repeats {
            if (mentionIndex < 0 || mentionIndex > 0xFFFF_FFFFL) {
                throw new IllegalArgumentException("mentionIndex must be an unsigned 32-bit integer: " + mentionIndex);
            }
        }

        @Override
        public Kind kind() {
            return Kind.REPEATS;
        }
    }

    /** Decodes one candidate disposition from its four unsigned 32-bit fields. */
    static TurboPiiDisposition fromNative(long disposition, long a, long b, long c) {
        return switch (Kind.fromNative(disposition)) {
            case KEPT -> new Kept(a);
            case ALLOWED_BY_RULE -> new AllowedByRule(
                    AllowScope.fromNative(a), AllowRuleKind.fromNative(b), c);
            case NESTED_IN -> new NestedIn(a);
            case REJECTED_BY_RULE -> new RejectedByRule(RejectReason.fromNative(a));
            case REPEATS -> new Repeats(a);
        };
    }
}
