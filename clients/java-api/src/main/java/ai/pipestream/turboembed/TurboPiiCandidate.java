package ai.pipestream.turboembed;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * Immutable value copy of one PII candidate row from {@code turbo:pii-candidates}, emitted when
 * {@link TurboAnalyzer.Config#piiDetail()} is enabled.
 *
 * <p>{@link #instance()} is the {@code turbo:pii-candidates} layer instance name;
 * {@link #piiInstance()} is the bound parent {@code turbo:pii} layer instance name that mention
 * indices in {@link #disposition()} refer to. Candidates never carry matched text or a normalized
 * value.
 */
public record TurboPiiCandidate(
        String instance,
        String piiInstance,
        long itemIndex,
        long fragmentIndex,
        long viewId,
        TurboSpanData.CoordinateRange viewRange,
        TurboSpanData.CoordinateRange sourceRange,
        long piiKindCode,
        String label,
        TurboPiiDetector detector,
        List<TurboPiiEvidence> evidence,
        Optional<TurboPiiScore> score,
        TurboPiiDisposition disposition) {

    public TurboPiiCandidate {
        Objects.requireNonNull(instance, "instance");
        Objects.requireNonNull(piiInstance, "piiInstance");
        Objects.requireNonNull(viewRange, "viewRange");
        Objects.requireNonNull(sourceRange, "sourceRange");
        Objects.requireNonNull(label, "label");
        Objects.requireNonNull(detector, "detector");
        evidence = List.copyOf(Objects.requireNonNull(evidence, "evidence"));
        Objects.requireNonNull(score, "score");
        Objects.requireNonNull(disposition, "disposition");
        if (itemIndex < 0 || fragmentIndex < 0 || viewId < 0 || piiKindCode <= 0) {
            throw new IllegalArgumentException("Invalid PII candidate indices or kind code");
        }
        TurboPiiKind.fromCode(piiKindCode);
    }

    /**
     * Creates a candidate whose bound parent {@code turbo:pii} instance has the same name as its
     * {@code turbo:pii-candidates} layer instance.
     */
    public TurboPiiCandidate(
            String instance,
            long itemIndex,
            long fragmentIndex,
            long viewId,
            TurboSpanData.CoordinateRange viewRange,
            TurboSpanData.CoordinateRange sourceRange,
            long piiKindCode,
            String label,
            TurboPiiDetector detector,
            List<TurboPiiEvidence> evidence,
            Optional<TurboPiiScore> score,
            TurboPiiDisposition disposition) {
        this(
                instance,
                instance,
                itemIndex,
                fragmentIndex,
                viewId,
                viewRange,
                sourceRange,
                piiKindCode,
                label,
                detector,
                evidence,
                score,
                disposition);
    }

    /** Returns the built-in PII kind when {@link #piiKindCode()} names a built-in kind. */
    public Optional<TurboPiiKind> piiKind() {
        return TurboPiiKind.fromCode(piiKindCode);
    }

    /**
     * Returns the index into the layer's custom-kind table when {@link #piiKindCode()} names a
     * custom kind.
     */
    public OptionalInt piiCustomKindIndex() {
        return TurboPiiKind.customIndex(piiKindCode);
    }
}
