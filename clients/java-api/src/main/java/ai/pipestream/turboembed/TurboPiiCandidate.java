package ai.pipestream.turboembed;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * Immutable value copy of one PII candidate row from {@code turbo:pii-candidates}, emitted when
 * {@link TurboAnalyzer.Config#piiDetail()} is enabled.
 *
 * <p>Candidates never carry matched text or a normalized value.
 */
public record TurboPiiCandidate(
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

    public TurboPiiCandidate {
        Objects.requireNonNull(instance, "instance");
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
