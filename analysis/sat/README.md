# turbo-sat

SaT (`Segment any Text`) neural sentence segmentation (`SatAnnotator`, `SatOptions`), running quantized 1-layer and 3-layer XLM-RoBERTa token-level boundary classifiers over sliding windows without relying on punctuation heuristics.

## Features

- **Punctuation-agnostic neural boundary scoring (`analysis/sat/src/lib.rs`, `analysis/sat/src/head.rs`, `analysis/sat/src/cuts.rs`)**: Evaluates subword token representations and per-character/token sigmoid boundary probabilities to segment well-formed prose, unpunctuated ASR transcripts, OCR text, and multilingual code-switched documents.
- **Deterministic fallback and provider wiring (`analysis/sat/src/fallback.rs`, `analysis/sat/src/annotator/provider.rs`)**: Processes arbitrarily long documents in overlapping subword windows with deterministic UTF-8 character span projection.
- **Document pipeline integration (`analysis/sat/src/annotator.rs`)**: `SatAnnotator` implements `Annotator` to emit `Sentences` directly onto `Document` (for paragraph segmentation, see `turbo_sentdetect::ParagraphDetector`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/sat/tests/annotator.rs`, `analysis/sat/tests/device.rs`, `analysis/sat/tests/parity.rs`, `analysis/sat/tests/speed.rs`, and `analysis/sat/tests/unpunctuated_eval.rs`.
- **Accuracy and speed**: Evaluated against rule-based UAX #29 segmentation on clean and corrupted text; see `docs/segmentation.md` for F1 and throughput comparisons.
