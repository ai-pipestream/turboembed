# turbo-sat

SaT (`Segment any Text`) neural sentence segmentation (`SatAnnotator`, `SatOptions`), running quantized 1-layer and 3-layer XLM-RoBERTa token-level boundary classifiers over sliding windows without relying on punctuation heuristics.

## Features

- **Punctuation-agnostic neural boundary scoring (`analysis/sat/src/lib.rs`, `analysis/sat/src/head.rs`, `analysis/sat/src/cuts.rs`)**: Evaluates subword token representations and per-character/token sigmoid boundary probabilities to segment well-formed prose, unpunctuated ASR transcripts, OCR text, and multilingual code-switched documents.
- **Deterministic fallback and provider wiring (`analysis/sat/src/fallback.rs`, `analysis/sat/src/annotator/provider.rs`)**: Processes arbitrarily long documents in overlapping subword windows with deterministic UTF-8 character span projection.
- **Document pipeline integration (`analysis/sat/src/annotator.rs`)**: `SatAnnotator` implements `Annotator` to emit `Sentences` directly onto `Document` (for paragraph segmentation, see `turbo_sentdetect::ParagraphDetector`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/sat/tests/annotator.rs`, `analysis/sat/tests/device.rs`, `analysis/sat/tests/parity.rs`, `analysis/sat/tests/speed.rs`, and `analysis/sat/tests/unpunctuated_eval.rs`.
- **Accuracy and speed**: Evaluated against rule-based UAX #29 segmentation on clean and corrupted text; see `docs/segmentation.md` for F1 and throughput comparisons.

## Parity with wtpsplit at f16

`analysis/sat/tests/parity.rs` holds every character's boundary probability to wtpsplit's within a bound that follows the dtype the encoder computes in: 1e-3 at f32, whose derivation is at the constant in the test, and 5e-3 at f16 or bf16, which a device computes in at `TURBO_TEST_PRECISION=fastest`. The f16 bound is set from the largest difference measured over the 90 texts of each fixture, 2.49e-3, with a margin of about twice that.

The tiny bundle, largest difference at f16 (at f32 every device gives the CPU's numbers, 3.4e-4 at most):

| Fixture | Arc Pro B70, Level Zero | RTX 4080 SUPER, CUDA | Apple M2, Metal |
|---|---:|---:|---:|
| `t0.5-s16-b62` | 1.63e-3 | 1.92e-3 | 3.26e-4 |
| `t0.5-s7-b30` | 1.51e-3 | 1.74e-3 | 2.46e-4 |
| `t0.25-s62-b62` | 1.76e-3 | 2.37e-3 | 1.24e-4 |
| `la8-t0.5-s16-b62` | 1.76e-3 | 1.88e-3 | no band on this device |
| `la8-t0.025-s7-b30` | 1.76e-3 | 1.88e-3 | no band on this device |

The released checkpoints, largest difference over each one's three fixtures:

| Checkpoint | Device | f32 | f16 |
|---|---|---:|---:|
| `sat-1l-sm` | Level Zero | 4.22e-4 | 5.81e-4 |
| `sat-1l-sm` | CUDA | 4.22e-4 | 2.15e-3 |
| `sat-3l` | CUDA | 1.58e-4 | 2.49e-3 |

The cuts are held at both bounds. A character may fall on the other side of the threshold only where wtpsplit's probability is within the bound of it, and every text with no such character has its pieces compared with wtpsplit's. On the tiny bundle at f16 that leaves 88 or 89 of the 90 texts compared on every device, and on `sat-3l` on CUDA 87 or 88, where at most 8 of the 82 characters within the bound of the 0.025 threshold fell on the other side. Each test prints the bound it used, the dtype, the largest difference, the characters within the bound of the threshold and how many of them fell on the other side.

`sat-3l` at threshold 0.025, stride 64 and block 510 has one character on the other side of the threshold at f32 on every device; its fixture is named `path=1` in `TURBO_TEST_SAT_FIXTURES`.
