# Native sentence evaluation

Sentence recipes use `Reference::GoldSentences`. It identifies the versioned
newline baseline, not a downloadable model. Asking it for a file or supplying
model bytes fails. The old sentence `SegmentationRecipe::gate` byte interface
also fails before reading either model. Token and other legacy recipes are
being retired separately.

Use `sentence_gold::gate` with a candidate `ModelId`, a `HeldOut::Sentences`
collection, one native predicted span vector per sample, the corpus/split name,
and the verified corpus SHA-256. MASC's existing `held_out` reader and the UD
recipe's `held_out` reader retain their sample construction and split rules.
No MASC pins or Penn reader were changed.

Generate predictions from the rule detector, SaT, or the native sentence head.
Use the exact sample text given to the gate, and preserve UTF-8 byte offsets.
The gate rejects missing predictions, wrong components, empty sets, invalid
UTF-8 bounds, overlapping or unordered spans, and reversed source ranges.
Both the candidate and the deterministic one-sentence-per-nonblank-line
baseline are scored against the same gold labels. The record includes each
sample's true-positive, predicted and gold counts, F1, the existing paired
bootstrap confidence interval and gate decision. Its sample digest covers
scored text, gold spans and source coordinates. The baseline digest identifies
its algorithm version; it is not a fabricated model-file digest.

At least 30 samples are required by the existing model gate. The caller must
supply verified corpus-file identity, release and split information; the gate
cannot establish provenance from a string supplied as a digest. Predictions
from spaCy or Stanza peers must be separately identified and scored on the same
gold samples, not treated as gold labels or used as a teacher.

## Completion evidence still required for #1644

The project-authored tests validate gate arithmetic and input rejection. They
are not corpus accuracy measurements. Recorded sentence F1 for rules and SaT
on EWT test, AnCora test and held-out MASC, plus matching spaCy and Stanza peer
records, remain outstanding. The native sentence head must be added when
available. Do not close #1644 based on these unit tests or the tiny encoder
snapshot alone.
