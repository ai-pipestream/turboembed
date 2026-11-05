# Native sentence evaluation

Sentence recipes use `Reference::GoldSentences`. It identifies the versioned
newline baseline, not a downloadable model. Asking it for a file or supplying
model bytes fails. The old sentence `SegmentationRecipe::gate` byte interface
also fails before reading either model. This command does not load any reference-model bytes.

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
are not corpus accuracy measurements. Executed rule records for EWT test, Spanish AnCora test and held-out MASC
are in [records/sentences-1644](records/sentences-1644/README.md). SaT and matching
spaCy and Stanza peer records remain outstanding. The native sentence head must be added when
available. Do not close #1644 based on these unit tests or the tiny encoder
snapshot alone.

## Offline rule evaluation command

`turbo-sentence-eval` runs the native rules against local gold labels. It never
fetches a corpus, loads a model, trains a head, or launches another runtime.
Obtain any required corpus licence acceptance separately before supplying data.
For a local CoNLL-U test file, use:

```sh
cargo run --release --locked -p turbo-train-recipes --bin turbo-sentence-eval -- \
  ud test.conllu EXPECTED_SHA256 en 'UD-EWT-r2.18/test'
```

Replace `EXPECTED_SHA256` with the independently verified file digest and name
the actual release/test split. The command hashes the bytes it reads and refuses
a mismatch before parsing. It requires a supported lowercase language tag and
nonblank `# text` on every sentence; it never scores the general reader's
missing-text placeholder. It uses the existing CoNLL-U sentence reader's groups
of ten sentences and source coordinates, including its final partial group.
This is an evaluation grouping protocol, not original document reconstruction.

For the recorded MASC 3.0.0 directory:

```sh
cargo run --release --locked -p turbo-train-recipes --bin turbo-sentence-eval -- \
  masc MASC-3.0.0
```

MASC requires the existing pinned corpus digest with no unrecorded override. It
uses the existing held-out split (every tenth document), Twitter exclusion and
sentence reader. Unreadable/skipped documents fail rather than moving the split.

Stdout is the existing model-gate record: counts, hashes, candidate identity,
paired bootstrap and decision, with no corpus text or local input paths. The
candidate includes rule version, language, configuration hash and group size;
its SHA-256 is the running executable's digest, identifying the actual compiled
code and tables. The corpus digest and sample digest identify the input and
exact scored labels. No scratch output files are created. Capture stdout only
where your application's storage policy allows aggregate evaluation records.

Exit status 0 means the unchanged rule-versus-newline gate passed, 2 means it
failed (the record is still emitted), and 1 means invalid input or an execution
error (no partial record). The existing minimum of 30 sample groups applies.
The bundled CLI tests are project-authored fixtures, not real-corpus accuracy
evidence. SaT, our trained head and spaCy/Stanza peer records remain separate
outstanding work; this command alone does not complete #1644.
