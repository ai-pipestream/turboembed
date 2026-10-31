# Native sentence-vector snapshot

`sentence_vectors_native.txt` contains eight 32-dimensional CPU Exact vectors
from `RuleSentenceDetector("en")`, `SubwordAnnotator`, and `SentenceVectors` on
the project-authored texts in `tests/sentence_vectors_native.rs`. Empty and
whitespace cases produce no rows. Each line starts with the case and sentence
indices, followed by the 32 float components.

The snapshot was captured from the native implementation based on main
`9d0c5844e`, using the generated two-layer tiny-BERT fixture and two CPU threads.
No external model, JVM runtime, or external golden output was used. This is an
integration regression snapshot, not trained-model accuracy or peer parity.

Fixture SHA-256 values:

- `manifest.json`: `273ee3dfe946989a27a68e37d58e1dd7bc537c06b404075279f02a1b640b91f7`
- `weights/model.safetensors`: `c258bffc3c37afc5a8b12324c0d29b81d31738fbf72183ade9c52c0d68e06180`

The mandatory test checks exact source-byte spans, every vector component with
absolute tolerance 2e-6, finite values, unit norms and exact row counts. It
covers abbreviation suppression, accented text, ideographs, CRLF, empty text
and whitespace. Snapshot changes require reviewing the native rule or encoder
change and recording new native execution evidence; never replace these rows
with external model output.
