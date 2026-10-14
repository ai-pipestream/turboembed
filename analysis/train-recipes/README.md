# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401). docs/analysis.md
("Models trained on permissive data" and "Model gates") describes both.

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry |

```
cargo run --release -p turbo-train-recipes -- masc --corpus <masc-dir> \
    --references <dir> --out <dir> [--unrecorded-corpus]
cargo run --release -p turbo-train-recipes -- fetch-masc --home <zoo-home>
```

`--references` is a directory holding the OpenNLP models under the file
names the zoo gives them (`opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin`,
`opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin`), as `cargo run -p
turbo-goldens-tool -- fetch-models <dir>` writes them; each is checked
against the zoo's SHA-256. The exit status is 1 when a gate fails and 2
on an error.

## The recipes

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | `opennlp-en-ud-ewt-sentence-1.3-2.5.4` | MASC sentence samples of 10 sentences | F-measure of sentence spans |
| `turbo-en-masc-3.0.0-tokens-1` | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | MASC Penn tokens, one sample per sentence | F-measure of token spans |

- Split: the documents in the reader's order (each directory's headers
  by name); positions 9, 19, 29 and so on are held out, one in ten. No
  randomness, so no seed.
- Parameters: those of the reference models (`ud::training_parameters`:
  GIS, 300 iterations, cutoff 5, 4 threads), the factory defaults for
  English, OpenNLP 2.5.4's whitespace, and a fixed manifest `Timestamp`
  (2026-10-14T00:00:00Z), so a run gives the same model bytes.
- Samples: what OpenNLP's MASC streams read, except that the token
  stream skips a sentence without a sample (no tokens, or an empty
  token) alone; OpenNLP's stops reading the corpus there. The card
  records how many were skipped.
- Licence: the models are Apache-2.0, trained on CC BY 3.0 US data, so
  their zoo class is permissive; the card carries the attribution CC BY
  asks for, which ships with the model.

## The release record

MASC is fetched through the zoo by digest and read from its extracted
directory. It is not checked in: CC BY would allow it with attribution,
but the policy prefers fetching by digest. `masc::RELEASE` is `NotRecorded`:
the release could not be downloaded where the recipes were written
(`anc.org` was not reachable), so its archive's size and digests and the
extracted directory's digest are not known. Until they are recorded:

- `fetch-masc` and `masc::dataset` refuse (`Error::NotRecorded`);
- `masc` refuses a corpus directory unless given `--unrecorded-corpus`,
  prints the directory's digest, and writes cards with
  `release_recorded: false`, which are not a release of the models.

To record it: download the release, record the archive (file name, format,
URL, size, SHA-256, SHA-512) and the digest `masc --unrecorded-corpus`
prints in `masc::RELEASE`. The zoo has no tar format yet, so a release
published as a gzip tar needs one before it can be listed.

## Not here yet

- The part-of-speech tagger recipe: it needs the part-of-speech event
  stream and its trainer, which are not in the workspace yet. MASC's
  Penn tags and the reader's `MascPosReader` are ready for it, and the
  gate takes accuracy counts (`Counts::accuracy`). The reference,
  `opennlp-en-ud-ewt-pos-1.3-2.5.4`, tags with the UD tag set, so its
  scores against MASC's Penn tags need a mapping or a Penn reference.
- Scores on MASC: none are recorded until the release is recorded and
  the recipes are run on it.

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text) and runs everything on them: the split in the reader's
order; both recipes trained, scored and gated against a model trained on
plainer text, which passes, and with the places swapped, which fails; a
model against itself, which passes with no difference; the same bytes
from a second training; the card, the model's directory and its zoo
entry; and the binary's refusal of an unrecorded corpus and its output.
With `TURBO_GOLDENS_MODELS` naming a directory that holds the OpenNLP
models, the published references are checked against the zoo and
scored on the test corpus, and the binary runs end to end.
