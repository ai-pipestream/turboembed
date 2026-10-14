# turbo-train-recipes

Recipes that train models only on permissive data, and the
accuracy gate each model passes before it ships (#401). docs/analysis.md
("Models trained on permissive data" and "Model gates") describes both.

| Module | What it holds |
|---|---|
| `masc` | MASC 3.0.0 (CC BY 3.0 US): the release record, its zoo dataset entry and fetch, the digest of an extracted copy, the fixed split |
| `segmentation` | The English sentence detector and tokenizer recipes: training on the split's training documents, the held-out samples, scoring a model with OpenNLP's evaluators, the gate against the OpenNLP English UD model |
| `postag` | The English part-of-speech tagger recipe, the same way: training on MASC's Penn tags, the held-out samples, scoring with `POSEvaluator`, the gate against OpenNLP's English Penn tagger (`en-pos-maxent.bin`, OpenNLP 1.5) |
| `card` | The model card, the model's directory (model, `card.json`, `gate.json`) and its permissive zoo entry, made only for a card whose gate against its recipe's reference (by id and SHA-256) passed on the recorded release |

```
cargo run --release -p turbo-train-recipes -- masc --corpus <masc-dir> \
    --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
cargo run --release -p turbo-train-recipes -- fetch-masc --home <zoo-home>
```

`--references` is a directory holding the reference models under their
published file names: `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin` and
`opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin`, as `cargo run -p
turbo-goldens-tool -- fetch-models <dir>` writes them and checked against
the zoo's SHA-256, and `en-pos-maxent.bin`, downloaded by hand from
<https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin> and checked
against the size and SHA-256 the recipe records (5,696,197 bytes,
`645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d`).
`--recipe` runs only the recipes named by the trained model's id; without
it every recipe runs, and a missing reference is an error. The exit
status is 1 when a gate fails and 2 on an error.

## The recipes

| Model | Reference | Data | Measure |
|---|---|---|---|
| `turbo-en-masc-3.0.0-sentence-1` | `opennlp-en-ud-ewt-sentence-1.3-2.5.4` | MASC sentence samples of 10 sentences | F-measure of sentence spans |
| `turbo-en-masc-3.0.0-tokens-1` | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | MASC Penn tokens, one sample per sentence | F-measure of token spans |
| `turbo-en-masc-3.0.0-pos-1` | `opennlp-en-pos-maxent-1.5` (`en-pos-maxent.bin`) | MASC Penn tokens and their `msd` tags, one sample per sentence | Word accuracy, one unit per sentence |

- Split: the documents in the reader's order (each directory's headers
  by name); positions 9, 19, 29 and so on are held out, one in ten. No
  randomness, so no seed.
- Parameters: those of the reference models (`ud::training_parameters`:
  GIS, 300 iterations, cutoff 5, 4 threads), the factory defaults for
  English (for the tagger: the default feature generator descriptor, no
  tag dictionary, beam size 3), OpenNLP 2.5.4's whitespace, and a fixed
  manifest `Timestamp` (2026-10-14T00:00:00Z), so a run gives the same
  model bytes.
- Samples: what OpenNLP's MASC streams read, except that the token
  stream skips a sentence without a sample (no tokens, or an empty
  token) alone, where OpenNLP's stops reading the corpus, and the
  part-of-speech stream skips a sentence with a token without a tag
  alone, where OpenNLP's throws and ends the training run; a sentence of
  no tokens gives the tagger no event and no held-out unit, and is
  skipped too. Held-out sentences are skipped the same way. The card
  records how many training sentences (`training.skipped_sentences`) and
  how many held-out sentences (`held_out.skipped_sentences`) were
  skipped, and a recipe fails when its reader skips more than one in a
  thousand of either (`MAX_SKIPPED_SENTENCES`), so a held-out document
  whose tags are missing cannot drop out of the gate.
- The tagger's gate: MASC's tags are Penn tags, so the reference is
  OpenNLP's English maxent tagger of the 1.5 models, `en-pos-maxent.bin`.
  OpenNLP's only English taggers trained on Penn tags are that one and
  the 1.5 perceptron tagger; the maxent one is trained with the
  algorithm the recipe uses. Both models are scored in the reference's
  tag set: the reference asked for Penn tags (`PosTagFormat::Penn`), its
  own outcomes, which come out unconverted, and the trained model with its
  outcomes as they are, MASC's tags. The card records both formats.
  MASC 3.0.0 writes some tags another way than the reference: brackets
  `(` and `)` where the reference has `-LRB-` and `-RRB-`, and tags of its
  own for hyphens and dashes (`-`), ellipses (`...`) and semicolons
  (`;`), tokens the reference tags `:`. The held-out tags and both
  models' tags are compared in the reference's spelling
  (`postag::REFERENCE_SPELLINGS`). A held-out token whose tag the
  reference cannot give even so would count against it whatever it did,
  so it is left out of both models' scores; the card lists those tokens
  by tag (`held_out.tokens.left_out`), and the recipe fails when they are
  more than one in a thousand of the held-out tokens
  (`MAX_TOKENS_LEFT_OUT`). MASC 3.0.0's held-out documents have 6 of
  85,292: ``UH` `` (3), `"` (2) and `PRN` (1). OpenNLP describes
  the reference's training data only as "(private) English data" (the
  1.5.1 test plan) and the zoo does not list it: the recipe records its
  address, size and SHA-256 (`postag::EN_POS_MAXENT`), a run reads it
  from `--references` and checks it, and it is only scored. It is never
  shipped or bundled, and the trained model is trained on MASC alone.
  The English UD tagger is not the reference: asked for Penn tags,
  `POSTagFormatMapper` turns each of its UD outcomes into one Penn tag
  (`VERB` into `VB`, `NOUN` into `NN`, `PUNCT` into `.`), so it can never
  give `VBD`, `NNS`, `,` or `TO`, and a gate against it would pass any
  tagger.
- Licence: the models are Apache-2.0, trained on CC BY 3.0 US data, so
  their zoo class is permissive; the card carries the attribution CC BY
  asks for, which ships with the model: MASC 3.0.0, by the American
  National Corpus project (anc.org), its address and its licence.
  Whether the ANC asks for a particular citation is to be confirmed on
  their site, which could not be reached when the recipes were written.

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

- Scores on MASC: none are recorded until the release is recorded and
  the recipes are run on it. No recipe has been run on MASC itself.

## Tests

`tests/recipes.rs` writes corpora in MASC's layout (`tests/common`, no
MASC text, Penn tags written for the tests) and runs everything on
them: the split in the reader's order; the three recipes trained,
scored and gated; the same bytes from a second training; the card, the
model's directory and its zoo entry; and the binary's refusal of an
unrecorded corpus and its output. It also reads the MASC corpora
checked in for the goldens (`analysis/goldens/data/inputs/formats/masc-*.txt`)
with the tagger recipe's reader and checks that it reads the samples
OpenNLP's `MascPOSSampleStream` read from them in the JVM (the
`masc/pos` goldens), and skips the sentences that stream threw on. The
test corpora are made of eight sentences, so a model trained on one has
seen every held-out sentence and its score there measures nothing. The gate
is checked against a model trained on plainer text: as the reference it
loses and the trained model passes, and as the candidate it loses far
more than the margin and fails, which is the check that tells the gate
from one that passes everything. A model against itself passes with no
difference.

The OpenNLP UD references need `TURBO_GOLDENS_MODELS` naming a
directory that holds them (`cargo run -p turbo-goldens-tool --
fetch-models <dir>`). With it, they are checked against the zoo's
digests and scored on the test corpus, the UD tagger asked for Penn tags
is checked to give only its 17 converted tags, and the binary runs end to
end on the recipes whose references are present (`--recipe`); without it
those parts skip, unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which
makes them fail. CI runs the test with both set, in the step that
requires the published models.

The tagger's reference, `en-pos-maxent.bin`, is local-only, as in
turbo-sequence's tests of the 1.5 models: its training data is not
documented, so CI does not fetch it. The parts that need it run when
`TURBO_TEST_OPENNLP_MODELS` names a directory holding it (a missing file
there fails), and skip without it even with
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. With it, it is checked against the
recorded digest, shown to return its 45 Penn outcomes unconverted, scored
and gated, and the binary runs all three recipes.

On the test corpus, with both directories set, the published references
score 63.16 (sentences, F-measure), 94.34 (tokens, F-measure) and 92.41
(`en-pos-maxent.bin`, accuracy), against 100 for the trained models,
which have seen every held-out sentence; these numbers show the gate is
wired through, not how the recipes do on MASC.
