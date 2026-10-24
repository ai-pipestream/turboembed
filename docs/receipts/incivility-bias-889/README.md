# Incivility precision, recall and identity-term bias records (#323, #889)

`all-minilm-l6-v2.json` and `bge-small-en-v1.5.json` are the output of
`analysis/embed/examples/incivility_bias`, one per encoder bundle. Each
has:

- `encoder`: the model id, bundle revision and manifest SHA-256;
- `head`: the fit options (targets, epochs, learning rate, L2) and time;
- `tuned_on_val`: per stage, the threshold with the best micro F1 on
  the Civil Comments val split, and that F1;
- `sets`: for the Civil Comments test split and the English madlibs,
  per stage, at the default threshold (0.5) and at the tuned one:
  - `precision_recall`, `evaluate_incivility_corpus`: per label
    true and false positives and negatives, precision, recall and F1,
    macro and micro averages, and how many positive predictions carry
    evidence;
  - `bias`, `evaluate_incivility_bias`: per label the overall AUC and
    per subgroup the subgroup, BPSN and BNSP AUC and the neutral-mention
    false positive rate at that threshold.

`docs/sentiment.md` ("Precision, recall and identity-term bias on real
data") names the data files, their revisions and digests, and
summarizes the numbers.

Command, run on an RTX 4080 SUPER with the turbo CUDA backend; the
defaults are the fit options of the records (0 and 1 targets, learning
rate 16, 500 epochs):

```sh
TURBO_CUDA_ROOT=/usr cargo build --release -p turbo-analysis-embed \
    --example incivility_bias --features turbo/cuda
target/release/examples/incivility_bias --bundle <bundle dir> --device cuda \
    --civil <dir with all_data_with_identities.csv> \
    --madlibs <dir with en_sentence_templates.csv and words.csv> \
    --cache <vector cache dir> --out <record.json>
```

The bundles are turboembed bundles of
`sentence-transformers/all-MiniLM-L6-v2` (commit c9745ed1) and
`BAAI/bge-small-en-v1.5` (commit 5c38ec7c).

The example's `token_lower` copies the annotator's token form (ASCII
lowercase, trimmed to letters, digits and the apostrophe) so the lexicon
numbers match what `IncivilityAnnotator` does. If the annotator moves to
the shared Unicode token key, these records need a re-run.
