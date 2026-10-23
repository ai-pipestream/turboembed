# Incivility identity-term bias records (#889)

`all-minilm-l6-v2.json` and `bge-small-en-v1.5.json` are the output of
`analysis/embed/examples/incivility_bias`, one per encoder bundle. Each
has the encoder (model id, bundle revision and manifest SHA-256), the
head's fit options and time, and, for the Civil Comments test split and
the English madlibs, the `evaluate_incivility_bias` result of the
lexicon, learned and fused stages: per label the overall AUC and per
subgroup the subgroup, BPSN and BNSP AUC and the neutral-mention false
positive rate at 0.5.

`docs/sentiment.md` ("Identity-term bias on real data") names the data
files, their revisions and digests, and summarizes the numbers.

Command, run on an RTX 4080 SUPER with the turbo CUDA backend:

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
