# Token-classifier references

Frozen per-token labels, softmax probabilities and aggregated entity spans
for `dslim/bert-base-NER`, revision
`d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`.

| file | what |
|---|---|
| `bert_base_ner.json` | 8 texts with every token's probability row, the word grouping, and the spans `transformers`' pipeline returns for aggregation strategies `simple`, `first` and `max`. |

## Schema

```json
{
  "schema": "turbo-reference-token-classify/1",
  "task": "token_classify",
  "model_id": "dslim/bert-base-NER",
  "revision": "d1a3e8f1...",      // hub commit the weights came from
  "source": "...",                // where the weights, config and vocabulary came from
  "labels": ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
  "activation": "softmax",
  "tagging": "BIO",
  "tokenization": { "truncation": "longest_first", "max_length": 512 },
  "aggregation_note": "...",
  "config_sha256": "...",         // the checkout's config.json
  "vocab_sha256": "...",          // the checkout's vocab.txt
  "produced_by": { "framework": "pytorch", "dtype": "float32", "device": "cpu",
                   "torch": "...", "transformers": "...", "numpy": "...", "python": "..." },
  "machine": "krick",
  "date": "2026-09-22",
  "command": "...",               // the command that regenerates this file
  "cases": [
    {
      "id": "classic",            // case name, used in test output
      "text": "Ada Lovelace visited Berlin with colleagues from Microsoft.",
      "n_tokens": 12,
      "input_ids": [101, ...],
      "tokens": [                 // one entry per column, specials included
        { "column": 1, "token": "Ada", "special": false,
          "byte_start": 0, "byte_end": 3,
          "label_id": 3, "label": "B-PER", "score": 0.9997,
          "probs": [ ... ] }      // one per label, in label order
      ],
      "words": [                  // see "Word grouping"
        { "byte_start": 0, "byte_end": 3, "text": "Ada",
          "first_column": 1, "n_tokens": 1,
          "first_label": "B-PER", "first_score": 0.9997,
          "max_label": "B-PER", "max_score": 0.9997 }
      ],
      "aggregation": {            // see "Aggregated spans"
        "simple": [ { "byte_start": 0, "byte_end": 12, "text": "Ada Lovelace",
                      "entity": "PER", "score": 0.99952 } ],
        "first":  [ ... ],
        "max":    [ ... ]
      }
    }
  ]
}
```

All offsets are byte offsets into `text`. `transformers` reports character
offsets; the generator converts them.

The case list covers a person with a sub-word surname, two people and a
place, three places of the same type in a row (so a `B-` tag has to start a
new group rather than extend the previous one), an organization followed by
two places, a hyphenated name (the hyphen is its own word and carries a weak
`I-PER`, which pulls a group's mean score down to 0.84), a case where the
first sub-token and the highest-scoring sub-token of one word disagree
(`Sao`: `B-ORG` first, `I-LOC` strongest, which is the only thing that
separates `first` from `max`), a sentence of names that all split into
sub-tokens, and a sentence with no entities at all.

## Word grouping

`words` is the grouping `transformers`' `aggregate_words` performs: a word is
a maximal run of columns that starts with a token the fast tokenizer did not
mark as a continuation. It carries the first sub-token's label and score and
the highest-scoring sub-token's label and score.

It is in the file because the providers' `TURBO_AGGREGATE_NONE` reports one
span per word whose first sub-token is not `O`, and `transformers` has no
word-aligned equivalent to compare that against: its own `"none"` strategy is
one entry per sub-token.

## Aggregated spans

`aggregation` holds what `transformers`' token-classification pipeline
returns, with `entity` being the pipeline's `entity_group`, that is the label
with its BIO prefix stripped.

The providers in this tree aggregate at the word level, so their strategies
line up with the pipeline's like this, and this is what
`crates/turbo-conformance/tests/live_tasks.rs` gates on:

| provider strategy | reference | why |
|---|---|---|
| `TURBO_AGGREGATE_NONE` | `words` with a first label other than `O` | word-aligned, one span per entity word |
| `TURBO_AGGREGATE_SIMPLE` | `aggregation.first` | the provider takes a word's label from its first sub-token, which is the pipeline's `first` |
| `TURBO_AGGREGATE_FIRST` | `aggregation.first` | same rule, same result |
| `TURBO_AGGREGATE_MAX` | `aggregation.max` | the word's label is its highest-scoring sub-token's |
| `TURBO_AGGREGATE_MODEL` | whatever `contract.aggregation` names | `simple` in the bundle on `krick` |

`aggregation.simple` is stored but is not a gate. The pipeline's `simple`
strategy is token-aligned: it can split one word into two spans, as it does
here for `Sao Paulo` (`Sa` as `ORG`, `o Paulo` as `LOC`) and for
`Nikolaus Blome` (`Nikola` and `us Blome`). The providers cannot produce that
shape, which `providers/cuda/src/lib.rs`'s `aggregate_spans` and
`docs/providers.md` both state. It is in the file so the difference is
visible as numbers instead of prose.

## Regenerating

From the repository root, on a machine with the checkouts under
`~/opt/models`:

```bash
uv run --no-project --with torch --with transformers --with numpy \
    python scripts/gen-reference-tasks.py --machine krick --only token_classify
```

The weights come from the hub at the pinned revision; the cased tokenizer
(`vocab.txt`, `do_lower_case` false) and the config come from
`~/opt/models/ner`, which holds no PyTorch weights of its own. The bundle
under `~/opt/bundles/ner-onnx` on `krick` declares `model_id` `ner` rather
than the Hugging Face id, so the id above comes from the checkout's
`config.json` and the receipts, not from the bundle.

The texts avoid non-ASCII punctuation. The providers split words on
whitespace and on ASCII punctuation only, while BERT's basic tokenizer also
splits on Unicode punctuation, so a text containing an em dash or a curly
quote would put the two word groupings out of step for a reason that has
nothing to do with the model. Letters with diacritics are unaffected and are
covered.
