# Classifier references

Frozen sequence-classification logits and softmax probabilities for
`distilbert/distilbert-base-uncased-finetuned-sst-2-english`, revision
`714eb0fa89d2f80546fda750413ed43d93601a13`.

| file | what |
|---|---|
| `sst2_distilbert.json` | 12 texts with the two logits and the two probabilities. |

## Schema

```json
{
  "schema": "turbo-reference-classify/1",
  "task": "classify",
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "revision": "714eb0fa...",      // hub commit the weights came from
  "source": "...",                // where the weights, config and tokenizer came from
  "labels": ["NEGATIVE", "POSITIVE"],
  "activation": "softmax",
  "tokenization": { "truncation": "longest_first", "max_length": 256 },
  "config_sha256": "...",         // the checkout's config.json
  "tokenizer_sha256": "...",      // the checkout's tokenizer.json
  "produced_by": { "framework": "pytorch", "dtype": "float32", "device": "cpu",
                   "torch": "...", "transformers": "...", "numpy": "...", "python": "..." },
  "machine": "rtx4080",
  "date": "2026-09-22",
  "command": "...",               // the command that regenerates this file
  "cases": [
    {
      "id": "clearly_positive",   // case name, used in test output
      "text": "...",
      "n_tokens": 12,             // columns after truncation
      "input_ids": [101, ...],
      "logits": [-4.349685, 4.702411],   // one per label, in label order
      "probs":  [0.000117, 0.999883]     // softmax of the logits
    }
  ]
}
```

`labels` is the order the probabilities are in, and the same order the
bundle's `contract.labels` must declare.

The case list covers a clearly positive and a clearly negative review, a mild
one of each, a mixed one, a neutral statement, a negation, a one-word text,
an empty string, an all-caps text with heavy punctuation, a French text, and
a review over 256 tokens so it is truncated. The empty string is the case
that matters most: its probabilities are 0.25 and 0.75, not a saturated pair,
so an implementation that mishandles a two-token row is visible.

`crates/turbo-conformance/tests/live_tasks.rs` writes all 12 texts as one
batch, so the run also covers padding: every reference row was computed
unpadded, and a provider that lets padding reach the attention or the pooling
fails the gate.

## Regenerating

From the repository root, on a machine with the checkouts under
`~/opt/models`:

```bash
uv run --no-project --with torch --with transformers --with numpy \
    python scripts/gen-reference-tasks.py --machine rtx4080 --only classify
```

The weights come from the hub at the pinned revision; the tokenizer and the
config come from `~/opt/models/sst2`, which holds no PyTorch weights of its
own. The bundle under `~/opt/bundles/sst2-onnx` on the reference host declares
`model_id` `sst2` rather than the Hugging Face id, so the id above comes from
the checkout's `config.json` and the receipts, not from the bundle.
