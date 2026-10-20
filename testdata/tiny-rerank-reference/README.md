# Tiny cross-encoder reference

These are actual upstream Transformers outputs from a tiny BERT encoder,
not hand-written scores. The encoder and tokenizer come from
`testdata/tiny-bert-bundle`. `generate.py` loads those encoder tensors into
`BertForSequenceClassification`, initializes a pooler and one-logit classifier
with torch seed 707, and saves only the added head and reference outputs here.
The tests assemble a temporary bundle with the existing encoder weights.

The four cases cover empty input, ordinary pairs, Unicode, and query/passages
longer than their budgets. Query budget is eight, sequence limit is 32, and
batch capacity is four. Inference is FP32 CPU with dropout disabled. The
reference program verifies batching parity and max-pools passage windows.

Generated with the repository reference image
`turbo-reference-rerank@sha256:9bcfa35769f1644b37c257e6664a0febef781852b946aa0be5421d0e90bfff7c`,
Transformers 5.17.0 and torch 2.14.0+cpu. To regenerate from the repository root:

```sh
docker run --rm --network none -v "$PWD:/repo:ro" -v "$PWD/testdata/tiny-rerank-reference:/output" turbo-reference-rerank@sha256:9bcfa35769f1644b37c257e6664a0febef781852b946aa0be5421d0e90bfff7c python /repo/testdata/tiny-rerank-reference/generate.py
```

The fixture has the same Apache-2.0 licensing as the repository's tiny BERT
fixture. No pretrained reranker checkpoint is committed here.
