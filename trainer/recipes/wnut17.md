# WNUT-17 native NER recipe

Part of #1524. This recipe trains a token head with CRF decoding on a
frozen encoder using `turbo-trainer`. It writes the existing native head
format, targeting `turboembed:entities`; no external model is its teacher.

## Source and licence

Leon Derczynski, Eric Nichols, Marieke van Erp and Nut Limsopatham,
*Results of the WNUT2017 Shared Task on Novel and Emerging Entity
Recognition* (2017). The publisher's
[README](https://github.com/leondz/emerging_entities_17/blob/e52a2d2a71ac1a051ca2d532eef28653c5c02603/README.md)
states CC BY 4.0. Retain this attribution and the corpus link with any
head trained on it. The #1524 ruling permits default-tier WNUT training;
this permission does not apply to MultiCoNER v2's share-alike data.

The recipe verifies all files at revision
`e52a2d2a71ac1a051ca2d532eef28653c5c02603`. Download them into an
application-owned corpus directory, keeping them for subsequent runs:

```sh
mkdir -p corpora/wnut17
for file in wnut17train.conll emerging.dev.conll emerging.test.annotated; do
  curl --fail --location --output "corpora/wnut17/$file" \
    "https://raw.githubusercontent.com/leondz/emerging_entities_17/e52a2d2a71ac1a051ca2d532eef28653c5c02603/$file"
done
turbo-train-head check --corpus wnut17 --dir corpora/wnut17
```

`emerging.test.annotated` is the single-label test file selected by the
[publisher's dataset loader](https://huggingface.co/datasets/leondz/wnut_17/blob/main/wnut_17.py).
Its size is 192,425 bytes and its SHA-256 is
`2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0`.
`emerging.test.conll` has alternative labels and must not be substituted
or reduced to the first alternative. Tab-only lines in the training file
are sentence boundaries. The recipe reads borrowed sentence blocks and
refuses malformed or unknown BIO labels without dropping rows.

The verified split sizes are 3,394/1,009/1,287 sentences and
62,730/15,733/23,394 tokens for train/dev/test. Types are person, location,
group, creative-work, corporation and product. They are not silently
mapped to MASC's person/location/organization/date schema, nor treated
as negative labels for types that this corpus does not annotate. The head
vocabulary is the fixed 13-label BIO schema, even if a local training
fixture lacks a type. It never comes from dev or test labels.

## Training and peer scoring

```sh
turbo-train-head train --corpus wnut17 --dir corpora/wnut17 \
  --bundle bundles/encoder --out runs/wnut17 --crf --threads 2 --commit COMMIT
turbo-train-head export --corpus wnut17 --dir corpora/wnut17 --out wnut-test.tsv
turbo-train-head peer --corpus wnut17 --dir corpora/wnut17 \
  --peer PEER-MODEL-AND-VERSION --column bio --predictions peer.tsv \
  --out runs/wnut-peer --commit COMMIT
```

Export contains each original token followed by a tab and `_`, with a
blank line between sentences; it contains no gold labels. Run spaCy or
Stanza on those exact tokens and replace `_` with its BIO prediction.
Import checks every token and sentence boundary, refuses invalid BIO,
and scores exact spans against gold with the same majority baseline and
paired bootstrap used for our head. `--map` permits explicit label
spelling mappings, such as `B-PERSON=B-person,I-PERSON=I-person`.
Do not equate a peer's broad organization label with corporation or
group without a separately documented evaluation protocol. A receipt
records the prediction-file digest, peer name and mapping. Authored
test predictions are protocol tests, never peer performance evidence.

Full accelerator runs are booked in issue #1524, with the
exact commit, encoder digest and command, before starting. The
CPU fixture tests exercise fitting, export/reload agreement and test-set
isolation; they do not establish corpus accuracy. MASC's cross-sentence
entity identity correction, missing type coverage, full German training, actual
spaCy/Stanza comparisons, model packaging and release remain in #1524.
No default model or claim that #1524 is complete is added by this recipe.

## Open-label span typing (`HEAD_OPEN_SPAN`, #1735)

`train-open-span --corpus wnut17` trains a zero-shot / open-label span
matcher (`open_ner` on `turboembed:entities`, `HEAD_OPEN_SPAN`,
`DECODE_SPANS_FLAT`, `max_words = 8`, `width_dim = 64`,
`CLASS_PERMISSIVE`) on the same pinned WNUT-17 files, encoding the six
type names (`person`, `location`, `group`, `creative-work`,
`corporation`, `product`) with the shared frozen encoder:

```sh
turbo-train-head train-open-span --corpus wnut17 --dir corpora/wnut17 \
  --bundle bundles/modernbert-base-l12 --device cuda \
  --epochs 15 --patience 5 --batch 32 --lr 0.003 --threads 8 \
  --out runs/wnut17-open-span --commit COMMIT

turbo-train-head train-open-span --corpus wnut17 --dir corpora/wnut17 \
  --bundle bundles/modernbert-base-l12 --device cuda \
  --hold-out location --epochs 15 --patience 5 --batch 32 --lr 0.003 --threads 8 \
  --out runs/wnut17-open-span-holdout --commit COMMIT
```

Recorded results on `answerdotai/ModernBERT-base` (`layers = 12`,
`hidden = 768`, manifest SHA-256
`3c520dec8d833a3594b15567d1de0c8c039a571c728fbf2a6247d4fa885a64b4`),
packaged as `turbo-en-wnut17-modernbert-base-l12-open-span-1`
(`CLASS_PERMISSIVE`, `COMPONENT_ENTITY_EXTRACTOR`):

| Run | Split / Subset | Model Span F1 | Majority Baseline F1 | Dictionary Matcher F1 | Paired Bootstrap Delta (95% CI) |
|---|---|---:|---:|---:|---:|
| All 6 types | dev (1,009 sents) | 0.3023 | - | - | - |
| All 6 types | test all 6 types (1,079 gold spans) | 0.2642 (P 0.3956, R 0.1983) | 0.0594 | 0.0000 | +0.2048 ([+0.1730, +0.2361]) |
| `--hold-out location` (trained on 5 types) | dev seen 5 types | 0.3212 | - | - | - |
| `--hold-out location` (trained on 5 types) | test all 6 types (1,079 gold spans) | 0.2513 | 0.0575 | 0.0000 | +0.1938 ([+0.1645, +0.2248]) |
| `--hold-out location` (trained on 5 types) | test seen 5 types (929 gold spans) | 0.2729 | 0.0653 | 0.0000 | - |
| `--hold-out location` (trained on 5 types) | test unseen `location` (150 gold spans) | 0.0343 | 0.0000 | 0.0000 | - |
