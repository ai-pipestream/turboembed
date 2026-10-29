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

Full accelerator runs are booked with krick-1 in issue #1524, with the
exact commit, encoder digest and command, before starting. The
CPU fixture tests exercise fitting, export/reload agreement and test-set
isolation; they do not establish corpus accuracy. MASC's cross-sentence
entity identity correction, missing type coverage, full German training, actual
spaCy/Stanza comparisons, model packaging and release remain in #1524.
No default model or claim that #1524 is complete is added by this recipe.
