# MASC Penn POS, test split (#1649)

The rows the MASC Penn POS head is gated against, all on the recorded
MASC 3.0.0 release (corpus digest `445449f5...`, the recipes' split,
6,384 test sentences, 85,292 words), all receipts of
`turbo-train-head` at commit `88cecf90`:

| row | receipt | accuracy | vs majority (paired bootstrap, 95%) |
|---|---|---|---|
| gold | | 1.0000 | |
| majority baseline (each word's commonest training tag) | `majority.json` | 0.8964 | |
| spaCy 3.8.7 `en_core_web_trf` 3.8.0, `tag_` as written | `spacy-trf-raw.json` | 0.8753 | -0.0211 [-0.0248, -0.0170] |
| the same, tag spellings mapped (`--map`) | `spacy-trf-mapped.json` | 0.8853 | -0.0110 [-0.0147, -0.0073] |
| our head | to come from the 4080 run | | |

No OpenNLP model or number is part of this record (#1509).

## Reading note

spaCy tags the gold words as given (pretokenized). It sits below the
majority baseline because MASC's tag conventions are not OntoNotes':

- The map rewrites spellings only: `-LRB-` `(`, `-RRB-` `)`, `HYPH` `-`,
  the opening quote's tag to `''`, `NFP` `...`.
- Left as differences: MASC tags every "to" `TO` where spaCy writes `IN`
  for a preposition (744 words), and MASC's gold gives `NN` to curly
  quotes, `*` and the pieces of a split "don't" (about 1,000 words), which
  no OntoNotes tagger writes.

So this row is the off-the-shelf peer on MASC's gold, not a ranking of
taggers in general. The majority baseline learns MASC's conventions from
MASC training data, as our head does.

## Reproduce

```sh
turbo-train-head export --corpus masc-pos --dir MASC-3.0.0 --out masc-test.conllu
# masc-test.conllu: sha256 37f9a06b8d33f7982ca0ffff6e4deead65f02d6086816ed033ac49446c7b5c4f
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
pip install spacy==3.8.7 click==8.1.8 \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
python analysis/eval/reference/peers/spacy/tag_predict.py masc-test.conllu spacy-trf.conllu en_core_web_trf
# spacy-trf.conllu: sha256 0fd96b59eaca3bb84d8a577af73098ae2b7cc19456972ec0f285550da60ff8cb (3 min on CPU)
turbo-train-head peer --corpus masc-pos --dir MASC-3.0.0 --peer spacy-3.8.7-en_core_web_trf-3.8.0 \
  --predictions spacy-trf.conllu --out raw
turbo-train-head peer --corpus masc-pos --dir MASC-3.0.0 --peer spacy-3.8.7-en_core_web_trf-3.8.0 \
  --predictions spacy-trf.conllu --map "-LRB-=(,-RRB-=),HYPH=-,\`\`='',NFP=..." --out mapped
turbo-train-head baseline --corpus masc-pos --dir MASC-3.0.0 --out majority
```
