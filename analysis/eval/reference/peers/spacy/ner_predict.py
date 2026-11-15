"""Write spaCy BIO entity labels for gold-tokenized sentences.

Usage: ner_predict.py [--with-tokens] [--gpu] GOLD.tsv OUT.tsv [MODEL] [LABELSET]

The gold file is `token<TAB>label` lines with blank lines between
sentences (the bench's entity gold form). Tokens are given to spaCy
as a pretokenized Doc, as WNUT-17 and MultiCoNER are conventionally
run; only the tok2vec/transformer and ner components run. With
`--with-tokens`, output lines are `token<TAB>label` for
`turbo-train-head peer --column bio`; `--gpu` enables `spacy.prefer_gpu()`.

Label maps, chosen by LABELSET (default wnut). wnut (spaCy
OntoNotes to the WNUT-17 label set): PERSON to person, ORG to
corporation, GPE, LOC and FAC to location, PRODUCT to product,
WORK_OF_ART to creative-work, NORP to group. multiconer (to the
MultiCoNER v2 coarse groups): PERSON to PER, ORG and NORP to GRP,
GPE, LOC and FAC to LOC, PRODUCT to PROD, WORK_OF_ART to CW.
nemotron-pii (to shared Nemotron-PII types, #1888): ORG to
company_name, DATE to date, TIME to time, LANGUAGE to language
(nemotron-pii-raw keeps those four OntoNotes names for explicit
`turbo-train-head peer --map` provenance).
Every other spaCy label (dates, money, quantities and the rest)
has no counterpart in the target set and is dropped. The map is part
of the record's conditioning: entity types are compared after it,
never before.
"""

import json
import sys

import spacy
from spacy.tokens import Doc

MAPS = {
    "wnut": {
        "PERSON": "person",
        "ORG": "corporation",
        "GPE": "location",
        "LOC": "location",
        "FAC": "location",
        "PRODUCT": "product",
        "WORK_OF_ART": "creative-work",
        "NORP": "group",
    },
    "multiconer": {
        "PERSON": "PER",
        "ORG": "GRP",
        "NORP": "GRP",
        "GPE": "LOC",
        "LOC": "LOC",
        "FAC": "LOC",
        "PRODUCT": "PROD",
        "WORK_OF_ART": "CW",
    },
    "nemotron-pii": {
        "ORG": "company_name",
        "DATE": "date",
        "TIME": "time",
        "LANGUAGE": "language",
    },
    "nemotron-pii-raw": {
        "ORG": "ORG",
        "DATE": "DATE",
        "TIME": "TIME",
        "LANGUAGE": "LANGUAGE",
    },
}


def gold_sentences(path):
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") and "\t" not in line:
                continue
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            current.append(line.split("\t")[0])
    if current:
        sentences.append(current)
    return sentences


def main():
    args = [a for a in sys.argv[1:] if a not in ("--with-tokens", "--gpu")]
    with_tokens = "--with-tokens" in sys.argv[1:]
    use_gpu = "--gpu" in sys.argv[1:]
    gold, out = args[0], args[1]
    model = args[2] if len(args) > 2 else "en_core_web_sm"
    labelset = args[3] if len(args) > 3 else "wnut"
    label_map = MAPS[labelset]
    if use_gpu:
        spacy.prefer_gpu()
    nlp = spacy.load(model)
    meta = nlp.meta
    print(json.dumps({
        "spacy": spacy.__version__,
        "model": meta.get("name"),
        "model_version": meta.get("version"),
        "label_map": label_map,
        "label_set": labelset,
        "with_tokens": with_tokens,
        "gpu": use_gpu,
    }), file=sys.stderr)
    keep = {"tok2vec", "transformer", "ner"}
    disable = [name for name in nlp.pipe_names if name not in keep]
    sentences = gold_sentences(gold)
    docs = (Doc(nlp.vocab, words=tokens) for tokens in sentences)
    with open(out, "w", encoding="utf-8") as f:
        for i, (tokens, doc) in enumerate(
            zip(sentences, nlp.pipe(docs, batch_size=256, disable=disable))
        ):
            if i:
                f.write("\n")
            labels = ["O"] * len(tokens)
            for ent in doc.ents:
                kind = label_map.get(ent.label_)
                if kind is None:
                    continue
                labels[ent.start] = f"B-{kind}"
                for j in range(ent.start + 1, ent.end):
                    labels[j] = f"I-{kind}"
            if with_tokens:
                f.write("\n".join(f"{t}\t{l}" for t, l in zip(tokens, labels)) + "\n")
            else:
                f.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main()
