"""Write spaCy BIO entity labels for gold-tokenized sentences.

Usage: ner_predict.py GOLD.tsv OUT.tsv [MODEL] [LABELSET]

The gold file is `token<TAB>label` lines with blank lines between
sentences (the bench's entity gold form). Tokens are given to spaCy
as a pretokenized Doc, as WNUT-17 and MultiCoNER are conventionally
run; only the tok2vec/transformer and ner components run.

Label maps, chosen by LABELSET (default wnut). wnut (spaCy
OntoNotes to the WNUT-17 label set): PERSON to person, ORG to
corporation, GPE, LOC and FAC to location, PRODUCT to product,
WORK_OF_ART to creative-work, NORP to group. multiconer (to the
MultiCoNER v2 coarse groups): PERSON to PER, ORG and NORP to GRP,
GPE, LOC and FAC to LOC, PRODUCT to PROD, WORK_OF_ART to CW.
Every other spaCy label (dates, money, quantities and the rest)
has no counterpart in either set and is dropped. The map is part
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
    gold, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_sm"
    labelset = sys.argv[4] if len(sys.argv) > 4 else "wnut"
    label_map = MAPS[labelset]
    nlp = spacy.load(model)
    meta = nlp.meta
    print(json.dumps({
        "spacy": spacy.__version__,
        "model": meta.get("name"),
        "model_version": meta.get("version"),
        "label_map": label_map,
        "label_set": labelset,
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
            f.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main()
