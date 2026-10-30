"""Write GLiNER BIO entity labels for gold-tokenized sentences.

Usage: ner_predict.py GOLD.tsv OUT.tsv

The gold file is `token<TAB>label` lines with blank lines between
sentences (the bench's entity gold form); the gold labels are
ignored here. GLiNER (urchade/gliner_small-v2.1, the checkpoint
the zoo pins) is a zero-shot span model with no pretokenized input,
so each sentence is joined with single spaces and run as text, and
the entity char spans are mapped back onto the gold tokens:

- The prompt labels are the WNUT-17 kinds as words: person,
  location, corporation, creative work, group, product. `creative
  work` is written back as creative-work; every other prompt label
  is its kind unchanged. The threshold is GLiNER's default 0.5.
- A gold token is covered by an entity when their char spans in
  the joined text overlap. Entities claim tokens highest score
  first; a claimed token keeps its first claim. The first token
  an entity claims is B-, the rest I-.

That join-and-map-back step is part of the record's conditioning:
unlike the spaCy and Stanza runners, this peer does not see the
gold tokenization as input, only as the grid its spans land on.
"""

import json
import sys

from gliner import GLiNER

LABELS = ["person", "location", "corporation", "creative work", "group", "product"]
MAP = {"creative work": "creative-work"}
THRESHOLD = 0.5
MODEL_DIR = "/gliner-model"


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


def label_tokens(model, tokens):
    spans, at = [], 0
    for token in tokens:
        spans.append((at, at + len(token)))
        at += len(token) + 1
    text = " ".join(tokens)
    claimed = [None] * len(tokens)
    entities = sorted(
        model.predict_entities(text, LABELS, threshold=THRESHOLD),
        key=lambda e: -e["score"],
    )
    for entity in entities:
        kind = MAP.get(entity["label"], entity["label"])
        mine = [
            i
            for i, (start, end) in enumerate(spans)
            if claimed[i] is None and start < entity["end"] and entity["start"] < end
        ]
        for j, i in enumerate(mine):
            claimed[i] = ("B-" if j == 0 else "I-") + kind
    return [c if c is not None else "O" for c in claimed]


def main():
    gold, out = sys.argv[1], sys.argv[2]
    model = GLiNER.from_pretrained(MODEL_DIR)
    print(json.dumps({
        "gliner": "0.2.21",
        "checkpoint": "urchade/gliner_small-v2.1@4e091416cf7c3481db542c2a3d26156916f3a47f",
        "labels": LABELS,
        "label_map": MAP,
        "threshold": THRESHOLD,
    }), file=sys.stderr)
    sentences = gold_sentences(gold)
    with open(out, "w", encoding="utf-8") as f:
        for i, tokens in enumerate(sentences):
            if i:
                f.write("\n")
            f.write("\n".join(label_tokens(model, tokens)) + "\n")
            if (i + 1) % 250 == 0:
                print(f"sentences {i + 1}/{len(sentences)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
