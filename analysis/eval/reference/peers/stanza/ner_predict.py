"""Write Stanza BIO entity labels for gold-tokenized sentences.

Usage: ner_predict.py GOLD.tsv OUT.tsv

The gold file is `token<TAB>label` lines with blank lines between
sentences (the bench's entity gold form). Stanza runs with
pretokenized input, as WNUT-17 and MultiCoNER are conventionally
run. Models are baked into the image; nothing downloads at run time.

Label map (Stanza OntoNotes to the WNUT-17 label set): PERSON to
person, ORG to corporation, GPE, LOC and FAC to location, PRODUCT
to product, WORK_OF_ART to creative-work, NORP to group. Every
other type is dropped. The map is part of the record's conditioning.
"""

import json
import sys

import stanza
from stanza import DownloadMethod

MAP = {
    "PERSON": "person",
    "ORG": "corporation",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PRODUCT": "product",
    "WORK_OF_ART": "creative-work",
    "NORP": "group",
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
    nlp = stanza.Pipeline(
        lang="en",
        dir="/stanza",
        processors="tokenize,ner",
        tokenize_pretokenized=True,
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    print(json.dumps({
        "stanza": stanza.__version__,
        "lang": "en",
        "processors": "tokenize,ner (pretokenized)",
        "label_map": MAP,
    }), file=sys.stderr)
    sentences = gold_sentences(gold)
    with open(out, "w", encoding="utf-8") as f:
        for i, tokens in enumerate(sentences):
            if i:
                f.write("\n")
            doc = nlp([tokens])
            labels = ["O"] * len(tokens)
            for sent in doc.sentences:
                for ent in sent.entities:
                    kind = MAP.get(ent.type)
                    if kind is None or not ent.tokens:
                        continue
                    start = ent.tokens[0].id[-1] - 1 if isinstance(ent.tokens[0].id, tuple) else ent.tokens[0].id - 1
                    labels[start] = f"B-{kind}"
                    for token in ent.tokens[1:]:
                        index = token.id[-1] - 1 if isinstance(token.id, tuple) else token.id - 1
                        labels[index] = f"I-{kind}"
            f.write("\n".join(labels) + "\n")
            if (i + 1) % 250 == 0:
                print(f"sentences {i + 1}/{len(sentences)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
