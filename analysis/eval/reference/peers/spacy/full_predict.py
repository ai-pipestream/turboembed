"""Write spaCy's full annotation for the given words of a CoNLL-U file.

Usage: full_predict.py WORDS.conllu OUT.conllu [MODEL]

Each sentence's word forms are read as a pretokenized Doc, so the words
are exactly the input's and only the pipeline decides: lemma in column 3,
UPOS in 4, XPOS in 5, the morphological features in 6, head and relation
in 7 and 8. `turbo-train-head export` writes the input and
`turbo-train-head peer` scores the output (--column lemma or deps).
Version facts go to stderr as one JSON line.
"""

import importlib.metadata
import json
import sys

import spacy
from spacy.tokens import Doc


def sentences(path):
    out, words = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    out.append(words)
                words = []
            elif not line.startswith("#"):
                words.append(line.split("\t")[1])
    if words:
        out.append(words)
    return out


def feats(token):
    morph = token.morph.to_dict()
    if not morph:
        return "_"
    out = []
    for k, vs in sorted(morph.items()):
        if isinstance(vs, str):
            vs = [vs]
        out.append(f"{k}={','.join(str(v) for v in vs)}")
    return "|".join(out)


def main():
    src, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_trf"
    nlp = spacy.load(model, exclude=["ner"])
    meta = nlp.meta
    facts = {
        "spacy": spacy.__version__,
        "model": meta.get("name"),
        "model_version": meta.get("version"),
        "pipeline": nlp.pipe_names,
    }
    try:
        facts["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    print(json.dumps(facts), file=sys.stderr)
    docs = (Doc(nlp.vocab, words=w) for w in sentences(src))
    with open(out, "w", encoding="utf-8") as f:
        for n, doc in enumerate(nlp.pipe(docs, batch_size=16), start=1):
            f.write(f"# sent_id = {n}\n")
            for i, tok in enumerate(doc, start=1):
                head = tok.head.i + 1 if tok.head is not tok else 0
                f.write("\t".join([
                    str(i), tok.text, tok.lemma_ or "_", tok.pos_ or "_", tok.tag_ or "_",
                    feats(tok), str(head), tok.dep_ or "_", "_", "_",
                ]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
