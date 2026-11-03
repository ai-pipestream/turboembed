"""Write spaCy's tags for the given words of a CoNLL-U file.

Usage: tag_predict.py WORDS.conllu OUT.conllu [MODEL]

Each sentence's word forms are tagged as a pretokenized Doc, so the
words are exactly the input's and only the tagger decides: UPOS
(`pos_`) goes in column 4, the language-specific tag (`tag_`, Penn for
English) in column 5. `turbo-train-head export` writes the input and
`turbo-train-head peer` scores the output. The parser, lemmatizer and
named entities are not run; they do not change a tag. Version facts go
to stderr as one JSON line.
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


def main():
    src, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_sm"
    nlp = spacy.load(model, exclude=["parser", "lemmatizer", "ner"])
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
        for n, doc in enumerate(nlp.pipe(docs, batch_size=64), start=1):
            f.write(f"# sent_id = {n}\n")
            for i, tok in enumerate(doc, start=1):
                f.write("\t".join([str(i), tok.text, "_", tok.pos_ or "_", tok.tag_ or "_", "_", "_", "_", "_", "_"]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
