"""Write spaCy CoNLL-U predictions for the raw text of a gold CoNLL-U file.

Usage: predict.py GOLD.conllu OUT.conllu [MODEL]

Documents are taken from `# newdoc` groups, their `# text` sentences
joined with one space, so the peer tokenizes and segments sentences
itself: nothing gold reaches it but the document text. Version facts
go to stderr as one JSON line.
"""

import json
import sys

import spacy


def documents(path):
    docs, doc, text = [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# newdoc"):
                if doc:
                    docs.append(doc)
                doc = []
            elif line.startswith("# text = "):
                if doc is None:
                    doc = []
                doc.append(line[len("# text = "):])
    if doc:
        docs.append(doc)
    return docs


def main():
    gold, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_sm"
    nlp = spacy.load(model)
    meta = nlp.meta
    import importlib.metadata

    print(json.dumps({
        "spacy": spacy.__version__,
        "model": meta.get("name"),
        "model_version": meta.get("version"),
        "lang": meta.get("lang"),
        "pipeline": nlp.pipe_names,
        "torch": importlib.metadata.version("torch"),
        "curated_transformers": importlib.metadata.version("curated-transformers"),
        "spacy_curated_transformers": importlib.metadata.version("spacy-curated-transformers"),
    }), file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        for sentences in documents(gold):
            doc = nlp(" ".join(sentences))
            for sent in doc.sents:
                for i, tok in enumerate(sent, start=1):
                    head = 0 if tok.head.i == tok.i else tok.head.i - sent.start + 1
                    deprel = "root" if tok.dep_ == "ROOT" else (tok.dep_ or "dep")
                    feats = str(tok.morph) or "_"
                    f.write("\t".join([
                        str(i), tok.text, tok.lemma_ or "_", tok.pos_ or "_",
                        tok.tag_ or "_", feats, str(head), deprel, "_", "_",
                    ]) + "\n")
                f.write("\n")


if __name__ == "__main__":
    main()
