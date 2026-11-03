"""Write Stanza CoNLL-U predictions for the raw text of a gold CoNLL-U file.

Usage: predict.py GOLD.conllu OUT.conllu [LANG]

Documents are taken from `# newdoc` groups, their `# text` sentences
joined with one space, so the peer tokenizes and segments sentences
itself. Version facts go to stderr as one JSON line. Models are baked
into the image; nothing is downloaded at run time.
"""

import json
import sys

import stanza
from stanza import DownloadMethod


def documents(path):
    docs, doc = [], None
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
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    nlp = stanza.Pipeline(
        lang=lang,
        dir="/stanza",
        processors="tokenize,mwt,pos,lemma,depparse",
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    print(json.dumps({
        "stanza": stanza.__version__,
        "lang": lang,
        "processors": "tokenize,mwt,pos,lemma,depparse",
    }), file=sys.stderr)
    docs_text = documents(gold)
    with open(out, "w", encoding="utf-8") as f:
        for i, sentences in enumerate(docs_text):
            doc = nlp(" ".join(sentences))
            for sent in doc.sentences:
                for word in sent.words:
                    f.write("\t".join([
                        str(word.id),
                        word.text,
                        word.lemma or "_",
                        word.upos or "_",
                        word.xpos or "_",
                        word.feats or "_",
                        str(word.head if word.head is not None else 0),
                        word.deprel or "dep",
                        word.deps or "_",
                        "_",
                    ]) + "\n")
                f.write("\n")
            if (i + 1) % 50 == 0:
                print(f"documents {i + 1}/{len(docs_text)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
