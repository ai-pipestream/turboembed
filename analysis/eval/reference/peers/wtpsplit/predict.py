"""Write wtpsplit sentence predictions for the raw text of a gold CoNLL-U file.

Usage: predict.py GOLD.conllu OUT.txt [MODEL]

Documents are taken from `# newdoc` groups, their `# text` sentences
joined with one space. Output is one sentence per line, documents
separated by blank lines, in gold order (the bench's sentence-list
format); sentence-internal newlines are written as spaces. Version
facts go to stderr as one JSON line. The model is baked into the
image; nothing is downloaded at run time.
"""

import importlib.metadata
import json
import sys

from wtpsplit import SaT


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
    model = sys.argv[3] if len(sys.argv) > 3 else "sat-3l"
    sat = SaT(model)
    print(json.dumps({
        "wtpsplit": importlib.metadata.version("wtpsplit"),
        "model": model,
    }), file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        for i, sentences in enumerate(documents(gold)):
            if i:
                f.write("\n")
            for sent in sat.split(" ".join(sentences)):
                f.write(sent.replace("\n", " ").strip() + "\n")


if __name__ == "__main__":
    main()
