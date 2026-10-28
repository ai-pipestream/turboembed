"""Write Trankit CoNLL-U predictions for the raw text of a gold CoNLL-U file.

Usage: predict.py GOLD.conllu OUT.conllu

Documents are taken from `# newdoc` groups, their `# text` sentences
joined with one space, so the peer tokenizes and segments sentences
itself. Multiword tokens are written expanded (word rows only).
Version facts go to stderr as one JSON line. Models are baked into
the image; nothing is downloaded at run time.
"""

import importlib.metadata
import json
import sys

from trankit import Pipeline


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


def words_of(sentence):
    words = []
    for token in sentence.get("tokens", []):
        words.extend(token.get("expanded") or [token])
    return words


def main():
    gold, out = sys.argv[1], sys.argv[2]
    nlp = Pipeline("english", cache_dir="/trankit-cache", gpu=False)
    print(json.dumps({
        "trankit": importlib.metadata.version("trankit"),
        "lang": "english",
        "embedding": "xlm-roberta-base",
    }), file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        for sentences in documents(gold):
            doc = nlp(" ".join(sentences))
            for sent in doc.get("sentences", []):
                words = words_of(sent)
                position = {word.get("id"): i + 1 for i, word in enumerate(words)}
                for i, word in enumerate(words, start=1):
                    head = word.get("head", 0)
                    head = position.get(head, 0) if isinstance(head, int) and head > 0 else 0
                    f.write("\t".join([
                        str(i),
                        str(word.get("text", "_")),
                        str(word.get("lemma") or "_"),
                        str(word.get("upos") or "_"),
                        str(word.get("xpos") or "_"),
                        str(word.get("feats") or "_"),
                        str(head),
                        str(word.get("deprel") or "dep"),
                        "_",
                        "_",
                    ]) + "\n")
                f.write("\n")


if __name__ == "__main__":
    main()
