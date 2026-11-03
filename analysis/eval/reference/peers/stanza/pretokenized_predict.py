"""Write Stanza's full annotation for the given words of a CoNLL-U file.

Usage: pretokenized_predict.py WORDS.conllu OUT.conllu [LANG]

Each sentence's word forms are the pipeline's input as given
(tokenize_pretokenized), so the words are exactly the input's and only
the pipeline decides: lemma in column 3, UPOS in 4, XPOS in 5, features
in 6, head and relation in 7 and 8. The multiword token processor is
left out: the input words are already the treebank's. `turbo-train-head
export` writes the input and `turbo-train-head peer` scores the output.
Version facts go to stderr as one JSON line. Models are baked into the
image; nothing is downloaded at run time.
"""

import json
import sys

import stanza
from stanza import DownloadMethod


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
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    nlp = stanza.Pipeline(
        lang=lang,
        dir="/stanza",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    print(json.dumps({
        "stanza": stanza.__version__,
        "lang": lang,
        "processors": "tokenize,pos,lemma,depparse",
        "pretokenized": True,
    }), file=sys.stderr)
    with open(out, "w", encoding="utf-8") as f:
        for n, sent in enumerate(sentences(src), start=1):
            doc = nlp([sent])
            f.write(f"# sent_id = {n}\n")
            for s in doc.sentences:
                for word in s.words:
                    f.write("\t".join([
                        str(word.id),
                        word.text,
                        word.lemma or "_",
                        word.upos or "_",
                        word.xpos or "_",
                        word.feats or "_",
                        str(word.head if word.head is not None else 0),
                        word.deprel or "dep",
                        "_",
                        "_",
                    ]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
