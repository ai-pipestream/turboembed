"""Write Trankit's full annotation for the given words of a CoNLL-U file.

Usage: pretokenized_predict.py WORDS.conllu OUT.conllu [TRANKIT_LANG]

Each sentence's word forms are the pipeline's input as given (a list of
token lists is pretokenized input to Trankit), so the words are exactly
the input's and only the pipeline decides: lemma in column 3, UPOS in 4,
XPOS in 5, features in 6, head and relation in 7 and 8. Lemmas come from
a separate tagless `lemmatize` pass over the same words (Trankit's
`posdep` does not lemmatize). `turbo-train-head export` writes the input
and `turbo-train-head peer` scores the output. Version facts go to stderr
as one JSON line. Models are baked into the image; nothing is downloaded
at run time.
"""

import importlib.metadata
import json
import sys

from trankit import Pipeline


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
    lang = sys.argv[3] if len(sys.argv) > 3 else "english"
    nlp = Pipeline(lang, cache_dir="/trankit-cache", gpu=False)
    print(json.dumps({
        "trankit": importlib.metadata.version("trankit"),
        "lang": lang,
        "embedding": "xlm-roberta-base",
        "pretokenized": True,
    }), file=sys.stderr)
    sents = sentences(src)
    tagged = nlp.posdep(sents)
    lemmatized = nlp.lemmatize(sents)
    with open(out, "w", encoding="utf-8") as f:
        for n, (sent, lem_sent) in enumerate(zip(tagged["sentences"], lemmatized["sentences"]), start=1):
            f.write(f"# sent_id = {n}\n")
            for i, (word, lem) in enumerate(zip(sent["tokens"], lem_sent["tokens"]), start=1):
                head = word.get("head", 0)
                f.write("\t".join([
                    str(i),
                    str(word.get("text", "_")),
                    str(lem.get("lemma") or "_"),
                    str(word.get("upos") or "_"),
                    str(word.get("xpos") or "_"),
                    str(word.get("feats") or "_"),
                    str(head if isinstance(head, int) else 0),
                    str(word.get("deprel") or "dep"),
                    "_",
                    "_",
                ]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()
