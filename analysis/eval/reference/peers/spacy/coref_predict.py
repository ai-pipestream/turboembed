"""Write spaCy experimental coreference clusters for exported LitBank works.

Usage: coref_predict.py EXPORT_DIR OUT.conll [MODEL]

Reads every <id>.conll under EXPORT_DIR as `turbo-train-head export
--corpus litbank` writes them (the gold words in the gold sentence
framing, the coreference column blank), runs the model on each
document's words as a pretokenized Doc, so the words are exactly the
input's, and writes one CoNLL file with every document, its predicted
clusters in the last column. Singletons are not written: the model's
cleaner drops them, and a word without a bracket is no mention. Version
facts go to stderr as one JSON line. `turbo-train-head peer --column
coref` scores the output.
"""

import json
import sys
from pathlib import Path

import spacy
from spacy.tokens import Doc


def read_work(path):
    """A work's id and its sentences' words, from the export's columns."""
    doc_id, sentences, current = None, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#begin document"):
                doc_id = line.split("(", 1)[1].split(")", 1)[0]
            elif line.startswith("#end document"):
                pass
            elif not line:
                if current:
                    sentences.append(current)
                current = []
            else:
                current.append(line.split("\t")[3])
    if current:
        sentences.append(current)
    return doc_id, sentences


def brackets(clusters, n_words, sentence_ends):
    """The coreference column of each word: `(c)`, `(c`, `c)` joined by |.

    A span crossing a sentence end is clipped to its own sentence: the
    model's mention was over-long, and a bracket may not cross.
    """
    cols = [[] for _ in range(n_words)]
    for cluster_id, spans in enumerate(clusters):
        for start, end in spans:
            end = min(end, min(e for e in sentence_ends if e > start))
            if start + 1 == end:
                cols[start].append(f"({cluster_id})")
            else:
                cols[start].append(f"({cluster_id}")
                cols[end - 1].append(f"{cluster_id})")
    return ["|".join(c) if c else "_" for c in cols]


def main():
    export_dir, out = Path(sys.argv[1]), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "en_coreference_web_trf"
    nlp = spacy.load(model)
    print(
        json.dumps({"spacy": spacy.__version__, "model": model, "pipeline": nlp.pipe_names}),
        file=sys.stderr,
    )
    works = sorted(export_dir.glob("*.conll"))
    with open(out, "w", encoding="utf-8") as f:
        for path in works:
            doc_id, sentences = read_work(path)
            words = [w for s in sentences for w in s]
            doc = Doc(nlp.vocab, words=words)
            doc = nlp(doc)
            # The resolver's mention spans, one `coref_clusters_<i>` span
            # group a cluster, as (first, last + 1) word indices.
            groups = [list(doc.spans[k]) for k in sorted(doc.spans.keys()) if k.startswith("coref_clusters")]
            ends = []
            at_end = 0
            for s in sentences:
                at_end += len(s)
                ends.append(at_end)
            # A span the model gives two clusters is kept in the first: the
            # CoNLL bracket column holds one span once.
            seen_spans = set()
            clusters = []
            for spans in groups:
                kept = []
                for span in spans:
                    se = (span.start, span.end)
                    if se not in seen_spans:
                        seen_spans.add(se)
                        kept.append(se)
                if kept:
                    clusters.append(kept)
            cols = brackets(clusters, len(words), ends)
            f.write(f"#begin document ({doc_id}); part 0\n")
            at = 0
            for s, sent in enumerate(sentences):
                for i, word in enumerate(sent):
                    f.write("\t".join([doc_id, str(s), str(i), word] + ["_"] * 8 + [cols[at]]) + "\n")
                    at += 1
                f.write("\n")
            f.write("#end document\n")
            print(f"{doc_id}: {len(groups)} clusters", file=sys.stderr)


if __name__ == "__main__":
    main()
