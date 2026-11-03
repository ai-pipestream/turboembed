"""Write Stanza coreference clusters for exported LitBank works.

Usage: coref_predict.py EXPORT_DIR OUT.conll [LANG]

Reads every <id>.conll under EXPORT_DIR as `turbo-train-head export
--corpus litbank` writes them (the gold words in the gold sentence
framing, the coreference column blank), runs Stanza's pipeline with
pretokenized input, so the words are exactly the input's, and writes one
CoNLL file with every document, its predicted clusters in the last
column. Version facts go to stderr as one JSON line. Models are baked
into the image; nothing is downloaded at run time.
`turbo-train-head peer --column coref` scores the output.
"""

import json
import sys
from pathlib import Path

import stanza
from stanza import DownloadMethod


def read_work(path):
    """A work's id and its sentences' words, from the export's columns."""
    doc_id, sentences, current = None, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#begin document"):
                doc_id = line.split("(", 1)[1].split(")", 1)[0]
            elif not line:
                if current:
                    sentences.append(current)
                current = []
            elif not line.startswith("#"):
                current.append(line.split("\t")[3])
    if current:
        sentences.append(current)
    return doc_id, sentences


def brackets(chains, n_words, sentence_ends):
    """The coreference column of each word: `(c)`, `(c`, `c)` joined by |.

    A span crossing a sentence end is clipped to its own sentence: the
    model's mention was over-long, and a bracket may not cross.
    """
    cols = [[] for _ in range(n_words)]
    for cluster_id, spans in enumerate(chains):
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
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    nlp = stanza.Pipeline(
        lang=lang,
        dir="/stanza",
        processors="tokenize,pos,coref",
        tokenize_pretokenized=True,
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    print(
        json.dumps({"stanza": stanza.__version__, "lang": lang, "processors": "tokenize,pos,coref"}),
        file=sys.stderr,
    )
    works = sorted(export_dir.glob("*.conll"))
    with open(out, "w", encoding="utf-8") as f:
        for path in works:
            doc_id, sentences = read_work(path)
            n_words = sum(len(s) for s in sentences)
            doc = nlp([list(s) for s in sentences])
            # Each chain's mentions as (start, end + 1) document word
            # indices; Stanza gives (sentence, start_word, end_word).
            chains = []
            offsets = []
            at_end = 0
            for s in sentences:
                offsets.append(at_end)
                at_end += len(s)
            for chain in doc.coref:
                spans = [(offsets[m.sentence] + m.start_word, offsets[m.sentence] + m.end_word) for m in chain.mentions]
                if spans:
                    chains.append(spans)
            # A span the model gives two chains is kept in the first: the
            # CoNLL bracket column holds one span once.
            seen_spans = set()
            deduped = []
            for spans in chains:
                kept = [s for s in spans if s not in seen_spans]
                seen_spans.update(kept)
                if kept:
                    deduped.append(kept)
            cols = brackets(deduped, n_words, [o + len(s) for o, s in zip(offsets, sentences)])
            f.write(f"#begin document ({doc_id}); part 0\n")
            at = 0
            for s, sent in enumerate(sentences):
                for i, word in enumerate(sent):
                    f.write("\t".join([doc_id, str(s), str(i), word] + ["_"] * 8 + [cols[at]]) + "\n")
                    at += 1
                f.write("\n")
            f.write("#end document\n")
            print(f"{doc_id}: {len(chains)} chains", file=sys.stderr)


if __name__ == "__main__":
    main()
