"""Write Stanza entity spans for raw documents.

Usage: span_predict.py DOCS.jsonl OUT.jsonl

DOCS.jsonl holds one {"id", "text"} object per line, as the
`rule_entities_gate` example of turbo-query exports it. Stanza's
default English tokenizer and OntoNotes NER read each text whole.
OUT.jsonl gets one {"id", "entities": [{"start", "end", "label"}]}
object per line, with UTF-8 byte offsets into the text. DATE, TIME,
MONEY and PERCENT map to date, time, money and percent; every other
type is dropped. Models are baked into the image; nothing downloads
at run time.
"""

import json
import sys

import stanza
from stanza import DownloadMethod

MAP = {"DATE": "date", "TIME": "time", "MONEY": "money", "PERCENT": "percent"}


def byte_offsets(text):
    out, at = [], 0
    for ch in text:
        out.append(at)
        at += len(ch.encode("utf-8"))
    out.append(at)
    return out


def main():
    docs, out = sys.argv[1], sys.argv[2]
    nlp = stanza.Pipeline(
        lang="en",
        dir="/stanza",
        processors="tokenize,ner",
        download_method=DownloadMethod.NONE,
        use_gpu=False,
        verbose=False,
    )
    print(json.dumps({"stanza": stanza.__version__, "processors": "tokenize,ner", "label_map": MAP}), file=sys.stderr)
    with open(docs, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d["text"]
            offsets = byte_offsets(text)
            ents = []
            for ent in nlp(text).ents:
                label = MAP.get(ent.type)
                if label is not None:
                    ents.append({"start": offsets[ent.start_char], "end": offsets[ent.end_char], "label": label})
            w.write(json.dumps({"id": d["id"], "entities": ents}) + "\n")


if __name__ == "__main__":
    main()
