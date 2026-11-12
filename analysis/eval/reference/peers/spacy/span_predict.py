"""Write spaCy entity spans for raw documents.

Usage: span_predict.py DOCS.jsonl OUT.jsonl [MODEL]

DOCS.jsonl holds one {"id", "text"} object per line, as the
`rule_entities_gate` example of turbo-query exports it. spaCy reads
each text whole and tokenizes it itself. OUT.jsonl gets one
{"id", "entities": [{"start", "end", "label"}]} object per line, with
UTF-8 byte offsets into the text. OntoNotes DATE, TIME, MONEY and
PERCENT map to date, time, money and percent; every other type is
dropped. Version facts go to stderr as one JSON line.
"""

import importlib.metadata
import json
import sys

import spacy

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
    model = sys.argv[3] if len(sys.argv) > 3 else "en_core_web_trf"
    nlp = spacy.load(model)
    facts = {
        "spacy": spacy.__version__,
        "model": nlp.meta.get("name"),
        "model_version": nlp.meta.get("version"),
        "pipeline": nlp.pipe_names,
        "label_map": MAP,
    }
    try:
        facts["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    print(json.dumps(facts), file=sys.stderr)
    with open(docs, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d["text"]
            nlp.max_length = max(nlp.max_length, len(text) + 1)
            offsets = byte_offsets(text)
            ents = []
            for ent in nlp(text).ents:
                label = MAP.get(ent.label_)
                if label is not None:
                    ents.append({"start": offsets[ent.start_char], "end": offsets[ent.end_char], "label": label})
            w.write(json.dumps({"id": d["id"], "entities": ents}) + "\n")


if __name__ == "__main__":
    main()
