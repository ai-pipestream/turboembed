"""Write Stanza BIO entity labels for gold-tokenized sentences.

Usage: ner_predict.py [--with-tokens] [--gpu] GOLD.tsv OUT.tsv [LABELSET]

The gold file is `token<TAB>label` lines with blank lines between
sentences (the bench's entity gold form). Stanza runs with
pretokenized input, as WNUT-17 and MultiCoNER are conventionally
run. Models are baked into the image; nothing downloads at run time.
With `--with-tokens`, output lines are `token<TAB>label` for
`turbo-train-head peer --column bio`; `--gpu` enables GPU execution.

Label maps, chosen by LABELSET (default wnut). wnut (Stanza
OntoNotes to the WNUT-17 label set): PERSON to person, ORG to
corporation, GPE, LOC and FAC to location, PRODUCT to product,
WORK_OF_ART to creative-work, NORP to group. multiconer (to the
MultiCoNER v2 coarse groups): PERSON to PER, ORG and NORP to GRP,
GPE, LOC and FAC to LOC, PRODUCT to PROD, WORK_OF_ART to CW.
nemotron-pii (to shared Nemotron-PII types, #1888): ORG to
company_name, DATE to date, TIME to time, LANGUAGE to language
(nemotron-pii-raw keeps those four OntoNotes names for explicit
`turbo-train-head peer --map` provenance).
Every other type is dropped. The map is part of the record's
conditioning.
"""

import json
import sys

import stanza
from stanza import DownloadMethod

MAPS = {
    "wnut": {
        "PERSON": "person",
        "ORG": "corporation",
        "GPE": "location",
        "LOC": "location",
        "FAC": "location",
        "PRODUCT": "product",
        "WORK_OF_ART": "creative-work",
        "NORP": "group",
    },
    "multiconer": {
        "PERSON": "PER",
        "ORG": "GRP",
        "NORP": "GRP",
        "GPE": "LOC",
        "LOC": "LOC",
        "FAC": "LOC",
        "PRODUCT": "PROD",
        "WORK_OF_ART": "CW",
    },
    "nemotron-pii": {
        "ORG": "company_name",
        "DATE": "date",
        "TIME": "time",
        "LANGUAGE": "language",
    },
    "nemotron-pii-raw": {
        "ORG": "ORG",
        "DATE": "DATE",
        "TIME": "TIME",
        "LANGUAGE": "LANGUAGE",
    },
}


def gold_sentences(path):
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") and "\t" not in line:
                continue
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            current.append(line.split("\t")[0])
    if current:
        sentences.append(current)
    return sentences


def main():
    args = [a for a in sys.argv[1:] if a not in ("--with-tokens", "--gpu")]
    with_tokens = "--with-tokens" in sys.argv[1:]
    use_gpu = "--gpu" in sys.argv[1:]
    gold, out = args[0], args[1]
    labelset = args[2] if len(args) > 2 else "wnut"
    label_map = MAPS[labelset]
    nlp = stanza.Pipeline(
        lang="en",
        dir="/stanza",
        processors="tokenize,ner",
        tokenize_pretokenized=True,
        download_method=DownloadMethod.NONE,
        use_gpu=use_gpu,
        verbose=False,
    )
    print(json.dumps({
        "stanza": stanza.__version__,
        "lang": "en",
        "processors": "tokenize,ner (pretokenized)",
        "label_map": label_map,
        "label_set": labelset,
        "with_tokens": with_tokens,
        "gpu": use_gpu,
    }), file=sys.stderr)
    sentences = gold_sentences(gold)
    # One pipeline call per chunk of sentences (a single pretokenized
    # document), so the NER processor batches within the chunk; the
    # labels per sentence are the same as one call per sentence.
    chunk = 512
    with open(out, "w", encoding="utf-8") as f:
        done = 0
        for base in range(0, len(sentences), chunk):
            batch = sentences[base:base + chunk]
            doc = nlp(batch)
            if len(doc.sentences) != len(batch):
                raise SystemExit(
                    f"stanza returned {len(doc.sentences)} sentences for {len(batch)}"
                )
            for offset, (tokens, sent) in enumerate(zip(batch, doc.sentences)):
                if base + offset:
                    f.write("\n")
                labels = ["O"] * len(tokens)
                for ent in sent.entities:
                    kind = label_map.get(ent.type)
                    if kind is None or not ent.tokens:
                        continue
                    start = ent.tokens[0].id[-1] - 1 if isinstance(ent.tokens[0].id, tuple) else ent.tokens[0].id - 1
                    labels[start] = f"B-{kind}"
                    for token in ent.tokens[1:]:
                        index = token.id[-1] - 1 if isinstance(token.id, tuple) else token.id - 1
                        labels[index] = f"I-{kind}"
                if with_tokens:
                    f.write("\n".join(f"{t}\t{l}" for t, l in zip(tokens, labels)) + "\n")
                else:
                    f.write("\n".join(labels) + "\n")
            done += len(batch)
            if done % 5000 < chunk or done == len(sentences):
                print(f"sentences {done}/{len(sentences)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
