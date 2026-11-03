"""Write GlotLID predictions for a language gold file.

Usage: lang_predict.py MODEL.bin GOLD.tsv OUT.txt

The gold file is `code<TAB>text` lines. One prediction is written per
line: an ISO 639-3 code and confidence, tab-separated. GlotLID v3's
script suffix is removed and its individual-language codes are mapped to
the macrolanguage codes used by the FLORES+ converter. The peer is not
restricted to the gold label set.
"""

import sys

import fasttext

MAP = {
    "cmn": "zho",
    "pes": "fas",
    "ekk": "est",
    "lvs": "lav",
    "khk": "mon",
    "zsm": "msa",
    "swh": "swa",
    "azj": "aze",
    "als": "sqi",
}


def gold_code(label):
    code = label.removeprefix("__label__").split("_", 1)[0].lower()
    return MAP.get(code, code)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    model_path, gold, out = sys.argv[1:4]
    model = fasttext.load_model(model_path)
    lines = []
    with open(gold, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text = line.split("\t", 1)[1]
            labels, scores = model.predict(text, k=1)
            if not labels:
                lines.append("und\t0\n")
            else:
                lines.append(f"{gold_code(labels[0])}\t{scores[0]:.6f}\n")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"{len(lines)} predictions", file=sys.stderr)


if __name__ == "__main__":
    main()
