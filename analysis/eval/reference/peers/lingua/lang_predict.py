"""Write lingua language predictions for a language gold file.

Usage: lang_predict.py GOLD.tsv OUT.txt

The gold file is `code<TAB>text` lines (the bench's --lang-gold
form); the gold codes are ignored here. One line is written per
gold line: the predicted ISO 639-3 code and lingua's confidence,
tab-separated (the bench's --lang-pred reader takes the first
field). A text lingua declines to name is `und`.

The detector is built from all 75 languages lingua 2.2.0 ships, as
shipped, in accurate mode with models preloaded: the peer is not
restricted to the gold label set. Its ISO 639-3 codes already match
the gold codes except Arabic: lingua's Arabic is Modern Standard
Arabic under the macrolanguage code ara, and the gold code for
FLORES+ arb_Arab is arb, so ara is written as arb. That one-line map
is part of the record's conditioning.
"""

import sys

from lingua import LanguageDetectorBuilder

MAP = {"ara": "arb"}


def main():
    gold, out = sys.argv[1], sys.argv[2]
    texts = []
    with open(gold, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            texts.append(line.split("\t", 1)[1])
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    lines = []
    for text in texts:
        values = detector.compute_language_confidence_values(text)
        if not values:
            lines.append("und\t0\n")
            continue
        top = max(values, key=lambda v: v.value)
        code = top.language.iso_code_639_3.name.lower()
        lines.append(f"{MAP.get(code, code)}\t{top.value:.6f}\n")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"{len(lines)} predictions", file=sys.stderr)


if __name__ == "__main__":
    main()
