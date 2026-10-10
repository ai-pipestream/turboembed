"""The text sets, from one model's golden texts.json: the golden
texts as they are; texts of 45 words (about 60 tokens); and 256-word
chunks. The words are the golden texts' own, in order, repeated to fill.
"""

import json
import sys


def chunks(words, size, count):
    out, i = [], 0
    while len(out) < count:
        out.append(" ".join(words[(i + k) % len(words)] for k in range(size)))
        i += size
    return out


def main(golden_texts, out_dir):
    with open(golden_texts, encoding="utf-8") as f:
        texts = json.load(f)
    words = [w for t in texts for w in t.split()]
    sets = {"golden": texts, "t60": chunks(words, 45, 2000), "w256": chunks(words, 256, 1000)}
    for name, s in sets.items():
        with open(f"{out_dir}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False)
        print(name, len(s), "texts,", sum(len(t) for t in s), "chars")


if __name__ == "__main__":
    main(*sys.argv[1:])
