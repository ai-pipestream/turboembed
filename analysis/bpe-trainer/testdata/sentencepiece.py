"""What Google's sentencepiece library (the Python package 0.2.2) trains
on small corpora with model_type=bpe and the identity normalization,
for tests/sentencepiece.rs to compare the trainer with. A yardstick for
the tests, never part of training.

    python3 sentencepiece.py

writes sentencepiece.jsonl and sentencepiece-random.jsonl beside it.

sentencepiece.jsonl has one line per run: the corpus's name, vocab_size, character_coverage,
and either the pieces of the model, each [piece, score, type] with the
type as ModelProto numbers it, or the library's error. Where the size is
too large, the error names the largest the corpus allows, and a run at
that size follows. The corpora are
read from the repository, one sentence per line, with "\\r" and "\\n"
in a text made spaces; tests/sentencepiece.rs reads them the same way:

  tokenizer-texts   testdata/tokenizer-texts.jsonl, the "text" of each line
  tiny-bpe          testdata/tiny-bpe/corpus.jsonl, one string per line
  bpe-inputs        the lines of analysis/goldens/data/inputs/bpe/*.txt,
                    the files in name order
  edge              edge.jsonl beside this script, one string per line:
                    the corners of the library's reading of a corpus (⁇
                    and ▅, U+2581 in the text, meta pieces, tabs, NUL,
                    lines of 4192 and 4193 bytes, scripts, long words)

sentencepiece-random.jsonl has RANDOM_RUNS small corpora of the letters
"ab" or "abc", drawn with random.Random(RANDOM_SEED): on each line the
corpus's lines and the pieces of the largest vocabulary the library
allows for it, with character_coverage 1.0, so that every model runs
out of pairs and shows what the library does after that.
"""

import glob
import io
import json
import os
import random

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

VOCAB_SIZES = [120, 600, 1000, 100000]
COVERAGES = [1.0, 0.9995, 0.98]
RANDOM_SEED = 194
RANDOM_RUNS = 300


def flat(t):
    return t.replace("\r", " ").replace("\n", " ")


def corpora():
    with open(os.path.join(REPO, "testdata/tokenizer-texts.jsonl"), encoding="utf-8") as f:
        texts = [flat(json.loads(l)["text"]) for l in f]
    with open(os.path.join(REPO, "testdata/tiny-bpe/corpus.jsonl"), encoding="utf-8") as f:
        tiny = [flat(json.loads(l)) for l in f]
    inputs = []
    for p in sorted(glob.glob(os.path.join(REPO, "analysis/goldens/data/inputs/bpe/*.txt"))):
        with open(p, encoding="utf-8") as f:
            inputs.extend(f.read().split("\n"))
    with open(os.path.join(HERE, "edge.jsonl"), encoding="utf-8") as f:
        edge = [json.loads(l) for l in f]
    return [("tokenizer-texts", texts), ("tiny-bpe", tiny), ("bpe-inputs", inputs), ("edge", edge)]


def train(lines, size, coverage):
    """The pieces, each [piece, score, type], or the library's error."""
    out = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines), model_writer=out, model_type="bpe",
        vocab_size=size, character_coverage=coverage,
        normalization_rule_name="identity", minloglevel=2)
    m = pb.ModelProto()
    m.ParseFromString(out.getvalue())
    return [[p.piece, p.score, p.type] for p in m.pieces]


def largest(error):
    """The size a "too high" error names, or None."""
    if "too high" not in error:
        return None
    return int(error.rstrip(".").rsplit(" ", 1)[1])


def write(name, runs):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, ensure_ascii=False, separators=(",", ":")) + "\n")


def fixed_runs():
    for name, lines in corpora():
        for coverage in COVERAGES:
            sizes = list(VOCAB_SIZES)
            done = set()
            while sizes:
                size = sizes.pop(0)
                if size in done:
                    continue
                done.add(size)
                run = {"corpus": name, "vocab_size": size, "character_coverage": coverage}
                try:
                    run["pieces"] = train(lines, size, coverage)
                except RuntimeError as e:
                    run["error"] = str(e)
                    if largest(run["error"]):
                        sizes.insert(0, largest(run["error"]))
                yield run


def random_runs():
    rng = random.Random(RANDOM_SEED)
    for _ in range(RANDOM_RUNS):
        letters = rng.choice(["ab", "abc"])
        lines = [
            " ".join("".join(rng.choice(letters) for _ in range(rng.randint(1, 8)))
                     for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 4))
        ]
        try:
            pieces = train(lines, 10000, 1.0)
        except RuntimeError as e:
            pieces = train(lines, largest(str(e)), 1.0)
        yield {"lines": lines, "pieces": [p[0] for p in pieces]}


def main():
    assert spm.__version__ == "0.2.2", spm.__version__
    write("sentencepiece.jsonl", fixed_runs())
    write("sentencepiece-random.jsonl", random_runs())


if __name__ == "__main__":
    main()
