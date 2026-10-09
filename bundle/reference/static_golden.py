"""Golden ids and vectors for a Model2Vec model, from Model2Vec itself.

Arguments: the model's directory (config.json, tokenizer.json and
model.safetensors, as its repository has them), a JSON-lines file of
{"text": ...} objects, and the directory to write to.

The texts are the file's, then EDGE_CASES. For each max_length, the
model's own (its config's, else StaticModel's default) and none, it
writes golden-<max_length>.safetensors: the ids StaticModel averages for
each text (I32 [n, width], padded with -1), their lengths (I32 [n]), and
StaticModel.encode's vectors in F32 ([n, dim]). Every text is encoded in
one call and alone, and the two must agree to the bit. texts.json holds
the texts in order, and versions.json what ran.

core/tests/static_parity.rs reads the directory against a bundle made
from the same model.
"""

import json
import os
import platform
import sys

import numpy as np
from model2vec import StaticModel
from safetensors.numpy import save_file

from static_reference import ids_of, vectors_of, versions

EDGE_CASES = [
    "",
    " ",
    "\t\n \r",
    "​",
    "\x00\x01\x7f",
    "☃",
    "\U0001F600\U0001F600",
    "Hello, World!",
    "HELLO world hello WORLD",
    "Café naïve RÉSUMÉ Ünïcödé",
    "é à combining marks",
    "ＡＢＣ　full width １２３",
    "東京は日本の首都です。",
    "东京是日本的首都。",
    "서울은 한국의 수도입니다.",
    "Москва — столица России.",
    "القاهرة هي عاصمة مصر.",
    "हिन्दी पाठ का एक उदाहरण",
    "[CLS] hello [SEP] [UNK] [PAD] [MASK] [cls] [pad]",
    "<s> </s> <unk> <pad> <mask>",
    "x [PAD] y [pad] z",
    "café😀x is one word",
    "a" * 101,
    "a" * 100,
    "def f(x):\n    return x ** 2  # square\n",
    "SELECT id, name FROM users WHERE id = 42;",
    "https://example.com/a/b?c=d&e=f#g",
    "word " * 3000,
    "\U0001F600 " * 600 + "word " * 600,
    "☃ " * 20 + "word " * 600,
    "international " * 600,
    "!!!???...,,,;;;",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\r\nand  double  spaces",
]


def main(model_dir, texts_path, out_dir):
    with open(texts_path, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    texts += EDGE_CASES
    model = StaticModel.from_pretrained(model_dir, force_download=False)
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for max_length in [model.max_length, None]:
        name = f"golden-{max_length if max_length is not None else 'none'}.safetensors"
        ids = [ids_of(model, t, max_length) for t in texts]
        vectors = vectors_of(model, texts, max_length, 1024)
        width = max([len(r) for r in ids] + [1])
        padded = np.full((len(ids), width), -1, dtype=np.int32)
        for i, r in enumerate(ids):
            padded[i, : len(r)] = r
        save_file(
            {
                "ids": padded,
                "lengths": np.array([len(r) for r in ids], dtype=np.int32),
                "embeddings": np.ascontiguousarray(vectors),
            },
            os.path.join(out_dir, name),
        )
        files[name] = max_length
    with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "versions.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "model2vec StaticModel.from_pretrained, tokenize, encode",
                "versions": f"{versions()}, python {platform.python_version()}",
                "model_dir": model_dir,
                "normalize": model.normalize,
                "median_token_length": model.median_token_length,
                "unk_token_id": model.unk_token_id,
                "embedding_dtype": str(model.embedding.dtype),
                "files": files,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
