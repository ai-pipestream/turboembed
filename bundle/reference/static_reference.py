"""Reference ids and vectors for a static bundle, from Model2Vec.

Arguments: the directory holding the table and the tokenizer (the staged
bundle for a distilled model, the upstream files for a Model2Vec one), the
cases file the bundle tool wrote, the safetensors file to write, and the
JSON file that says what ran.

Model2Vec's StaticModel is built from the table as stored (embeddings,
and weights and mapping when the file has them) and the tokenizer file,
with the bundle's max_length and normalization. A case's ids are what
StaticModel averages for it at max_length: the text cut to max_length
times the median token length in characters, encoded with no special
tokens and cut to max_length tokens, the unknown token dropped. Its
vector is StaticModel.encode's, in F32. Where the directory is a
Model2Vec model (it has config.json), StaticModel.from_pretrained must
give the same vectors bit for bit, and so must one call over every case.
"""

import json
import os
import sys
from importlib import metadata

import numpy as np
import tokenizers
from model2vec import StaticModel
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer


def model2vec_version():
    """The installed Model2Vec, with the commit it was installed from."""
    version = metadata.version("model2vec")
    try:
        direct = json.loads(metadata.distribution("model2vec").read_text("direct_url.json") or "{}")
    except (FileNotFoundError, ValueError):
        direct = {}
    url = direct.get("url", "")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit:
        url = f"{url}@{commit}"
    return f"{version} ({url})" if url else version


def versions():
    return (
        f"model2vec {model2vec_version()}, tokenizers {tokenizers.__version__}, numpy {np.__version__}"
    )


def build(model_dir, weights, tokenizer, normalize, max_length):
    """StaticModel from the stored table and the tokenizer file."""
    tensors = load_file(os.path.join(model_dir, weights))
    table = tensors["embeddings"] if "embeddings" in tensors else tensors["embedding.weight"]
    return StaticModel(
        vectors=table,
        tokenizer=Tokenizer.from_file(os.path.join(model_dir, tokenizer)),
        normalize=normalize,
        weights=tensors.get("weights"),
        token_mapping=tensors.get("mapping"),
        max_length=max_length,
    )


def ids_of(model, text, max_length):
    """The ids StaticModel averages for `text` at `max_length`."""
    model._set_max_length_in_tokenizer(max_length)
    try:
        cut = text if max_length is None else text[: max_length * model.median_token_length]
        return model.tokenize([cut])[0]
    finally:
        model._set_max_length_in_tokenizer(model.max_length)


def vectors_of(model, texts, max_length, batch_size):
    """StaticModel.encode's vectors, in F32, checked against one text at a
    time: Model2Vec's vectors do not depend on the batch."""
    batched = model.encode(texts, max_length=max_length, batch_size=batch_size, use_multiprocessing=False)
    batched = np.asarray(batched)
    for i, t in enumerate(texts):
        one = np.asarray(model.encode([t], max_length=max_length, use_multiprocessing=False))[0]
        if not np.array_equal(one, batched[i]):
            sys.exit(f"text {i}: encode alone and in a batch differ")
    return batched.astype(np.float32)


def rows(ids, pad_id):
    """The ids as [n, width] padded with pad_id, and their lengths."""
    width = max([len(r) for r in ids] + [1])
    out = np.full((len(ids), width), pad_id, dtype=np.int32)
    for i, r in enumerate(ids):
        out[i, : len(r)] = r
    return out, np.array([len(r) for r in ids], dtype=np.int32)


def main(model_dir, cases_path, out_path, produced_by_path):
    with open(cases_path, encoding="utf-8") as f:
        spec = json.load(f)
    max_length = spec["max_length"]
    model = build(model_dir, spec["weights"], spec["tokenizer"], spec["normalize"], max_length)
    texts = [c["prefix"] + c["text"] for c in spec["cases"]]
    vectors = vectors_of(model, texts, max_length, spec["max_batch"])
    if os.path.exists(os.path.join(model_dir, "config.json")):
        upstream = StaticModel.from_pretrained(model_dir, force_download=False)
        if upstream.normalize != spec["normalize"] or upstream.max_length != max_length:
            sys.exit(
                f"the model's config says normalize {upstream.normalize} and max_length {upstream.max_length}, "
                f"the bundle {spec['normalize']} and {max_length}"
            )
        theirs = np.asarray(upstream.encode(texts, use_multiprocessing=False)).astype(np.float32)
        if not np.array_equal(theirs, vectors):
            sys.exit("StaticModel.from_pretrained gives other vectors than the table and tokenizer")
    ids, lengths = rows([ids_of(model, t, max_length) for t in texts], spec["pad_id"])
    save_file(
        {
            "ids": np.ascontiguousarray(ids),
            "lengths": np.ascontiguousarray(lengths),
            "embeddings": np.ascontiguousarray(vectors),
        },
        out_path,
    )
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "model2vec",
                "tool_version": f"{versions()}, StaticModel.encode, table as stored",
                "args": sys.argv[1:],
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
