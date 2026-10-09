"""Reference ids and vectors for a static bundle, from Model2Vec.

Arguments: the bundle directory the tool staged, the cases file it wrote,
the safetensors file to write, and the JSON file that says what ran.

The table and the tokenizer are the bundle's own. Model2Vec's StaticModel
holds the table in F32 and tokenizes with the bundle's tokenizer file and
no special tokens, as it does for every model it distils. The ids are cut
to max_seq on the right, as the bundle truncates, and then the unknown
token is left out of the mean, as StaticModel leaves it out; a token whose
weight in the bundle is 0 is left out the same way, and the script refuses
a case that holds one other than the unknown token, since StaticModel has
no such weight.
"""

import json
import sys

import model2vec
import numpy as np
import tokenizers
from model2vec import StaticModel
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer


def main(bundle_dir, cases_path, out_path, produced_by_path):
    with open(cases_path, encoding="utf-8") as f:
        spec = json.load(f)
    tensors = load_file(f"{bundle_dir}/{spec['weights']}")
    table = tensors["embeddings"].astype(np.float32)
    weights = tensors["weights"].astype(np.float32)
    tok = Tokenizer.from_file(f"{bundle_dir}/{spec['tokenizer']}")
    tok.no_truncation()
    tok.no_padding()
    unk = spec["unk_id"]
    model = StaticModel(vectors=table, tokenizer=tok, normalize=True)
    if model.unk_token_id != unk:
        sys.exit(f"StaticModel reads unk id {model.unk_token_id}, the bundle says {unk}")

    texts = [c["prefix"] + c["text"] for c in spec["cases"]]
    rows = []
    vectors = []
    for i, text in enumerate(texts):
        ids = tok.encode(text, add_special_tokens=False).ids[: spec["max_seq"]]
        rows.append(ids)
        dropped = [t for t in ids if weights[t] == 0 and t != unk]
        if dropped:
            sys.exit(f"case {i}: tokens {dropped} have weight 0, which StaticModel has no way to say")
        live = [t for t in ids if t != unk]
        if not live:
            sys.exit(f"case {i}: no token but the unknown one, so no vector")
        # StaticModel's own mean of the rows and normalization, in F32.
        vec = model._encode_helper(live).mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-32)
        vectors.append(vec.astype(np.float32))
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), spec["pad_id"], dtype=np.int32)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
    lengths = np.array([len(r) for r in rows], dtype=np.int32)

    # StaticModel.encode on the same texts, in batches of max_batch, must
    # give the same rows wherever the bundle's cut leaves a text whole.
    batched = model.encode(texts, max_length=None, batch_size=spec["max_batch"], use_multiprocessing=False)
    whole = [i for i, t in enumerate(texts) if len(tok.encode(t, add_special_tokens=False).ids) <= spec["max_seq"]]
    worst = float(np.abs(np.stack(vectors)[whole] - batched[whole]).max())
    if worst > 1e-4:
        sys.exit(f"StaticModel.encode differs from the rows by {worst}")
    save_file(
        {
            "ids": np.ascontiguousarray(ids),
            "lengths": np.ascontiguousarray(lengths),
            "embeddings": np.ascontiguousarray(np.stack(vectors)),
        },
        out_path,
    )
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "model2vec",
                "tool_version": (
                    f"{model2vec.__version__} (tokenizers {tokenizers.__version__}, numpy {np.__version__}, float32)"
                ),
                "args": sys.argv[1:],
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
