"""Reference ids and vectors for a bundle, from the upstream pipeline.

Arguments: the upstream model directory, the cases file the bundle tool
wrote, the safetensors file to write, and the JSON file that says what ran.
"""

import json
import sys

import numpy as np
import sentence_transformers
import torch
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer


def main(model_dir, cases_path, out_path, produced_by_path):
    with open(cases_path, encoding="utf-8") as f:
        spec = json.load(f)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(model_dir, device="cpu", local_files_only=True)
    model.max_seq_length = spec["max_seq"]
    texts = [c["prefix"] + c["text"] for c in spec["cases"]]

    # The ids the model sees, padded to the longest row with the pad id.
    features = model.preprocess(texts)
    ids = features["input_ids"].numpy().astype(np.int32)
    lengths = features["attention_mask"].numpy().sum(axis=1).astype(np.int32)

    # One text at a time, so no row is computed beside padding.
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=1, convert_to_numpy=True, precision="float32")
    save_file(
        {
            "ids": np.ascontiguousarray(ids),
            "lengths": np.ascontiguousarray(lengths),
            "embeddings": np.ascontiguousarray(embeddings.astype(np.float32)),
        },
        out_path,
    )
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "sentence-transformers",
                "tool_version": sentence_transformers.__version__,
                "args": [
                    "--device", "cpu",
                    "--dtype", "float32",
                    "--batch-size", "1",
                    "--max-seq", str(spec["max_seq"]),
                    "--torch", torch.__version__,
                ],
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
