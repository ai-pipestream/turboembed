"""Reference ids and vectors for a bundle, from the upstream pipeline.

Arguments: the upstream model directory, the cases file the bundle tool
wrote, the safetensors file to write, and the JSON file that says what ran.
"""

import json
import sys

import numpy as np
import sentence_transformers
import torch
import transformers
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer


def main(model_dir, cases_path, out_path, produced_by_path):
    with open(cases_path, encoding="utf-8") as f:
        spec = json.load(f)
    torch.use_deterministic_algorithms(True)
    if spec.get("task") == "TASK_RERANK":
        from rerank import run
        run(model_dir, spec, out_path)
        with open(produced_by_path, "w", encoding="utf-8") as f:
            json.dump({"tool": "transformers", "tool_version": f"{transformers.__version__} (torch {torch.__version__}, cpu, float32)", "args": sys.argv[1:]}, f)
        return
    if spec.get("task") == "TASK_EXTRACTIVE_QA":
        from qa_reference import tensors
        save_file(tensors(model_dir, spec), out_path)
        with open(produced_by_path, "w", encoding="utf-8") as f:
            json.dump({"tool": "transformers", "tool_version": transformers.__version__,
                       "args": sys.argv[1:]}, f)
        return
    model = SentenceTransformer(model_dir, device="cpu", local_files_only=True)
    model.max_seq_length = spec["max_seq"]
    texts = [c["prefix"] + c["text"] for c in spec["cases"]]

    # The ids the model sees, padded to the longest row with the pad id.
    features = model.preprocess(texts)
    ids = features["input_ids"].numpy().astype(np.int32)
    lengths = features["attention_mask"].numpy().sum(axis=1).astype(np.int32)

    # One text at a time, so no row is computed beside padding. Then all of
    # them in batches of max_batch, which must give the same rows: that is
    # what the manifest's max_batch says was checked.
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=1, convert_to_numpy=True, precision="float32")
        batched = model.encode(texts, batch_size=spec["max_batch"], convert_to_numpy=True, precision="float32")
    worst = float(np.abs(embeddings - batched).max())
    if worst > 1e-4:
        sys.exit(f"batch {spec['max_batch']} differs from batch 1 by {worst}")
    tensors = {
        "ids": np.ascontiguousarray(ids),
        "lengths": np.ascontiguousarray(lengths),
        "embeddings": np.ascontiguousarray(embeddings.astype(np.float32)),
    }
    if "classifier" in spec:
        kind = spec["classifier"]["kind"]
        labels = spec["classifier"]["labels"]
        if kind == "CLASSIFIER_TOKEN":
            cls_model = transformers.AutoModelForTokenClassification.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float32
            )
        elif kind == "CLASSIFIER_SEQUENCE":
            cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float32
            )
        else:
            sys.exit(f"unknown classifier kind {kind!r}")
        cls_model.eval()
        id2label = getattr(cls_model.config, "id2label", None)
        if id2label is not None:
            got = [id2label.get(i, id2label.get(str(i))) for i in range(len(labels))]
            if got != labels:
                sys.exit(f"classifier labels {labels} do not match config.json id2label {got}")
        rows = []
        with torch.no_grad():
            for i in range(len(texts)):
                length = int(lengths[i])
                row_ids = torch.from_numpy(ids[i : i + 1, :length]).to(torch.long)
                mask = torch.ones_like(row_ids)
                out = cls_model(input_ids=row_ids, attention_mask=mask).logits.numpy().astype(np.float32)
                if kind == "CLASSIFIER_TOKEN":
                    rows.append(out[0, :length, :])
                else:
                    rows.append(out)
        logits = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(labels)), dtype=np.float32)
        tensors["logits"] = np.ascontiguousarray(logits)
    save_file(tensors, out_path)
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "sentence-transformers",
                "tool_version": (
                    f"{sentence_transformers.__version__} "
                    f"(transformers {transformers.__version__}, torch {torch.__version__}, cpu, float32)"
                ),
                "args": sys.argv[1:],
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
