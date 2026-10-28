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
from safetensors import safe_open
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer

from cut import unexplained


def checkpoint_keys(path):
    """The tensor names of the checkpoint transformers loads from `path`."""
    import glob
    import os

    names = set()
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt") as f:
            names.update(f.keys())
    return names


def upstream_layers(config):
    """The encoder layers the upstream config declares."""
    for name in ("num_hidden_layers", "n_layer"):
        if isinstance(getattr(config, name, None), int):
            return name, getattr(config, name)
    sys.exit(f"{type(config).__name__}: no layer count (num_hidden_layers or n_layer) to cut")


def check_loaded(module, cut=None):
    """Load the encoder of a sentence-transformers module again with
    transformers' loading report, and refuse a model that left any of its
    own tensors at their random initial values: a missing key, a key of
    the wrong shape, or a checkpoint tensor the model did not take (a
    renamed or fused tensor). The one exception is a task head beside the
    encoder (ModernBERT's checkpoint is the masked-LM model; its `head.`
    and `decoder.` tensors are not the encoder's): a checkpoint tensor
    outside the encoder's prefix may go unused. With `cut`, the layer
    count a cut model keeps, the tensors of the layers past it may go
    unused too, and no other. The model checked is then
    held equal, tensor for tensor, to the one that runs."""
    model = module.auto_model
    path = model.config._name_or_path
    again, info = type(model).from_pretrained(
        path, config=model.config, local_files_only=True, output_loading_info=True
    )
    keys = checkpoint_keys(path)
    bad = {
        "missing": sorted(info["missing_keys"]),
        "mismatched": sorted(str(k) for k in info["mismatched_keys"]),
        "unexpected": unexplained(info["unexpected_keys"], keys, model.base_model_prefix + ".", cut),
        "errors": [str(e) for e in info.get("error_msgs", [])],
    }
    if any(bad.values()):
        sys.exit(f"{type(model).__name__} did not load cleanly from {path}: {bad}")
    ran, loaded = model.state_dict(), again.state_dict()
    if ran.keys() != loaded.keys() or any(not torch.equal(ran[k], loaded[k]) for k in ran):
        sys.exit(f"{type(model).__name__}: the model that runs differs from the one loaded from {path}")
    return sorted(set(info["unexpected_keys"]))


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
    # The reference is fp32 on CPU (the Dockerfile's contract), so the
    # weights load as float32 whatever dtype the checkpoint stores;
    # a BF16 checkpoint otherwise computes in BF16, whose rows are
    # batch-size dependent beyond the check below.
    # A bundle that runs fewer layers than the upstream model has is cut
    # there: the reference runs the same first layers and the same final
    # norm, and the checkpoint's later layers go unused.
    config_kwargs, cut = {}, None
    if "layers" in spec:
        name, upstream = upstream_layers(transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True))
        if spec["layers"] > upstream:
            sys.exit(f"the bundle runs {spec['layers']} layers; the upstream model has {upstream}")
        if spec["layers"] < upstream:
            if "classifier" in spec:
                sys.exit("a classifier is trained on every layer; a cut model has no reference logits")
            config_kwargs, cut = {name: spec["layers"]}, spec["layers"]
    model = SentenceTransformer(
        model_dir,
        device="cpu",
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs=config_kwargs,
    )
    for module in model:
        if hasattr(module, "auto_model"):
            unused = check_loaded(module, cut)
            if unused:
                print(f"unused task-head tensors: {unused}", file=sys.stderr)
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
                model_dir, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True
            )
        elif kind == "CLASSIFIER_SEQUENCE":
            cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True
            )
        else:
            sys.exit(f"unknown classifier kind {kind!r}")
        cls_model, info = cls_model
        if info["missing_keys"] or info["mismatched_keys"]:
            sys.exit(f"classifier did not load cleanly: missing {info['missing_keys']}, mismatched {info['mismatched_keys']}")
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
