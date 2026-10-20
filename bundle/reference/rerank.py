"""Cross-encoder pair references from the upstream classifier on CPU."""
import json
import time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.numpy import save_file


def run(model_dir, spec, out_path):
    torch.set_num_threads(spec.get("threads", 1))
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.backend_tokenizer.no_padding()
    tok.backend_tokenizer.no_truncation()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    windows, owners, first = [], [], []
    for i, case in enumerate(spec["cases"]):
        q = tok.backend_tokenizer.encode(case["text"], add_special_tokens=False)
        q.truncate(spec["query_tokens"])
        p = tok.backend_tokenizer.encode(case["passage"], add_special_tokens=False)
        capacity = spec["max_seq"] - len(q.ids) - tok.num_special_tokens_to_add(pair=True)
        if capacity < 1:
            raise ValueError("query budget leaves no passage token")
        p.truncate(capacity)
        first.append(len(windows))
        for piece in [p] + p.overflowing:
            encoded = tok.backend_tokenizer.post_process(q, piece, add_special_tokens=True)
            windows.append({"input_ids": encoded.ids, "token_type_ids": encoded.type_ids})
            owners.append(i)
    lengths = np.array([len(windows[i]["input_ids"]) for i in first], dtype=np.int32)
    width = int(max(lengths))
    ids = np.full((len(first), width), tok.pad_token_id, dtype=np.int32)
    types = np.zeros_like(ids)
    for row, i in enumerate(first):
        n = lengths[row]
        ids[row, :n] = windows[i]["input_ids"]
        types[row, :n] = windows[i]["token_type_ids"]

    def score(batch):
        scores = np.full((len(first), 1), -np.inf, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(windows), batch):
                inputs = tok.pad(windows[start:start+batch], return_tensors="pt")
                if model.config.model_type != "bert":
                    inputs.pop("token_type_ids", None)
                logits = model(**inputs).logits.numpy()
                if logits.shape[1] != 1:
                    raise ValueError("rerank classifier must have one logit")
                for j, logit in enumerate(logits):
                    scores[owners[start+j]] = np.maximum(scores[owners[start+j]], logit)
        return scores
    scores = score(1)
    batched = score(spec["max_batch"])
    if not np.isfinite(scores).all() or np.max(np.abs(scores-batched)) > 1e-4:
        raise ValueError("non-finite scores or batch score drift")
    save_file({"ids": ids, "types": types, "lengths": lengths, "scores": scores}, out_path)
