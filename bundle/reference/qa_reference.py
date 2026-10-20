"""Joint-input reference tensors from Hugging Face question-answering models."""
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def tensors(model_dir, spec):
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    q = spec["qa"]
    limit = q["max_seq"]
    query_limit = q.get("max_query_tokens", 64)
    overlap = q.get("doc_stride", 128)
    rows, types, outputs = [], [], []
    for case in spec["cases"]:
        question = tok.encode(case["text"], add_special_tokens=False)[:query_limit]
        context = tok.encode(case["context"], add_special_tokens=False)
        room = limit - len(question) - (4 if model.config.model_type == "roberta" else 3)
        if not 0 <= overlap < room:
            raise ValueError("doc_stride must be below context capacity")
        start = 0
        while True:
            end = min(start + room, len(context))
            if model.config.model_type == "bert":
                ids = [tok.cls_token_id] + question + [tok.sep_token_id] + context[start:end] + [tok.sep_token_id]
                segment = [0] * (len(question) + 2) + [1] * (end - start + 1)
            elif model.config.model_type == "roberta":
                ids = [tok.bos_token_id] + question + [tok.eos_token_id] * 2 + context[start:end] + [tok.eos_token_id]
                segment = [0] * len(ids)
            else:
                raise ValueError("QA supports BERT and RoBERTa")
            with torch.no_grad():
                kw = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
                if model.config.model_type == "bert":
                    kw["token_type_ids"] = torch.tensor([segment])
                out = model(**kw)
            outputs.append(torch.stack([out.start_logits[0], out.end_logits[0]], dim=-1).numpy())
            rows.append(ids)
            types.append(segment)
            if end == len(context):
                break
            start += room - overlap
    width = max(map(len, rows))
    ids = np.full((len(rows), width), tok.pad_token_id, dtype=np.int32)
    segments = np.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = row
        segments[i, :len(row)] = types[i]
    return {
        "ids": ids, "types": segments,
        "lengths": np.array([len(r) for r in rows], dtype=np.int32),
        "logits": np.concatenate(outputs).astype(np.float32),
    }
