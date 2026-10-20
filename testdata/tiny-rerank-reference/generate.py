"""Generate actual tiny BERT cross-encoder scores in the pinned reference image.

Mount the repository at /repo and the output directory at /output, then run
python /repo/testdata/tiny-rerank-reference/generate.py.
"""
import json
from pathlib import Path
import sys
import tempfile
import torch
from safetensors.torch import load_file, save_file
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
sys.path.insert(0, "/")
from rerank import run

source = Path("/repo/testdata/tiny-bert-bundle")
out = Path("/output")
m = json.loads((source / "manifest.json").read_text())
a = m["architecture"]
torch.manual_seed(707)
config = BertConfig(vocab_size=a["vocab_size"], hidden_size=a["hidden"],
    num_hidden_layers=a["layers"], num_attention_heads=a["heads"],
    intermediate_size=a["intermediate"], max_position_embeddings=a["max_positions"],
    type_vocab_size=a["token_types"], layer_norm_eps=a["layer_norm_eps"], num_labels=1)
model = BertForSequenceClassification(config).eval()
state = load_file(str(source / "weights/model.safetensors"))
model.bert.load_state_dict(state, strict=False)
head = {k: v for k, v in model.state_dict().items() if k.startswith("classifier.") or k.startswith("bert.pooler.")}
save_file(head, str(out / "head.safetensors"))
cases = [{"text": q, "passage": p, "prompt_role": "PROMPT_NONE"} for q,p in [
    ("", ""), ("What is France?", "Paris is in France."),
    ("Café 東京 🙂", "Berlin is in Germany."), ("query "*30, "passage "*70)]]
spec = {"task":"TASK_RERANK", "max_seq":32, "query_tokens":8, "max_batch":4, "cases":cases}
(out / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False)+"\n")
with tempfile.TemporaryDirectory() as work:
    model.save_pretrained(work)
    Path(work, "tokenizer.json").write_bytes((source / "tokenizer.json").read_bytes())
    Path(work, "tokenizer_config.json").write_text(json.dumps({"tokenizer_class":"BertTokenizer", "do_lower_case":True,
       "unk_token":"[UNK]", "sep_token":"[SEP]", "pad_token":"[PAD]", "cls_token":"[CLS]", "mask_token":"[MASK]"}))
    run(work, spec, str(out / "reference.safetensors"))
