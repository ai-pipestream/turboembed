"""Reproduce the tiny QA fixture using the existing tiny BERT encoder.

Run in the reference image with the repository mounted at /repo. This
untrained head proves arithmetic and offsets, not SQuAD model accuracy.
"""
import json
import pathlib
import sys
import torch
import transformers
from safetensors.torch import load_file, save_file
from safetensors.numpy import save_file as save_numpy
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast
from qa_reference import tensors


root = pathlib.Path(sys.argv[1])
out = root / "testdata/tiny-qa-reference"
upstream = root / "target/wave22/qa-upstream"
out.mkdir(parents=True, exist_ok=True)
upstream.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(745)
cfg = BertConfig(vocab_size=30522, hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=4, intermediate_size=64, max_position_embeddings=512)
model = BertForQuestionAnswering(cfg).eval()
weights = load_file(str(root / "testdata/tiny-bert-bundle/weights/model.safetensors"))
missing, unexpected = model.bert.load_state_dict(weights, strict=False)
assert not missing and all(k.startswith("pooler.") for k in unexpected), (missing, unexpected)
model.save_pretrained(upstream)
tok = BertTokenizerFast(tokenizer_file=str(root / "testdata/tiny-bert-bundle/tokenizer.json"))
tok.save_pretrained(upstream)
save_file({"qa_outputs.weight": model.qa_outputs.weight.detach().contiguous(),
           "qa_outputs.bias": model.qa_outputs.bias.detach().contiguous()}, str(out / "head.safetensors"))
spec = {"task": "TASK_EXTRACTIVE_QA",
        "qa": {"max_seq": 24, "max_batch": 2, "max_query_tokens": 6, "doc_stride": 5},
        "cases": [{"text": "Where is the café?", "context": "The café is in Montréal."},
                  {"text": "What is here?", "context": "alpha " * 35 + "Café 東京 here."},
                  {"text": "An impossible question", "context": ""},
                  {"text": "question " * 20, "context": "emoji 🙂 café"}]}
save_numpy(tensors(str(upstream), spec), str(out / "reference.safetensors"))
(out / "cases.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
(out / "receipt.json").write_text(json.dumps(
    {"torch": torch.__version__, "transformers": transformers.__version__, "seed": 745,
     "model": "existing tiny BERT with untrained two-output QA head"}, indent=2) + "\n")
