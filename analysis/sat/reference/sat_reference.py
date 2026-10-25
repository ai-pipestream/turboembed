"""The reference SaT run that analysis/sat is compared with, in the
container of this directory (Dockerfile). Three commands:

  tiny <tokenizer.json> <out-dir> [lookahead]
      Writes a small SaT model with random weights: wtpsplit's
      SubwordXLMForTokenClassification over XLM-RoBERTa, two layers,
      hidden 32, four heads, one label, no limited lookahead (or
      wtpsplit's `lookahead`, the keys ahead over all the layers, which
      it divides among them: 8 is 4 a layer; the weights are the same), the
      positions of max_seq 64, initialised from seed 0, with its
      classifier's weights drawn again with standard deviation 1 so that
      the boundary probabilities spread over (0, 1). Beside it the
      tokenizer, as transformers' XLMRobertaTokenizerFast saves it.
      <out-dir>/model and <out-dir>/tokenizer.

  reference <model-dir> <tokenizer-dir> <recipe.json> <out.safetensors> <report.json> <container>
      The bundle's reference file for the recipe's cases: the ids the
      model sees (the template's <s> and </s> around the text, cut at
      embed.max_seq), their lengths, and the mean of the encoder's final
      states over each row, unnormalized, as POOLING_MEAN and
      NORMALIZE_NONE say. The report is what turbo-bundle seal reads.

  segment <model-dir> <tokenizer.json> <texts.jsonl> <out.jsonl> <threshold> <stride> <block-size>
      For each text, wtpsplit's SaT.predict_proba (the boundary
      probability of every character) and SaT.split (the sentences),
      with those settings, one text at a time. The first line names
      the versions and the settings.

      The tokenizer is the model's tokenizer.json as the file says
      (PreTrainedTokenizerFast), which is how transformers 4, which SaT
      was trained with, loads XLM-RoBERTa's. transformers 5 builds
      XLMRobertaTokenizer from the vocabulary with a pipeline of its own
      instead: no collapse of space runs, a split at every Unicode
      whitespace character, so U+2028, U+2029 and U+0085 end words, and
      no word start piece for whitespace at the end of a text.
"""

import hashlib
import json
import sys
import types

import numpy as np
import torch
import transformers
import wtpsplit
from safetensors.numpy import save_file
from transformers import PreTrainedTokenizerFast, XLMRobertaTokenizerFast
from wtpsplit import SaT
from wtpsplit.configs import SubwordXLMConfig
from wtpsplit.models import SubwordXLMForTokenClassification


def versions():
    return f"wtpsplit {wtpsplit.__version__} (transformers {transformers.__version__}, torch {torch.__version__}, cpu, float32)"


def tiny(tokenizer_json, out, lookahead=None):
    tok = XLMRobertaTokenizerFast(tokenizer_file=tokenizer_json, model_max_length=64)
    tok.save_pretrained(out + "/tokenizer")
    torch.manual_seed(0)
    cfg = SubwordXLMConfig(
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=66,
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=1,
        lookahead=None if lookahead is None else int(lookahead),
        lookahead_split_layers=None,
    )
    model = SubwordXLMForTokenClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 1.0)
        model.classifier.bias.zero_()
    model.save_pretrained(out + "/model")


def reference(model_dir, tokenizer_dir, recipe_path, out_path, report_path, container):
    with open(recipe_path, encoding="utf-8") as f:
        manifest = json.load(f)["manifest"]
    max_seq = manifest["embed"]["max_seq"]
    texts = []
    for c in manifest["reference"]["cases"]:
        if c["prompt_role"] != "PROMPT_NONE":
            sys.exit("the SaT recipe's cases have no prompt")
        texts.append(c["text"])
    torch.use_deterministic_algorithms(True)
    tok = XLMRobertaTokenizerFast.from_pretrained(tokenizer_dir)
    model = SubwordXLMForTokenClassification.from_pretrained(model_dir).eval()
    enc = tok(texts, truncation=True, max_length=max_seq, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    def pooled(i, m):
        with torch.no_grad():
            h = model.roberta(input_ids=i, attention_mask=m).last_hidden_state
        w = m.unsqueeze(-1).to(h.dtype)
        return ((h * w).sum(1) / w.sum(1)).numpy()

    one = np.concatenate([pooled(ids[k : k + 1, : int(mask[k].sum())], mask[k : k + 1, : int(mask[k].sum())]) for k in range(len(texts))])
    batched = pooled(ids, mask)
    worst = float(np.abs(one - batched).max())
    if worst > 1e-4:
        sys.exit(f"the batch differs from one text at a time by {worst}")
    tensors = {
        "ids": np.ascontiguousarray(ids.numpy().astype(np.int32)),
        "lengths": np.ascontiguousarray(mask.numpy().sum(axis=1).astype(np.int32)),
        "embeddings": np.ascontiguousarray(one.astype(np.float32)),
    }
    if "classifier" in manifest:
        classifier = manifest["classifier"]
        if classifier["kind"] != "CLASSIFIER_TOKEN":
            sys.exit("the SaT reference requires CLASSIFIER_TOKEN")
        labels = classifier["labels"]
        upstream = [model.config.id2label[i] for i in range(model.classifier.out_features)]
        if labels != upstream:
            sys.exit("the SaT classifier labels differ from upstream id2label")
        # Unpadded rows in reference order, including special tokens, as
        # the token-classifier bundle verifier expects. Preserve every
        # output column; SaT's boundary consumer selects NEWLINE_INDEX.
        with torch.no_grad():
            logits = [
                model(input_ids=ids[k:k + 1, :int(mask[k].sum())],
                      attention_mask=mask[k:k + 1, :int(mask[k].sum())]).logits[0].numpy()
                for k in range(len(texts))
            ]
        tensors["logits"] = np.ascontiguousarray(np.concatenate(logits).astype(np.float32))
    save_file(tensors, out_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"tool": "wtpsplit", "tool_version": versions(), "args": sys.argv[2:], "container": container}, f)


def file_tokenizer(tokenizer_json):
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def segment(model_dir, tokenizer_json, texts_path, out_path, threshold, stride, block_size):
    threshold, stride, block_size = float(threshold), int(stride), int(block_size)
    torch.use_deterministic_algorithms(True)
    tok = file_tokenizer(tokenizer_json)
    # SaT.__init__ loads its tokenizer with AutoTokenizer; it is handed
    # this one instead, and sets the rest from it as it would.
    loader = wtpsplit.AutoTokenizer
    wtpsplit.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: tok)
    try:
        sat = SaT(model_dir, tokenizer_name_or_path=tokenizer_json, hub_prefix=None)
    finally:
        wtpsplit.AutoTokenizer = loader
    assert sat.tokenizer is tok
    with open(model_dir + "/model.safetensors", "rb") as f:
        weights = hashlib.sha256(f.read()).hexdigest()
    with open(texts_path, encoding="utf-8") as f:
        texts = [json.loads(line) for line in f if line.strip()]
    with open(out_path, "w", encoding="utf-8") as out:
        header = {
            "tool": versions(),
            "weights_sha256": weights,
            "threshold": threshold,
            "stride": stride,
            "block_size": block_size,
        }
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for text in texts:
            enc = sat.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
            probs = sat.predict_proba(text, stride=stride, block_size=block_size)
            split = sat.split(text, threshold=threshold, stride=stride, block_size=block_size)
            record = {
                "text": text,
                "ids": enc["input_ids"],
                "offsets": [list(o) for o in enc["offset_mapping"]],
                "probs": [float(p) for p in np.asarray(probs, dtype=np.float32)],
                "split": split,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"tiny": tiny, "reference": reference, "segment": segment}[command](*args)
