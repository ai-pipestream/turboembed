"""Write the small ModernBERT that testdata/tiny-modernbert-reference was made from.

Three layers, hidden 32, two heads of width 16, a GeGLU MLP 48 wide (Wi
is [96, 32]), 64 positions, attention global on layer 0 and local on
layers 1 and 2 (global_attn_every_n_layers 3) with a window of 8 (each
query sees the keys within 4 of it), RoPE theta 160000 on the global
layer and 10000 on the local ones, LayerNorm eps 1e-5 without bias.
The weights are transformers' initialisation with seed 0, saved as
ModernBertForMaskedLM, as the upstream checkpoint is (the encoder under
`model.`, the masked-LM head beside it); every norm weight, which that
initialisation leaves at one, is then drawn at random around one, so a
norm read from the wrong tensor shows. The tokenizer is the byte-level
BPE given. The directory carries the sentence-transformers files for mean
pooling with no normalization, which is what sentence-transformers builds
for the upstream checkpoint, so reference.py runs on it as on the real
model. Run it in the reference container, then run reference.py on the
directory it writes.

Arguments: the tokenizer.json to use, and the directory to write.
"""

import json
import os
import sys

import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast


def sentence_transformers_files(out, dim, max_seq):
    os.makedirs(out + "/1_Pooling", exist_ok=True)
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    pooling = {
        "word_embedding_dimension": dim,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
    }
    for path, value in [
        ("modules.json", modules),
        ("1_Pooling/config.json", pooling),
        ("sentence_bert_config.json", {"max_seq_length": max_seq, "do_lower_case": False}),
    ]:
        with open(os.path.join(out, path), "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
            f.write("\n")


def main(tokenizer_json, out):
    torch.manual_seed(0)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="</s>", pad_token="<pad>",
        cls_token="<s>", mask_token="<mask>", model_max_length=64,
    )
    cfg = ModernBertConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
        hidden_activation="gelu", max_position_embeddings=64, norm_eps=1e-5, norm_bias=False,
        global_attn_every_n_layers=3, local_attention=8, global_rope_theta=160000.0, local_rope_theta=10000.0,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id, classifier_pooling="mean", initializer_range=0.2,
    )
    model = ModernBertForMaskedLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    model.save_pretrained(out)
    tok.save_pretrained(out)
    sentence_transformers_files(out, cfg.hidden_size, cfg.max_position_embeddings)


if __name__ == "__main__":
    main(*sys.argv[1:])
