"""Write the small nomic-bert that testdata/tiny-nomic-reference was made from.

Two layers, hidden 32, two heads of width 16, a SwiGLU MLP 48 wide
(`fc12` the gate, `fc11` the value), 64 positions, RoPE theta 1000 on
every layer, post-norm LayerNorm eps 1e-12 with bias, two token types,
vocabulary 30528 over the 30522 WordPiece entries of the tokenizer given
(nomic-embed-text-v1.5 pads its table the same way). The weights are
transformers' initialisation of its native NomicBertModel with seed 0,
saved under the upstream checkpoint's names (transformers writes `Wqkv`,
`fc11`, `fc12` back); every norm weight and bias, which that
initialisation leaves at one and zero, is then drawn at random, so a norm
read from the wrong tensor shows. config.json also carries the upstream
config's keys that say how the model computes (`rotary_emb_*`,
`prenorm`, `use_rms_norm`, `activation_function`, `n_positions`, ...),
with the values this model has, so a recipe check written against the
upstream config reads this one too. The directory carries the
sentence-transformers files for mean pooling with no normalization, as
the upstream model does. Run it in the reference container, then run
reference.py on the directory it writes.

Arguments: the tokenizer.json to use, and the directory to write.
"""

import sys

import torch
from tiny_modernbert import sentence_transformers_files
from transformers import NomicBertConfig, NomicBertModel, PreTrainedTokenizerFast


def main(tokenizer_json, out):
    torch.manual_seed(0)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        model_max_length=64,
    )
    cfg = NomicBertConfig(
        vocab_size=30528, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=48,
        hidden_act="silu", max_position_embeddings=64, type_vocab_size=2, layer_norm_eps=1e-12,
        pad_token_id=tok.pad_token_id, initializer_range=0.2, rope_parameters={"rope_type": "default", "rope_theta": 1000.0},
        activation_function="swiglu", n_positions=64, n_inner=48, layer_norm_epsilon=1e-12,
        rotary_emb_base=1000, rotary_emb_fraction=1.0, rotary_emb_interleaved=False, rotary_scaling_factor=None,
        prenorm=False, use_rms_norm=False, qkv_proj_bias=False, mlp_fc1_bias=False, mlp_fc2_bias=False,
    )
    model = NomicBertModel(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "LayerNorm" in name or "layernorm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.2 * torch.randn_like(p))
    model.save_pretrained(out)
    tok.save_pretrained(out)
    sentence_transformers_files(out, cfg.hidden_size, cfg.max_position_embeddings)


if __name__ == "__main__":
    main(*sys.argv[1:])
