"""Write the small DeBERTa-v2 that testdata/tiny-deberta-bundle was made from.

Two layers, hidden 32, four heads, 8 relative buckets over 512 positions,
DeBERTa-v3's shared-key profile (relative attention, c2p and p2c, the
relative table LayerNormed, no absolute positions or token types),
initialised with seed 0, saved by sentence-transformers with mean pooling
and L2 normalization, beside the upstream MiniLM tokenizer. Run it in the
reference container, then run reference.py on the directory it writes
(turbo-bundle reference does).

Arguments: the tokenizer.json to use, and the directory to write.
"""

import sys

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast


def main(tokenizer_json, out):
    torch.manual_seed(0)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        model_max_length=512,
    )
    cfg = DebertaV2Config(
        vocab_size=30522, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        hidden_act="gelu", max_position_embeddings=512, type_vocab_size=0, layer_norm_eps=1e-7,
        relative_attention=True, max_relative_positions=-1, position_buckets=8, position_biased_input=False,
        pos_att_type=["p2c", "c2p"], share_att_key=True, norm_rel_ebd="layer_norm",
    )
    hf = out + "/hf"
    DebertaV2Model(cfg).save_pretrained(hf)
    tok.save_pretrained(hf)
    transformer = models.Transformer(hf, max_seq_length=128)
    SentenceTransformer(modules=[transformer, models.Pooling(32, "mean"), models.Normalize()]).save(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
