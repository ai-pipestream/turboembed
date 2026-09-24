"""Write the small BERT that testdata/tiny-bert-reference was made from.

Two layers, hidden 32, four heads, initialised with seed 0, saved by
sentence-transformers with mean pooling and L2 normalization, beside the
upstream MiniLM tokenizer. Run it in the reference container, then run
reference.py on the directory it writes.

Arguments: the tokenizer.json to use, and the directory to write.
"""

import sys

import torch
from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast


def main(tokenizer_json, out):
    torch.manual_seed(0)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        model_max_length=512,
    )
    cfg = BertConfig(
        vocab_size=30522, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=512,
    )
    hf = out + "/hf"
    BertModel(cfg).save_pretrained(hf)
    tok.save_pretrained(hf)
    transformer = models.Transformer(hf, max_seq_length=64)
    SentenceTransformer(modules=[transformer, models.Pooling(32, "mean"), models.Normalize()]).save(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
