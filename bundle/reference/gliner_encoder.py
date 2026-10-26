"""Write GLiNER's DeBERTa-v3 encoder as an upstream model directory.

From the zoo's urchade-gliner-small-v2.1 checkpoint and the
microsoft-deberta-v3-small-tokenizer files: the checkpoint's
token_rep_layer.bert_layer.model tensors as a DebertaV2Model of the
tokenizer's config.json, its vocabulary the checkpoint's (128004 rows,
with GLiNER's added tokens); spm.model converted to tokenizer.json, with
[MASK] and GLiNER's [FLERT], <<ENT>> and <<SEP>> pieces at the ids the
checkpoint uses, its normalizer in SentencePiece's own order; saved by sentence-transformers with mean pooling and no
normalization. Run it in the reference container, then run reference.py on
the directory it writes (turbo-bundle reference does).

Arguments: the DeBERTa config.json, spm.model, the GLiNER checkpoint, and
the directory to write.
"""

import json
import sys

import torch
import transformers.convert_slow_tokenizer as convert
from sentence_transformers import SentenceTransformer, models
from tokenizers import AddedToken, Tokenizer, normalizers
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast

PREFIX = "token_rep_layer.bert_layer.model."
ADDED = {"[MASK]": 128000, "[FLERT]": 128001, "<<ENT>>": 128002, "<<SEP>>": 128003}


class Spm:
    """What DebertaV2Converter reads of a slow tokenizer: the model file
    and DeBERTa-v3's options; [CLS] and [SEP] are pieces 1 and 2."""

    do_lower_case = False
    split_by_punct = False

    def __init__(self, path):
        self.vocab_file = path

    def convert_tokens_to_ids(self, token):
        return {"[CLS]": 1, "[SEP]": 2}[token]


def main(config_json, spm_model, checkpoint, out):
    tok = convert.DebertaV2Converter(Spm(spm_model)).converted()
    # The converter strips first; SentencePiece itself normalizes with the
    # character map and then removes the extra whitespace, as the core
    # does: the map, the collapse of spaces, then the strip.
    steps = {type(n).__name__: n for n in tok.normalizer}
    tok.normalizer = normalizers.Sequence([steps["Precompiled"], steps["Replace"], steps["Strip"]])
    # A sequence of one Metaspace is that Metaspace.
    (metaspace,) = list(tok.pre_tokenizer)
    tok.pre_tokenizer = metaspace
    # [MASK] and GLiNER's markers as pieces, at the checkpoint's ids, as
    # XLM-R's tokenizer holds <mask>: the core reads the special tokens
    # from the model's vocabulary.
    spec = json.loads(tok.to_str())
    spec["model"]["vocab"].extend([t, 0.0] for t in ADDED)
    tok = Tokenizer.from_str(json.dumps(spec))
    # [MASK] an added token as well; GLiNER's markers are pieces alone,
    # which GLiNER writes by id.
    tok.add_special_tokens([AddedToken("[MASK]", special=True)])
    got = {t: tok.token_to_id(t) for t in ADDED}
    if got != ADDED:
        sys.exit(f"added tokens at {got}, the checkpoint uses {ADDED}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    cfg = DebertaV2Config.from_json_file(config_json)
    cfg.vocab_size = encoder["embeddings.word_embeddings.weight"].shape[0]
    if cfg.vocab_size != tok.get_vocab_size():
        sys.exit(f"the checkpoint has {cfg.vocab_size} rows, the tokenizer {tok.get_vocab_size()} pieces")
    model = DebertaV2Model(cfg)
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        sys.exit(f"the checkpoint's encoder does not fit DebertaV2Model: missing {missing}, unexpected {unexpected}")

    hf = out + "/hf"
    model.save_pretrained(hf)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
        mask_token="[MASK]", model_max_length=512,
    )
    fast.save_pretrained(hf)
    transformer = models.Transformer(hf, max_seq_length=512)
    SentenceTransformer(modules=[transformer, models.Pooling(cfg.hidden_size, "mean")]).save(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
