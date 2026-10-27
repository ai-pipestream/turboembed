"""Write a GLiNER or GLiNER2 checkpoint's DeBERTa-v3 encoder as an upstream
model directory.

From the zoo's urchade-gliner-small-v2.1 checkpoint (4e091416) and the
microsoft-deberta-v3-small-tokenizer files (microsoft/deberta-v3-small at
a36c7390): the checkpoint's token_rep_layer.bert_layer.model tensors as a
DebertaV2Model of the tokenizer's config.json, loaded with strict=False and
then refused on any missing key but position_ids or any unexpected one, its vocabulary the checkpoint's (128004 rows,
with GLiNER's added tokens); spm.model converted to tokenizer.json, with
[MASK] and GLiNER's [FLERT], <<ENT>> and <<SEP>> pieces at the ids the
checkpoint uses, its normalizer in SentencePiece's own order; saved by
sentence-transformers with mean pooling and no normalization. Run it in
the reference container, then run reference.py on the directory it writes
(turbo-bundle reference does).

From the zoo's fastino-gliner2-base-v1 checkpoint (#920) the same, with
its encoder. tensors and its own files: encoder_config/config.json,
spm.model, model.safetensors (read as safetensors), and added_tokens.json
for the added pieces (128011 rows).

From the zoo's jackboyla-glirel-large-v0 checkpoint (#920, 40a523e1) the
same, with GLiNER's encoder prefix, its pytorch_model.bin and the
microsoft-deberta-v3-large-tokenizer files (microsoft/deberta-v3-large at
64a8c8ea): GLiREL's tokenizer adds [FLERT], [REL], [E] and [/E] after
[MASK] (128005 rows), which the added-tokens argument `glirel` names.

Arguments: the DeBERTa config.json, spm.model, the checkpoint
(pytorch_model.bin or model.safetensors), the directory to write, and,
optionally, the checkpoint's encoder tensor prefix (default GLiNER's) and
its added_tokens.json (default GLiNER's four pieces), whose tokens become
pieces in id order, or `glirel` for GLiREL's five.
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
GLIREL = {"[MASK]": 128000, "[FLERT]": 128001, "[REL]": 128002, "[E]": 128003, "[/E]": 128004}


def added(path):
    """The added tokens of a checkpoint's added_tokens.json, in id order,
    each at the next id after the last: the pieces are appended in that
    order, so their ids must run on from the vocabulary without a gap."""
    with open(path) as f:
        table = json.load(f)
    return dict(sorted(table.items(), key=lambda kv: kv[1]))


def load(checkpoint):
    """The checkpoint's tensors: safetensors by name, else torch's pickle."""
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint)
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


class Spm:
    """What DebertaV2Converter reads of a slow tokenizer: the model file
    and DeBERTa-v3's options; [CLS] and [SEP] are pieces 1 and 2."""

    do_lower_case = False
    split_by_punct = False

    def __init__(self, path):
        self.vocab_file = path

    def convert_tokens_to_ids(self, token):
        return {"[CLS]": 1, "[SEP]": 2}[token]


def main(config_json, spm_model, checkpoint, out, prefix=PREFIX, added_tokens=None):
    pieces = GLIREL if added_tokens == "glirel" else added(added_tokens) if added_tokens else ADDED
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
    spec["model"]["vocab"].extend([t, 0.0] for t in pieces)
    tok = Tokenizer.from_str(json.dumps(spec))
    # [MASK] an added token as well; GLiNER's markers are pieces alone,
    # which GLiNER writes by id.
    tok.add_special_tokens([AddedToken("[MASK]", special=True)])
    got = {t: tok.token_to_id(t) for t in pieces}
    if got != pieces:
        sys.exit(f"added tokens at {got}, the checkpoint uses {pieces}")

    state = load(checkpoint)
    encoder = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not encoder:
        sys.exit(f"the checkpoint has no tensors under {prefix!r}")
    cfg = DebertaV2Config.from_json_file(config_json)
    cfg.vocab_size = encoder["embeddings.word_embeddings.weight"].shape[0]
    if cfg.vocab_size != tok.get_vocab_size():
        sys.exit(f"the checkpoint has {cfg.vocab_size} rows, the tokenizer {tok.get_vocab_size()} pieces")
    model = DebertaV2Model(cfg)
    # strict=False, then every key checked here: position_ids is a buffer
    # the checkpoint need not carry; anything else missing or unexpected
    # stops the run.
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
