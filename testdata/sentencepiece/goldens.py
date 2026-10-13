"""Expected SentencePiece output for the core's SentencePiece tests, from
Google's sentencepiece library (the Python package, which wraps the C++
library): a yardstick for tests, never part of encoding.

    python3 goldens.py train <botchan.txt>
    python3 goldens.py goldens [<models dir>]

`train` writes the small models of MODELS beside this script, trained on
the repository's tokenizer texts and botchan.txt from the sentencepiece
repository (its data/ directory, a public domain novel). `goldens`
writes <name>.goldens.jsonl for every model it finds: the committed
models here, XLM-RoBERTa's model rebuilt from testdata/bge-m3 (see
xlmr_model), and the models of FETCHED when they are in <models dir>.

Each golden file starts with a header line naming the model's file, its
SHA-256 and the versions that ran. Then one line per text:

  text          the input
  normalized    Normalize(text), as UTF-8
  norm_to_orig  for each byte of `normalized` and one past its end, the
                byte of `text` it came from (the library's norm_to_orig)
  ids           encode(text), the ids
  offsets       [begin, end] of each piece in bytes of `text`, from the
                SentencePieceText proto (encode with return_type proto)
  decoded       decode(ids)

The last line is the long text: every text of the file joined by "\\n",
repeated `repeat` times, with the number of pieces and the SHA-256 of
"id begin end\\n" for each piece. It is long enough for the unigram
search to rebase its scores.

`goldens` also writes variants.goldens.jsonl: for each edit of VARIANTS
to a committed model (the whitespace switches set otherwise, or
user-defined pieces added that overlap the model's own), the number of
texts and the SHA-256 of one line per text of texts() and VARIANT_TEXTS,
"index TAB normalized TAB norm_to_orig TAB ids TAB offsets TAB decoded",
the lists comma-separated and each offset "begin:end".
core/tests/sentencepiece.rs makes the same edits and the same lines.
"""

import hashlib
import json
import os
import random
import sys
import tempfile

import google.protobuf
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.dirname(HERE)

SENTENCEPIECE = "0.2.2"
PROTOBUF = "7.36.1"

# Texts beyond testdata/tokenizer-texts.jsonl: the corners of the
# normalizer, the pieces and byte fallback.
EXTRA = [
    "",
    " ",
    "   ",
    "\t",
    "\n",
    "a",
    "Hello world",
    " Hello   world  ",
    "Hello\u3000world\u3000\u3000again",
    "a\u00a0b\u2009c\u202fd",
    "zero\u200bwidth\u200cjoiner\u200dhere\ufeffbom",
    "\ufb01nancial \ufb02uid \ufb00 \ufb03 \ufb06",
    "\u2460 \u246a \u32ff \uff2b\uff21\uff34\uff21\uff2b\uff21\uff2e\uff21",
    "\uff76\uff80\uff76\uff85 \uff76\uff9e\uff8a\uff9f half width",
    "e\u0301 a\u0300 \u00c5 A\u030a o\u0308\u0304",
    "\u1112\u1161\u11ab \u1100\u1173\u11af \ud55c\uae00",
    "<mask> the [URL] token",
    "a <mask>b[URL]c<mask><mask>",
    "control <s> tokens </s> inline <ctl> <unk> <pad>",
    "\u2581 already \u2581\u2581 metaspace\u2581",
    "emoji \U0001f642\U0001f97a \U0001f469\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 \U0001f1e9\U0001f1ea",
    "\U00013000\U00013001 rare \U0001d400\U0001d401",
    "\u0000 nul \u0001 ctrl \u007f del \u0085 nel",
    "\ufffd replacement \ufffd\ufffd",
    "trailing spaces   ",
    "a\r\nb\rc\n\nd",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "\u0149 \u01c5 \u1e9e \u00df \u338f \u2103 \u2122 \u00bd \u2026",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 \u05e9\u05dc\u05d5\u05dd",
    "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22 \u0939\u093f\u0928\u094d\u0926\u0940",
    "\u03a9\u2248\u00e7\u221a\u222b\u02dc\u00b5\u2264 \u1f00\u03c1\u03b5\u03c4\u03ae",
    "\u201cfancy\u201d \u2018single\u2019 \u2014 dash \u2026 ellipsis",
    "\t\ttabs\t\tbetween\t",
    "ABC AB A ABCD x--y ---- A\u200bB",
    "1234567890 3.14159 2026-10-12 0x1F",
    "UPPER lower MiXeD case",
    "\u6771\u4eac\u30bf\u30ef\u30fc\u3078\u884c\u304d\u307e\u3057\u305f \u4f60\u597d\uff0c\u4e16\u754c\uff01",
]

# Random strings over a small alphabet, so pieces meet at every place.
ALPHABET = "ab AB\u2581-\u00e9\u6f22\U0001f642\u3000\t\u200bxyz<>[]"


def random_texts():
    rng = random.Random(193)
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 30))) for _ in range(40)]


def texts():
    out = []
    with open(os.path.join(TESTDATA, "tokenizer-texts.jsonl"), encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
    return out + EXTRA + random_texts()


# The custom rules of `unigram-rules`, as normalization_rule_tsv holds
# them: source code points TAB target code points. Rules that share a
# prefix (A, AB, ABC), a deletion and rules that write spaces.
RULES = [
    ("41", "61"),
    ("41 42", "78"),
    ("41 42 43", "79 7A"),
    ("2D 2D", "2014"),
    ("200B", ""),
    ("3000", "20"),
    ("9", "20 20"),
    ("FB01", "66 69"),
]

# The denormalizer of `unigram-rules`, which decode runs: "x" back to
# "AB", the em dash back to "--".
DENORM_RULES = [
    ("78", "41 42"),
    ("2014", "2D 2D"),
]

COMMON = dict(
    hard_vocab_limit=False,
    self_test_sample_size=0,
    shuffle_input_sentence=False,
    num_threads=1,
    character_coverage=1.0,
)

# name: (training options, edit of the trained proto or None)
MODELS = {
    "unigram-identity": (dict(model_type="unigram", vocab_size=800, normalization_rule_name="identity",
                              user_defined_symbols=["<mask>", "[URL]"], control_symbols=["<ctl>"]), "unused"),
    "unigram-bytes": (dict(model_type="unigram", vocab_size=800, normalization_rule_name="identity",
                           byte_fallback=True, character_coverage=0.98), None),
    "unigram-suffix": (dict(model_type="unigram", vocab_size=600, normalization_rule_name="identity",
                            treat_whitespace_as_suffix=True, remove_extra_whitespaces=False), None),
    "unigram-rules": (dict(model_type="unigram", vocab_size=600, normalization_rule_tsv="rules.tsv",
                           denormalization_rule_tsv="denorm.tsv"), "no-escape"),
    "bpe-identity": (dict(model_type="bpe", vocab_size=800, normalization_rule_name="identity",
                          user_defined_symbols=["<mask>", "[URL]"]), "unused"),
    "bpe-bytes": (dict(model_type="bpe", vocab_size=800, normalization_rule_name="identity",
                       byte_fallback=True, character_coverage=0.98, remove_extra_whitespaces=False,
                       split_digits=True), None),
}

# Edits of committed models for variants.goldens.jsonl: (name, model,
# add_dummy_prefix, remove_extra_whitespaces, treat_whitespace_as_suffix,
# user-defined pieces appended). None leaves a switch as the model has it.
# The pieces are in none of the models and overlap their pieces and each
# other ("wor" and "orl" in "world").
USER_PIECES = ["hel", "llo", "a b", "xyz", "wor", "orl", "e\u0301", "\u2581\u2581"]
VARIANTS = [
    (f"{model}-adp{int(adp)}-rew{int(rew)}-suf{int(suf)}", model, adp, rew, suf, [])
    for model in ["t5", "unigram-identity", "unigram-bytes", "bpe-identity", "bpe-bytes"]
    for adp in (True, False)
    for rew in (True, False)
    for suf in (False, True)
] + [
    (f"{model}-user", model, None, None, None, USER_PIECES)
    for model in ["t5", "unigram-identity", "unigram-bytes", "bpe-identity", "bpe-bytes"]
] + [
    (f"{model}-user-adp0-rew0", model, False, False, None, USER_PIECES)
    for model in ["t5", "unigram-bytes", "bpe-bytes"]
]

# Texts beyond texts() for the variants: the user-defined pieces, and
# whitespace at the ends and in runs.
VARIANT_TEXTS = [
    "hello world",
    "  hello  world  ",
    "Hello wor ld orl",
    "a b a  b ab",
    "xyz xyzxyz",
    "\u2581\u2581 \u2581\u2581\u2581",
    "e\u0301e\u0301 e\u0301",
    "hel lo llo hello",
    " \U0001f642",
    "\U0001f642 ",
    "\u6f22 \u6f22",
    "\t \t",
]

# Published models, not committed: (file, where it is published).
FETCHED = [
    ("mt5.model", "https://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"),
    ("umt5.model", "https://storage.googleapis.com/t5-data/vocabs/umt5.256000/sentencepiece.model"),
    ("llama2.model",
     "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.model"),
]


def check_versions():
    if spm.__version__ != SENTENCEPIECE or google.protobuf.__version__ != PROTOBUF:
        raise SystemExit(f"needs sentencepiece {SENTENCEPIECE} and protobuf {PROTOBUF}, "
                         f"found {spm.__version__} and {google.protobuf.__version__}")


def train(botchan):
    check_versions()
    work = tempfile.mkdtemp()
    corpus = os.path.join(work, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as out:
        with open(botchan, encoding="utf-8") as f:
            out.write(f.read())
        for t in texts():
            out.write(t.replace("\r", " ").replace("\n", " ") + "\n")
    with open(os.path.join(work, "rules.tsv"), "w", encoding="utf-8") as f:
        for src, dst in RULES:
            f.write(f"{src}\t{dst}\n")
    with open(os.path.join(work, "denorm.tsv"), "w", encoding="utf-8") as f:
        for src, dst in DENORM_RULES:
            f.write(f"{src}\t{dst}\n")
    cwd = os.getcwd()
    os.chdir(work)
    try:
        for name, (opts, edit) in MODELS.items():
            options = dict(COMMON)
            options.update(opts)
            spm.SentencePieceTrainer.Train(input=corpus, model_prefix=name, **options)
            m = pb.ModelProto()
            with open(name + ".model", "rb") as f:
                m.ParseFromString(f.read())
            if edit == "unused":
                # Every 50th normal piece of two or more characters is
                # UNUSED: the unigram search passes over it, and BPE
                # splits it back into the pieces it was merged from.
                normal = [p for p in m.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL and len(p.piece) > 1]
                for p in normal[::50]:
                    p.type = pb.ModelProto.SentencePiece.UNUSED
            elif edit == "no-escape":
                m.normalizer_spec.escape_whitespaces = False
            # No training inputs are kept in the file.
            m.trainer_spec.ClearField("input")
            m.trainer_spec.ClearField("model_prefix")
            m.normalizer_spec.ClearField("normalization_rule_tsv")
            m.denormalizer_spec.ClearField("normalization_rule_tsv")
            with open(os.path.join(HERE, name + ".model"), "wb") as f:
                f.write(m.SerializeToString(deterministic=True))
            print(name, len(m.pieces))
    finally:
        os.chdir(cwd)


def xlmr_model():
    """XLM-RoBERTa's SentencePiece model, as far as testdata/bge-m3 keeps
    it: <unk>, <s> and </s> in SentencePiece's order, then the cut down
    vocabulary's pieces after <unk> with their scores, <mask> left out
    (fairseq adds it), and the normalizer with the file's character map.
    Ids are SentencePiece's, one under the tokenizer file's."""
    import base64
    with open(os.path.join(TESTDATA, "bge-m3", "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    m = pb.ModelProto()
    for piece, kind in (("<unk>", 2), ("<s>", 3), ("</s>", 3)):
        p = m.pieces.add()
        p.piece, p.score, p.type = piece, 0.0, kind
    for piece, score in t["model"]["vocab"][4:]:
        if piece == "<mask>":
            continue
        p = m.pieces.add()
        p.piece, p.score, p.type = piece, score, 1
    m.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    m.trainer_spec.vocab_size = len(m.pieces)
    charsmap = next(n for n in t["normalizer"]["normalizers"] if n["type"] == "Precompiled")
    m.normalizer_spec.name = "nmt_nfkc"
    m.normalizer_spec.precompiled_charsmap = base64.b64decode(charsmap["precompiled_charsmap"])
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.remove_extra_whitespaces = True
    m.normalizer_spec.escape_whitespaces = True
    return m.SerializeToString(deterministic=True)


def utf8_offsets(text):
    """The byte offset of each character of `text`, and of its end."""
    out, at = [], 0
    for c in text:
        out.append(at)
        at += len(c.encode("utf-8"))
    out.append(at)
    return out


def golden(name, data, out_path):
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as f:
        f.write(data)
    sp = spm.SentencePieceProcessor(model_file=f.name)
    os.unlink(f.name)
    all_texts = texts()
    with open(out_path, "w", encoding="utf-8") as out:
        header = dict(model=name, sha256=hashlib.sha256(data).hexdigest(), sentencepiece=spm.__version__,
                      protobuf=google.protobuf.__version__)
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for text in all_texts:
            ids = sp.encode(text)
            proto = sp.encode(text, return_type="proto")
            assert [p.id for p in proto.pieces] == ids, (name, text)
            normalized, char_offsets = sp.Normalize(text, with_offsets=True)
            # The library gives a source per character; each byte of a
            # normalized character comes from the same source.
            src = utf8_offsets(text)
            norm_to_orig = []
            for c, o in zip(normalized, char_offsets):
                norm_to_orig += [src[o]] * len(c.encode("utf-8"))
            norm_to_orig.append(src[char_offsets[-1]] if char_offsets else 0)
            line = dict(text=text, normalized=normalized, norm_to_orig=norm_to_orig, ids=ids,
                        offsets=[[p.begin, p.end] for p in proto.pieces], decoded=sp.decode(ids))
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
        joined = "\n".join(all_texts)
        repeat = 60000 // len(joined.encode("utf-8")) + 1
        long = "\n".join([joined] * repeat)
        proto = sp.encode(long, return_type="proto")
        h = hashlib.sha256()
        for p in proto.pieces:
            h.update(f"{p.id} {p.begin} {p.end}\n".encode())
        out.write(json.dumps(dict(long=dict(repeat=repeat, count=len(proto.pieces), sha256=h.hexdigest()))) + "\n")
    print(out_path)


def variant_model(model, adp, rew, suf, user):
    m = pb.ModelProto()
    with open(os.path.join(HERE, model + ".model"), "rb") as f:
        m.ParseFromString(f.read())
    if adp is not None:
        m.normalizer_spec.add_dummy_prefix = adp
    if rew is not None:
        m.normalizer_spec.remove_extra_whitespaces = rew
    if suf is not None:
        m.trainer_spec.treat_whitespace_as_suffix = suf
    for piece in user:
        p = m.pieces.add()
        p.piece, p.score, p.type = piece, 0.0, pb.ModelProto.SentencePiece.USER_DEFINED
    return m.SerializeToString(deterministic=True)


def variant_goldens(out_path):
    all_texts = texts() + VARIANT_TEXTS
    with open(out_path, "w", encoding="utf-8") as out:
        header = dict(sentencepiece=spm.__version__, protobuf=google.protobuf.__version__)
        out.write(json.dumps(header) + "\n")
        for name, model, adp, rew, suf, user in VARIANTS:
            sp = spm.SentencePieceProcessor()
            sp.LoadFromSerializedProto(variant_model(model, adp, rew, suf, user))
            h = hashlib.sha256()
            for i, text in enumerate(all_texts):
                ids = sp.encode(text)
                proto = sp.encode(text, return_type="proto")
                assert [p.id for p in proto.pieces] == ids, (name, text)
                normalized, char_offsets = sp.Normalize(text, with_offsets=True)
                src = utf8_offsets(text)
                norm_to_orig = []
                for c, o in zip(normalized, char_offsets):
                    norm_to_orig += [src[o]] * len(c.encode("utf-8"))
                norm_to_orig.append(src[char_offsets[-1]] if char_offsets else 0)
                fields = [str(i), normalized, ",".join(map(str, norm_to_orig)), ",".join(map(str, ids)),
                          ",".join(f"{p.begin}:{p.end}" for p in proto.pieces), sp.decode(ids)]
                h.update(("\t".join(fields) + "\n").encode("utf-8"))
            out.write(json.dumps(dict(variant=name, texts=len(all_texts), sha256=h.hexdigest())) + "\n")
    print(out_path)


def goldens(models_dir):
    check_versions()
    variant_goldens(os.path.join(HERE, "variants.goldens.jsonl"))
    for name in list(MODELS) + ["t5"]:
        with open(os.path.join(HERE, name + ".model"), "rb") as f:
            golden(name + ".model", f.read(), os.path.join(HERE, name + ".goldens.jsonl"))
    golden("xlmr (rebuilt from bge-m3)", xlmr_model(), os.path.join(HERE, "xlmr.goldens.jsonl"))
    if models_dir:
        for file, _ in FETCHED:
            path = os.path.join(models_dir, file)
            with open(path, "rb") as f:
                data = f.read()
            golden(file, data, os.path.join(HERE, file.replace(".model", ".goldens.jsonl")))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        train(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "goldens":
        goldens(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        raise SystemExit(__doc__)
