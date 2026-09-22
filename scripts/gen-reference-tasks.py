#!/usr/bin/env python3
"""Generate the precision references the live task suite compares against.

The reranker, classifier and token-classifier references are computed from
the Hugging Face checkpoints in PyTorch, float32, on the CPU. They are the
numbers `crates/turbo-conformance/tests/live_tasks.rs` holds a provider to,
the way `testdata/reference_embeddings/` holds the embedding suite.

Outputs (one file per model, overwritten in place):

    testdata/reference_rerank/ms_marco_minilm_l6.json
    testdata/reference_classify/sst2_distilbert.json
    testdata/reference_token_classify/bert_base_ner.json

Each file records the model id, the revision, the torch/transformers
versions, the machine and the date, so a stale reference is visible without
rerunning it. The case lists below are fixed, so a rerun on the same
checkout reproduces the same numbers.

Usage (uv brings its own CPU wheels; nothing is installed into the tree):

    uv run --no-project --with torch --with transformers --with numpy \
        python scripts/gen-reference-tasks.py --machine krick

Options:
    --machine NAME   machine name recorded in each file (default: hostname)
    --date YYYY-MM-DD  date recorded in each file (default: today)
    --models DIR     directory holding the rerank/sst2/ner checkouts
                     (default: ~/opt/models)
    --only NAME      generate one of rerank, classify, token_classify
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMMAND = (
    "uv run --no-project --with torch --with transformers --with numpy "
    "python scripts/gen-reference-tasks.py"
)

# Hugging Face repositories the three bundles under ~/opt/bundles were
# imported from. Weights come from the hub at a pinned revision; the local
# checkouts supply the tokenizers the bundles were built from.
#
# The reranker bundle declares `cross-encoder/ms-marco-MiniLM-L-12-v2`, but
# its ONNX graph has six encoder layers and 22.7M parameters, which is
# L-6-v2; L-12-v2 has twelve layers and 33.4M. The local checkout's
# config.json is the L-6-v2 config (byte-identical to the hub's) while its
# model.safetensors is a 12-layer file, so loading the checkout as a whole
# silently drops half the weights. The reference is therefore built from
# the hub repository that matches the ONNX.
RERANK_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLASSIFY_REPO = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
NER_REPO = "dslim/bert-base-NER"

# A passage long enough that the 256 column budget truncates the pair.
LONG_PASSAGE = (
    "The history of the city stretches back more than seven centuries. "
    "Merchants settled along the river because the crossing was shallow "
    "enough for carts, and a market grew beside the ford. "
) + (
    "Records from the guild archives list the names of the wool traders, "
    "the tanners, the coopers, the bakers and the smiths who paid the town "
    "levy in each of those years, along with the sums they owed and the "
    "dates on which the treasurer entered them in the ledger. "
) * 6

LONG_REVIEW = (
    "I went into this film expecting very little and came out three hours "
    "later with a great deal to think about. "
) + (
    "The photography lingers on faces for longer than is comfortable, the "
    "score is restrained almost to the point of absence, and the editing "
    "refuses the easy cut that a lesser picture would have reached for, "
    "which together make the long central sequence feel earned rather than "
    "indulgent. "
) * 6

RERANK_CASES = [
    ("short_relevant", "How many people live in Berlin?",
     "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("short_irrelevant", "How many people live in Berlin?",
     "New York City is famous for its pizza and bagels."),
    ("short_topical", "How many people live in Berlin?",
     "Berlin is well known for its museums."),
    ("capital_exact", "What is the capital of France?",
     "Paris is the capital and most populous city of France."),
    ("capital_restated", "What is the capital of France?",
     "France is a country in Western Europe whose capital is Paris."),
    ("capital_wrong_country", "What is the capital of France?",
     "The capital of Japan is Tokyo, a city of 14 million people."),
    ("one_word_query", "photosynthesis",
     "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in sugars."),
    ("empty_document", "How many people live in Berlin?", ""),
    ("long_document", "When did the wool traders pay the town levy?", LONG_PASSAGE),
    ("non_english_de", "Wie viele Menschen leben in Berlin?",
     "Berlin hat 3.520.031 gemeldete Einwohner auf einer Flaeche von 891,82 Quadratkilometern."),
    ("non_english_fr", "Quelle est la capitale de la France ?",
     "Paris est la capitale et la ville la plus peuplee de la France."),
    ("query_equals_document", "How many people live in Berlin?",
     "How many people live in Berlin?"),
]

CLASSIFY_CASES = [
    ("clearly_positive", "I absolutely loved this movie, it was wonderful."),
    ("clearly_negative", "This was a dreadful, boring waste of time."),
    ("mild_positive", "A pleasant enough way to spend an afternoon."),
    ("mild_negative", "It drags in the middle and never quite recovers."),
    ("mixed", "The acting is superb but the script is a mess."),
    ("neutral_statement", "The film was released in nineteen ninety four."),
    ("negation", "I would not call this a bad film at all."),
    ("one_word", "terrible"),
    ("empty", ""),
    ("punctuation_and_case", "WOW!!! Best. Movie. Ever."),
    ("non_english_fr", "Ce film est vraiment magnifique et tres emouvant."),
    ("long_review", LONG_REVIEW),
]

NER_CASES = [
    ("classic", "Ada Lovelace visited Berlin with colleagues from Microsoft."),
    ("politics", "Angela Merkel met Barack Obama in Washington last Monday."),
    ("adjacent_same_type", "Germany, France and Italy signed the agreement."),
    ("org_then_loc", "Siemens opened a research office in Munich and in Prague."),
    ("hyphenated", "Jean-Claude Juncker worked for the European Commission in Brussels."),
    ("accented", "Sao Paulo and Zurich hosted the meeting of the World Bank."),
    ("subword_names", "Wolfgang Schaeuble and Nikolaus Blome interviewed Ursula von der Leyen."),
    ("no_entities", "The weather turned cold and the meeting was moved indoors."),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def byte_offsets(text, char_start, char_end):
    """Character offsets from a fast tokenizer to byte offsets in `text`."""
    return (
        len(text[:char_start].encode("utf-8")),
        len(text[:char_end].encode("utf-8")),
    )


def provenance(args, torch, transformers, numpy, model_id, revision, source, extra):
    d = {
        "model_id": model_id,
        "revision": revision,
        "source": source,
        "produced_by": {
            "framework": "pytorch",
            "dtype": "float32",
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "python": platform.python_version(),
        },
        "machine": args.machine,
        "date": args.date,
        "command": COMMAND,
    }
    d.update(extra)
    return d


def f32(x):
    """Round-trip a float through float32 so the JSON holds no fp64 tail."""
    import numpy as np

    return float(np.float32(x))


def load_checked(auto, repo_id, revision, torch, allow=()):
    """Load `repo_id` in float32 and refuse a checkpoint that does not fit.

    A config and a weight file that disagree (the reranker checkout on krick
    pairs a 6-layer config with a 12-layer safetensors) load without an
    exception and produce numbers that look plausible and are wrong, so the
    unused and missing keys are checked here instead.
    """
    model, info = auto.from_pretrained(
        repo_id, revision=revision, dtype=torch.float32, output_loading_info=True
    )
    stray = [
        k
        for k in list(info.get("missing_keys", [])) + list(info.get("unexpected_keys", []))
        if not any(k.startswith(a) for a in allow)
    ]
    if stray:
        raise SystemExit(f"{repo_id}: checkpoint does not match the architecture: {stray[:8]}")
    return model.eval()


def resolve_revision(repo_id):
    """The commit the hub currently serves for `repo_id`, or None offline."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception as e:  # offline, or the hub refused
        print(f"  warning: could not resolve the revision of {repo_id}: {e}", file=sys.stderr)
        return None


def gen_rerank(args, torch, transformers, np):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = args.models / "rerank"
    tok = AutoTokenizer.from_pretrained(src)
    model = load_checked(
        AutoModelForSequenceClassification, RERANK_REPO, args.rerank_revision, torch
    )
    assert model.config.num_hidden_layers == 6, model.config.num_hidden_layers

    cases = []
    for name, query, doc in RERANK_CASES:
        enc = tok(query, doc, truncation="longest_first", max_length=args.rerank_max_seq, return_tensors="pt")
        ids = [int(i) for i in enc["input_ids"][0]]
        if ids.count(tok.sep_token_id) < 2:
            # transformers drops an empty second sequence and encodes
            # `[CLS] query [SEP]`, a single segment. A reranker's pair
            # encoder does not: it emits `[CLS] query [SEP] [SEP]` with the
            # trailing [SEP] in segment 1, which is what the providers
            # produce and what "this query against an empty document"
            # means. The pair is built explicitly so the reference is that
            # encoding and not the collapsed one.
            q = tok(query, add_special_tokens=False)["input_ids"]
            ids = [tok.cls_token_id] + q + [tok.sep_token_id, tok.sep_token_id]
            types = [0] * (len(q) + 2) + [1]
            enc = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "token_type_ids": torch.tensor([types]),
            }
        with torch.no_grad():
            logit = model(**enc).logits
        assert logit.shape == (1, 1), logit.shape
        value = float(logit[0, 0])
        cases.append(
            {
                "id": name,
                "query": query,
                "document": doc,
                "n_tokens": len(ids),
                "input_ids": ids,
                "token_type_ids": [int(t) for t in enc["token_type_ids"][0]],
                "logit": f32(value),
                "sigmoid": f32(1.0 / (1.0 + np.exp(-np.float64(value)))),
            }
        )

    out = provenance(
        args, torch, transformers, np, RERANK_REPO, args.rerank_revision,
        f"weights from the hub; tokenizer from the local checkout {src}",
        {
            "schema": "turbo-reference-rerank/1",
            "task": "rerank",
            "labels": ["LABEL_0"],
            "head": "BertForSequenceClassification, one logit",
            "activation_note": (
                "The checkpoint's sentence-transformers default activation is Identity "
                "(config.json sbert_ce_default_activation_function); the bundle under "
                "~/opt/bundles/rerank-onnx declares contract.activation `sigmoid`. Both "
                "the logit and its sigmoid are stored, so a provider is compared against "
                "whichever its bundle declares."
            ),
            "tokenization": {
                "truncation": "longest_first",
                "max_length": args.rerank_max_seq,
                "note": (
                    "matches TURBO_TRUNCATE_MODEL on a session with max_seq = max_length. "
                    "An empty document is encoded as `[CLS] query [SEP] [SEP]` with the "
                    "trailing [SEP] in segment 1; transformers collapses that pair to a "
                    "single segment, the providers do not, and the pair encoding is the "
                    "one this reference holds."
                ),
            },
            "config_sha256": sha256(src / "config.json"),
            "tokenizer_sha256": sha256(src / "tokenizer.json"),
        },
    )
    out["cases"] = cases
    return "reference_rerank/ms_marco_minilm_l6.json", out


def gen_classify(args, torch, transformers, np):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = args.models / "sst2"
    tok = AutoTokenizer.from_pretrained(src)
    model = load_checked(AutoModelForSequenceClassification, CLASSIFY_REPO, args.classify_revision, torch)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    cases = []
    for name, text in CLASSIFY_CASES:
        enc = tok(text, truncation=True, max_length=args.classify_max_seq, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits.double(), dim=-1)
        cases.append(
            {
                "id": name,
                "text": text,
                "n_tokens": int(enc["input_ids"].shape[1]),
                "input_ids": [int(i) for i in enc["input_ids"][0]],
                "logits": [f32(v) for v in logits.tolist()],
                "probs": [f32(v) for v in probs.tolist()],
            }
        )

    out = provenance(
        args, torch, transformers, np, CLASSIFY_REPO, args.classify_revision,
        f"weights from the hub; tokenizer and config from the local checkout {src}",
        {
            "schema": "turbo-reference-classify/1",
            "task": "classify",
            "labels": labels,
            "activation": "softmax",
            "tokenization": {"truncation": "longest_first", "max_length": args.classify_max_seq},
            "config_sha256": sha256(src / "config.json"),
            "tokenizer_sha256": sha256(src / "tokenizer.json"),
        },
    )
    out["cases"] = cases
    return "reference_classify/sst2_distilbert.json", out


def gen_token_classify(args, torch, transformers, np):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    src = args.models / "ner"
    tok = AutoTokenizer.from_pretrained(src)
    model = load_checked(AutoModelForTokenClassification, NER_REPO, args.ner_revision, torch, allow={"bert.pooler"})
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    strategies = ["simple", "first", "max"]
    pipes = {
        s: pipeline(
            "token-classification",
            model=model,
            tokenizer=tok,
            aggregation_strategy=s,
            device=-1,
        )
        for s in strategies
    }

    cases = []
    for name, text in NER_CASES:
        enc = tok(
            text,
            truncation=True,
            max_length=args.ner_max_seq,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits.double(), dim=-1)
        ids = [int(i) for i in enc["input_ids"][0]]
        special = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
        tokens = []
        for col, tid in enumerate(ids):
            row = probs[col].tolist()
            label_id = int(np.argmax(row))
            cs, ce = offsets[col]
            bs, be = byte_offsets(text, cs, ce)
            tokens.append(
                {
                    "column": col,
                    "token": tok.convert_ids_to_tokens(tid),
                    "special": bool(special[col]),
                    "byte_start": bs,
                    "byte_end": be,
                    "label_id": label_id,
                    "label": labels[label_id],
                    "score": f32(row[label_id]),
                    "probs": [f32(v) for v in row],
                }
            )

        # Word grouping, the way transformers' `aggregate_words` does it: a
        # word is a maximal run that starts with a token the fast tokenizer
        # did not mark as a continuation (`len(token) != len(text[start:end])`
        # is transformers' own sub-word test for WordPiece). The provider's
        # aggregation is word-aligned, so this is the reference for
        # `TURBO_AGGREGATE_NONE`, which transformers' pipeline has no
        # equivalent of (its "none" is per token).
        words = []
        for t in tokens:
            if t["special"]:
                continue
            surface = text.encode("utf-8")[t["byte_start"]:t["byte_end"]].decode("utf-8")
            is_subword = len(t["token"]) != len(surface)
            if not is_subword or not words:
                words.append({
                    "byte_start": t["byte_start"],
                    "byte_end": t["byte_end"],
                    "text": surface,
                    "first_column": t["column"],
                    "n_tokens": 1,
                    "first_label": t["label"],
                    "first_score": t["score"],
                    "max_label": t["label"],
                    "max_score": t["score"],
                })
            else:
                w = words[-1]
                w["byte_end"] = t["byte_end"]
                w["text"] = text.encode("utf-8")[w["byte_start"]:w["byte_end"]].decode("utf-8")
                w["n_tokens"] += 1
                if t["score"] > w["max_score"]:
                    w["max_score"] = t["score"]
                    w["max_label"] = t["label"]

        aggregation = {}
        for s in strategies:
            spans = []
            for ent in pipes[s](text):
                bs, be = byte_offsets(text, int(ent["start"]), int(ent["end"]))
                spans.append(
                    {
                        "byte_start": bs,
                        "byte_end": be,
                        "text": text.encode("utf-8")[bs:be].decode("utf-8"),
                        "entity": ent["entity_group"],
                        "score": f32(ent["score"]),
                    }
                )
            aggregation[s] = spans

        cases.append(
            {
                "id": name,
                "text": text,
                "n_tokens": len(ids),
                "input_ids": ids,
                "tokens": tokens,
                "words": words,
                "aggregation": aggregation,
            }
        )

    out = provenance(
        args, torch, transformers, np, NER_REPO, args.ner_revision,
        f"weights from the hub; tokenizer (vocab.txt, cased) and config from the local checkout {src}",
        {
            "schema": "turbo-reference-token-classify/1",
            "task": "token_classify",
            "labels": labels,
            "activation": "softmax",
            "tagging": "BIO",
            "tokenization": {"truncation": "longest_first", "max_length": args.ner_max_seq},
            "aggregation_note": (
                "`aggregation` holds the spans transformers' token-classification "
                "pipeline returns for aggregation_strategy simple, first and max, with "
                "the pipeline's character offsets converted to byte offsets in `text`. "
                "`entity` is the pipeline's entity_group, that is the label with its "
                "BIO prefix stripped. `words` is the same word grouping applied to "
                "the per-token labels, which is what a word-aligned provider "
                "reports for TURBO_AGGREGATE_NONE; transformers' own \"none\" is "
                "per token and has no word-aligned equivalent."
            ),
            "config_sha256": sha256(src / "config.json"),
            "vocab_sha256": sha256(src / "vocab.txt"),
        },
    )
    out["cases"] = cases
    return "reference_token_classify/bert_base_ner.json", out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default=socket.gethostname())
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p.add_argument("--models", type=Path, default=Path(os.path.expanduser("~/opt/models")))
    p.add_argument("--out", type=Path, default=REPO / "testdata")
    p.add_argument("--only", choices=["rerank", "classify", "token_classify"], action="append")
    p.add_argument("--rerank-max-seq", type=int, default=256)
    p.add_argument("--classify-max-seq", type=int, default=256)
    p.add_argument("--ner-max-seq", type=int, default=512)
    args = p.parse_args()

    import numpy as np
    import torch
    import transformers

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    transformers.set_seed(0)

    args.rerank_revision = resolve_revision(RERANK_REPO)
    args.classify_revision = resolve_revision(CLASSIFY_REPO)
    args.ner_revision = resolve_revision(NER_REPO)

    wanted = args.only or ["rerank", "classify", "token_classify"]
    gens = {"rerank": gen_rerank, "classify": gen_classify, "token_classify": gen_token_classify}
    for name in wanted:
        print(f"generating {name}", file=sys.stderr)
        rel, doc = gens[name](args, torch, transformers, np)
        path = args.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {path} ({path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
