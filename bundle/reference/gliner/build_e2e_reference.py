"""End-to-end GLiNER reference for multi-sentence documents, using the upstream
GLiNER 0.2.21 heads as build_reference.py does.

Usage: build_e2e_reference.py <checkpoint-dir> <out.json> [reps]

The checkpoint directory holds the wheel, pytorch_model.bin, spm.model and
encoder-config.json (see docs/gliner.md). Documents are built from the fixed
sentences below. Windows follow the native rule independently: whole
sentences while the window has at most 384 words and 512 pieces including
the label prompt and final separator. Each window is scored and decoded on its
own. With reps > 0 each benchmark document is also timed end to end on one
thread (tokenize, window, encode, heads, decode); the median is recorded.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import zipfile
import torch
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2TokenizerFast

SENTENCES = [
    "Alice Johnson joined Microsoft in Seattle in March 2019 .",
    "She later moved to Paris to lead the European research team .",
    "The United Nations held its climate summit in Glasgow last November .",
    "Barack Obama spoke with Angela Merkel about trade between Germany and the United States .",
    "Toyota opened a new factory near Nagoya on 4 April 2021 .",
    "Dr. Priya Raman of Stanford University published the study in Nature .",
    "The Amazon River flows through Brazil , Peru and Colombia .",
    "Zoë Martín visited the Louvre with her brother Mateo on Sunday .",
    "Goldman Sachs reported record profits for the third quarter of 2023 .",
    "The European Central Bank in Frankfurt kept interest rates unchanged .",
    "Lionel Messi signed with Inter Miami after leaving Paris Saint-Germain .",
    "NASA launched the Artemis mission from Cape Canaveral in Florida .",
    "In 1969 , Neil Armstrong became the first person to walk on the Moon .",
    "The Tokyo Stock Exchange fell sharply on Monday morning .",
    "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911 .",
    "Apple announced the new iPhone at its headquarters in Cupertino , California .",
    "The World Health Organization warned about a new outbreak in West Africa .",
    "Satya Nadella met with Indian officials in New Delhi on Tuesday .",
    "The Golden Gate Bridge in San Francisco opened in May 1937 .",
    "Oxford University Press released a revised edition of the dictionary .",
    "Greta Thunberg addressed the European Parliament in Strasbourg .",
    "Samsung Electronics is headquartered in Suwon , South Korea .",
    "The Berlin Wall fell on 9 November 1989 .",
    "Jeff Bezos founded Amazon in a garage in Bellevue , Washington .",
    "The Mississippi River runs from Minnesota to the Gulf of Mexico .",
    "Professor Chen Wei of Tsinghua University gave a keynote in Beijing .",
    "The International Monetary Fund lowered its growth forecast for Argentina .",
    "Serena Williams won her first Wimbledon title in 2002 .",
    "Siemens and Bosch both have large offices in Munich .",
    "The Treaty of Versailles was signed in France in June 1919 .",
    "Elon Musk said Tesla would build a plant outside Berlin .",
    "Rome hosted the Olympic Games in 1960 .",
]
# Many pieces per word, so the 512-piece limit closes windows before 384 words.
HEAVY = [
    "Model XJ-9000-B47 was certified under ISO-27001:2022 by TÜV-Süd-Rheinland in Großenhain .",
    "Kraftfahrzeughaftpflichtversicherung premiums at Württembergische-Gemeinde-Versicherung rose 7.25% .",
    "Ngozi Okonjo-Iweala met Szczepan Wróblewski-Kowalczyk at the Rijksmuseum in Amsterdam .",
]
LABELS = ["person", "organization", "location", "date"]
MAX_WORDS, MAX_PIECES, MAX_WIDTH = 384, 512, 12


def documents():
    s = SENTENCES
    docs = {
        "short-3": s[0:3],
        "short-utf8": [s[7], s[10], s[13], s[22]],
        "short-2": s[15:17],
        "long-2x": s + s[:6],
        "long-6x": [s[(i * 7) % len(s)] for i in range(len(s) * 3)],
        "long-pieces": (HEAVY + s[:3]) * 12,
    }
    bench = {f"bench-{n}": [] for n in (100, 400, 1600)}
    for name, sentences in bench.items():
        target, i = int(name.split("-")[1]), 0
        while sum(len(x.split()) for x in sentences) < target:
            sentences.append(s[i % len(s)])
            i += 1
    return docs, bench


root = Path(sys.argv[1])
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.set_num_threads(1)
wheel = root / "gliner-0.2.21-py3-none-any.whl"
pkg = types.ModuleType("gliner_reference")
pkg.__path__ = []
sys.modules[pkg.__name__] = pkg
with zipfile.ZipFile(wheel) as z:
    for name in ["layers", "span_rep", "utils", "decoder"]:
        mod = types.ModuleType(f"gliner_reference.{name}")
        mod.__package__ = "gliner_reference"
        sys.modules[mod.__name__] = mod
        sub = "decoding" if name in ["utils", "decoder"] else "modeling"
        exec(compile(z.read(f"gliner/{sub}/{name}.py"), f"gliner/{name}.py", "exec"), mod.__dict__)
from gliner_reference.layers import create_projection_layer, LstmSeq2SeqEncoder  # noqa: E402
from gliner_reference.span_rep import SpanMarkerV0  # noqa: E402
from gliner_reference.decoder import SpanDecoder  # noqa: E402

weights = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)


def part(prefix):
    return {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}


ec = json.loads((root / "encoder-config.json").read_text())
ec["vocab_size"] = weights["token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight"].shape[0]
encoder = DebertaV2Model(DebertaV2Config(**ec)).eval()
encoder.load_state_dict(part("token_rep_layer.bert_layer.model."), strict=True)
projection = torch.nn.Linear(768, 512).eval()
projection.load_state_dict(part("token_rep_layer.projection."), strict=True)
rnn = LstmSeq2SeqEncoder(types.SimpleNamespace(hidden_size=512)).eval()
rnn.load_state_dict(part("rnn."), strict=True)
span = SpanMarkerV0(512, MAX_WIDTH).eval()
span.load_state_dict(part("span_rep_layer.span_rep_layer."), strict=True)
prompt = create_projection_layer(512, 0.4).eval()
prompt.load_state_dict(part("prompt_rep_layer."), strict=True)
tok = DebertaV2TokenizerFast(vocab_file=str(root / "spm.model"))
tok.add_tokens(["[FLERT]", "<<ENT>>", "<<SEP>>"])
assert tok.convert_tokens_to_ids("<<ENT>>") == 128002
PROMPT = sum((["<<ENT>>", label] for label in LABELS), []) + ["<<SEP>>"]


def windows(sentences):
    """Word ranges of whole sentences, packed as the native annotator does."""
    words = [w for s in sentences for w in s.split()]
    batch = tok(PROMPT + words, is_split_into_words=True, truncation=False)
    word_ids = batch.word_ids()
    pieces = [0] * len(words)
    for w in word_ids:
        if w is not None and w >= len(PROMPT):
            pieces[w - len(PROMPT)] += 1
    prefix = sum(1 for w in word_ids if w is None or w < len(PROMPT)) - 1
    out, start, at, count, used = [], 0, 0, 0, prefix
    for s in sentences:
        n = len(s.split())
        size = sum(pieces[at:at + n])
        if count + n > MAX_WORDS or used + size + 1 > MAX_PIECES:
            out.append((start, at))
            start, count, used = at, 0, prefix
        count, used, at = count + n, used + size, at + n
    out.append((start, at))
    return words, out


def score(words):
    batch = tok(PROMPT + words, is_split_into_words=True, return_tensors="pt", truncation=False)
    ids = batch["input_ids"][0].tolist()
    word_ids = batch.word_ids()
    positions = [word_ids.index(i + len(PROMPT)) for i in range(len(words))]
    markers = [i for i, t in enumerate(ids) if t == 128002]
    h = projection(encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state)
    w = rnn(h[:, positions], torch.ones((1, len(words)), dtype=torch.bool))
    p = prompt(h[:, markers])
    n = len(words)
    indices = torch.tensor([[(i, min(i + k, n - 1)) for i in range(n) for k in range(MAX_WIDTH)]])
    logits = torch.einsum("blkd,bcd->blkc", span(w, indices), p)[0]
    return ids, logits


def decode(words, logits, flat):
    labels = {i + 1: label for i, label in enumerate(LABELS)}
    selected = SpanDecoder(None).decode([words], labels, logits.unsqueeze(0), flat_ner=flat, threshold=0.5)[0]
    return [[a, b + 1, LABELS.index(label), s] for a, b, label, s in selected]


def run(sentences, keep):
    words, ranges = windows(sentences)
    out = []
    for a, b in ranges:
        ids, logits = score(words[a:b])
        w = dict(start=a, end=b, decoded={m: decode(words[a:b], logits, m == "flat") for m in ["flat", "nested"]})
        if keep:
            n = b - a
            flat = [float(logits[i, k, c]) for i in range(n) for k in range(min(MAX_WIDTH, n - i)) for c in range(4)]
            w.update(ids=ids, logits=flat)
        out.append(w)
    return out


docs, bench = documents()
result = dict(
    checkpoint="urchade/gliner_small-v2.1",
    revision="4e091416cf7c3481db542c2a3d26156916f3a47f",
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    torch=torch.__version__,
    labels=LABELS,
    threshold=0.5,
    documents=[],
    bench=[],
)
with torch.no_grad():
    for name, sentences in docs.items():
        result["documents"].append(dict(name=name, sentences=sentences, windows=run(sentences, True)))
    for name, sentences in bench.items():
        entry = dict(name=name, sentences=sentences, words=sum(len(s.split()) for s in sentences))
        entry["ranges"] = windows(sentences)[1]
        if reps:
            run(sentences, False)
            times = []
            for _ in range(reps):
                t = time.perf_counter()
                entry["windows"] = len(run(sentences, False))
                times.append((time.perf_counter() - t) * 1e3)
            entry["python_median_ms"] = sorted(times)[reps // 2]
            print(f"python {name} words={entry['words']} windows={entry['windows']} median_ms={entry['python_median_ms']:.1f}")
        result["bench"].append(entry)
Path(sys.argv[2]).write_text(json.dumps(result) + "\n")
print("documents", len(result["documents"]), "windows", [len(d["windows"]) for d in result["documents"]])
