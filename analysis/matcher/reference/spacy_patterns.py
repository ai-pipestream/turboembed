"""spaCy 3.8.7 oracle over UD English EWT r2.16 gold token annotations.

Arguments: EWT.conllu DATA_DIR OUTPUT.json. DATA_DIR contains the unchanged
Explosion drugs_patterns.jsonl and this project's operators.jsonl.
Run in the pinned reference container; no model substitution or mocking.
"""
import hashlib
import json
import pathlib
import sys
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher

assert spacy.__version__ == "3.8.7"
corpus, data, output = map(pathlib.Path, sys.argv[1:])
rules = []
for name in ["drugs_patterns.jsonl", "operators.jsonl"]:
    rules.extend(json.loads(line) for line in (data / name).read_text().splitlines())
nlp = spacy.blank("en")
matcher = Matcher(nlp.vocab, validate=True)
for rule in rules:
    matcher.add(rule["label"], [rule["pattern"]])
sentences = []
for block in corpus.read_text().split("\n\n"):
    lines = block.splitlines()
    text = next((l[9:] for l in lines if l.startswith("# text = ")), None)
    rows = [l.split("\t") for l in lines if l and not l.startswith("#")]
    if not text or not rows or any(not r[0].isdigit() for r in rows):
        continue
    words = [r[1] for r in rows]
    at, starts, spaces = 0, [], []
    for word in words:
        if not text.startswith(word, at):
            break
        starts.append(at)
        at += len(word)
        space = at < len(text) and text[at] == " "
        spaces.append(space)
        at += int(space)
    else:
        if at != len(text):
            continue
        doc = Doc(nlp.vocab, words=words, spaces=spaces,
                  pos=[r[3] for r in rows], tags=[r[4] for r in rows],
                  lemmas=[r[2] for r in rows], morphs=[r[5] for r in rows])
        assert doc.text == text
        hits = sorted(set((nlp.vocab.strings[label], a, b) for label, a, b in matcher(doc)))
        sentences.append((doc, rows, hits))
# Every drug-bearing sentence, plus twenty short real sentences to test
# inflected words and non-matches. No selection on Rust's output.
selected = [s for s in sentences if any(h[0] == "DRUG" for h in s[2])]
selected += [s for s in sentences if len(s[0]) < 18 and not any(h[0] == "DRUG" for h in s[2])][:20]
selected += [s for s in sentences if len(s[0]) < 20 and any(h[0] in ("FUZZY", "FUZZY1", "FUZZY_IN") for h in s[2]) and s not in selected][:10]
cases = []
for doc, rows, hits in selected:
    offsets = [(len(doc.text[:t.idx].encode()), len(doc.text[:t.idx+len(t.text)].encode())) for t in doc]
    cases.append({"text": doc.text, "spans": offsets,
                  "pos": [r[3] for r in rows], "tags": [r[4] for r in rows],
                  "lemmas": [r[2] for r in rows], "morph": [str(t.morph) for t in doc],
                  "hits": hits})
output.write_text(json.dumps({"spacy": spacy.__version__,
    "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(), "cases": cases}, ensure_ascii=False) + "\n")
print(json.dumps({"spacy":spacy.__version__, "corpus_sha256":hashlib.sha256(corpus.read_bytes()).hexdigest(),
                  "sentences":len(selected), "tokens":sum(len(d) for d,_,_ in selected),
                  "matches":sum(len(h) for _,_,h in selected),
                  "drug_matches":sum(sum(l=="DRUG" for l,_,_ in h) for _,_,h in selected)}))
