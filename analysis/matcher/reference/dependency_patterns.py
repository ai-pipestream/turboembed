"""Pinned spaCy 3.8.7 dependency oracle on UD EWT r2.16 gold annotations.
Usage: python dependency_patterns.py EWT.conllu OUTPUT.json
No training/model substitution. Original text and gold dependency heads retained.
"""
import hashlib
import json
import pathlib
import sys
import spacy
from spacy.tokens import Doc
from spacy.matcher import DependencyMatcher

assert spacy.__version__ == "3.8.7"
corpus, output = map(pathlib.Path, sys.argv[1:])
digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
assert digest == "3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b"
ops = [">", "<", ">>", "<<", ".", ".*", ";", ";*", "$+", "$-", "$++", "$--",
       ">+", ">-", ">++", ">--", "<+", "<-", "<++", "<--"]
rules = [{"label": op, "pattern": [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {}},
    {"LEFT_ID": "a", "REL_OP": op, "RIGHT_ID": "b", "RIGHT_ATTRS": {}}]} for op in ops]
# Published pattern4, unchanged, spaCy release-v3.8.7 matcher test fixture.
rules.append({"label": "published", "pattern": [
    {"RIGHT_ID": "jumped", "RIGHT_ATTRS": {"ORTH": "jumped"}},
    {"LEFT_ID": "jumped", "REL_OP": ">", "RIGHT_ID": "fox", "RIGHT_ATTRS": {"ORTH": "fox"}}]})
# Multi-node branching, node reuse, DEP sets and shared lexical predicates.
rules += [{"label": "branch", "pattern": [
    {"RIGHT_ID": "v", "RIGHT_ATTRS": {"POS": "VERB"}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "n", "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "obj"]}}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "again", "RIGHT_ATTRS": {"DEP": {"NOT_IN": ["punct"]}}}]},
    {"label": "chain", "pattern": [
    {"RIGHT_ID": "n", "RIGHT_ATTRS": {"POS": "NOUN"}},
    {"LEFT_ID": "n", "REL_OP": "<", "RIGHT_ID": "v", "RIGHT_ATTRS": {}},
    {"LEFT_ID": "v", "REL_OP": ">", "RIGHT_ID": "back", "RIGHT_ATTRS": {"POS": "NOUN"}}]},
    {"label": "fuzzy", "pattern": [
    {"RIGHT_ID": "a", "RIGHT_ATTRS": {"LOWER": {"FUZZY1": "said"}}},
    {"LEFT_ID": "a", "REL_OP": ">", "RIGHT_ID": "b", "RIGHT_ATTRS": {"MORPH": {"INTERSECTS": ["Number=Sing"]}}}]}]
nlp = spacy.blank("en")
matcher = DependencyMatcher(nlp.vocab, validate=True)
for rule in rules:
    matcher.add(rule["label"], [rule["pattern"]])
cases = []
for block in corpus.read_text().split("\n\n"):
    lines = block.splitlines()
    text = next((l[9:] for l in lines if l.startswith("# text = ")), None)
    rows = [l.split("\t") for l in lines if l and not l.startswith("#")]
    if not text or not 2 <= len(rows) <= 30 or any(not r[0].isdigit() for r in rows):
        continue
    words = [r[1] for r in rows]
    at, spaces = 0, []
    for word in words:
        if not text.startswith(word, at):
            break
        at += len(word)
        space = at < len(text) and text[at] == " "
        spaces.append(space)
        at += int(space)
    else:
        if at != len(text):
            continue
        heads = [int(r[6]) - 1 if r[6] != "0" else i for i, r in enumerate(rows)]
        doc = Doc(nlp.vocab, words=words, spaces=spaces, heads=heads, deps=[r[7] for r in rows],
                  pos=[r[3] for r in rows], tags=[r[4] for r in rows],
                  lemmas=[r[2] for r in rows], morphs=[r[5] for r in rows])
        assert doc.text == text
        hits = sorted((nlp.vocab.strings[label], ids) for label, ids in matcher(doc))
        cases.append({"text": text, "heads": heads, "deps": [r[7] for r in rows],
            "pos": [r[3] for r in rows], "morph": [str(t.morph) for t in doc],
            "spans": [[len(text[:t.idx].encode()), len(text[:t.idx+len(t.text)].encode())] for t in doc],
            "hits": hits})
        if len(cases) == 200:
            break
assert len(cases) == 200
# Twenty paired original sentences exercise independent roots in one document.
for first, second in zip(cases[:20], cases[20:40]):
    text = first["text"] + " " + second["text"]
    words, spaces, heads, deps, pos, morph = [], [], [], [], [], []
    for part_index, part in enumerate([first, second]):
        raw = part["text"].encode()
        base = len(words)
        for i, (a, b) in enumerate(part["spans"]):
            words.append(raw[a:b].decode())
            spaces.append((b < len(raw) and raw[b:b+1] == b" ") or (part_index == 0 and i == len(part["spans"])-1))
        heads.extend(base + h for h in part["heads"])
        deps.extend(part["deps"])
        pos.extend(part["pos"])
        morph.extend(part["morph"])
    doc = Doc(nlp.vocab, words=words, spaces=spaces, heads=heads, deps=deps, pos=pos, morphs=morph)
    assert doc.text == text
    cases.append({"text": text, "heads": heads, "deps": deps, "pos": pos, "morph": morph,
        "spans": [[len(text[:t.idx].encode()), len(text[:t.idx+len(t.text)].encode())] for t in doc],
        "hits": sorted((nlp.vocab.strings[label], ids) for label, ids in matcher(doc))})
counts = {op: sum(sum(label == op for label, _ in c["hits"]) for c in cases) for op in ops}
assert all(counts.values())
output.write_text(json.dumps({"spacy": spacy.__version__, "corpus_sha256": digest,
                             "rules": [{"label": r["label"], "pattern": json.dumps(r["pattern"])} for r in rules], "cases": cases}, ensure_ascii=False) + "\n")
print(json.dumps({"sentences": len(cases), "tokens": sum(len(c["heads"]) for c in cases),
                  "matches": sum(len(c["hits"]) for c in cases), "operators": counts}, sort_keys=True))
