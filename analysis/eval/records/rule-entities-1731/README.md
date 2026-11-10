# Rule date, time, money and percent entities against gold (#1731)

Exact-span F1 of the deterministic `date`, `time`, `money` and `percent`
entities (`RuleEntityProjector` and `ConsensusEntityMerger` over the
temporal, time, money and quantity layers, the pipeline of
`analysis/query/tests/rule_entities_gold.rs`) against gold, beside spaCy
`en_core_web_trf` and Stanza on the same text. A match is the same UTF-8
byte span and the same type; each document counts its distinct spans.
Intervals are the peer's F1 minus the rules', over 1,000 paired bootstrap
resamples of whole documents (seed 1731, 2.5th to 97.5th percentile).

The only references are gold, the rules themselves and the two peers (#1509).

## MASC held-out documents, `date`

MASC 3.0.0, corpus SHA-256 `445449f5...afd` (4,425 files), the recipes'
split (documents at 9 modulo 10, Twitter genre excluded): the 39 held-out
documents with a named entity layer, 443 gold `date` spans, none
discontinuous. 85 of them are among the entities MASC places in no single
sentence; spans are scored over the whole document, so they count.

| system | tp | fp | fn | precision | recall | F1 | vs rules (95%) |
|---|---|---|---|---|---|---|---|
| rules (the deterministic baseline) | 68 | 57 | 375 | 0.5440 | 0.1535 | 0.2394 | |
| spaCy 3.8.7 `en_core_web_trf` 3.8.0, `DATE` | 242 | 250 | 201 | 0.4919 | 0.5463 | 0.5176 | +0.2782 [+0.1485, +0.4465] |
| Stanza 1.10.1 OntoNotes, `DATE` | 198 | 183 | 245 | 0.5197 | 0.4470 | 0.4806 | +0.2411 [+0.1479, +0.3878] |

The MASC NE head of #1730 has no trained checkpoint yet, so it has no row.

Reading note: MASC's `date` is wider than a calendar date. The commonest
gold spans the rules miss are deictic and vague times (`now` or `Now` 19, `tonight`
10, `night` 9, `today` 6), decades and centuries (`the 1990s`, `21st
century`), durations (`next four years`, `7 years`, `ten years`), bare
years (`1990`, `1998`), weekdays, months and seasons. The rules project
only `Temporals` (days, weeks, months, quarters, years written as dates)
and clock times, and leave durations and bare numerals alone by design.
Of the 57 rule spans not in gold, 30 are one timestamp (`2008-11-10`)
repeated in the page furniture of `data/written/ficlets/1402`, which MASC
does not annotate; most of the rest are full dates (`28 Nov 2000`, `April
8, 2012`) that MASC leaves unannotated.

## The reviewed set, all four types

The four documents of `analysis/query/tests/support/rule_entities_corpus.rs`
(32 gold spans: 12 `date`, 6 `time`, 8 `money`, 6 `percent`). The set was
written for the rules and reviewed against them, so the rules' 1.0 is a
regression floor, not a ranking.

| system | date | time | money | percent | micro F1 | micro vs rules (95%) |
|---|---|---|---|---|---|---|
| rules | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | |
| spaCy `en_core_web_trf` | 0.9167 | 0.2857 | 0.2353 | 1.0000 | 0.6269 | -0.3731 [-0.5000, -0.2063] |
| Stanza OntoNotes | 0.8696 | 0.7273 | 0.3750 | 1.0000 | 0.7419 | -0.2581 [-0.4000, -0.0714] |

Reading note: the peers' `money` misses are mostly OntoNotes span
conventions, not missed amounts: OntoNotes leaves the currency symbol out
(`4,500` for `$4,500`, `75.50` for `£75.50`), and spaCy splits `$350K`.
spaCy's `time` spans take a following word (`9:30am and`) or split `5pm`.
Both peers type `3 hours` as `time`. With four documents the intervals
are wide.

## Reproduce

The peer images are built from `analysis/eval/reference/peers/spacy-trf`
and `analysis/eval/reference/peers/stanza`; the runs used image IDs
`sha256:4ec00d0f...0496` (spaCy 3.8.7, `en_core_web_trf` 3.8.0, torch as
the image reports it, 2.15.0) and `sha256:968c0e5e...b1ee6` (Stanza
1.10.1), on CPU with no network.

```sh
cargo build --release -p turbo-query --example rule_entities_gate
G=target/release/examples/rule_entities_gate
P=analysis/eval/reference/peers
$G export --set masc --masc MASC-3.0.0 --out masc.jsonl
# masc.jsonl: sha256 2389f9ebb8f0b460b532e2fd9d03d2c7477ae95173fe1ac83f89cba918300edb
$G export --set reviewed --out reviewed.jsonl
# reviewed.jsonl: sha256 5126146999e27b36265dc5aa442a2a268979bf5bc6ba5613eb50b3f371a475e7
for set in masc reviewed; do
  docker run --rm --network none -v "$PWD/$P/spacy:/s:ro" -v "$PWD:/d" --entrypoint python \
    turbo-peer-spacy-trf:3.8.7 /s/span_predict.py /d/$set.jsonl /d/$set.spacy-trf.jsonl en_core_web_trf
  docker run --rm --network none -v "$PWD/$P/stanza:/s:ro" -v "$PWD:/d" --entrypoint python \
    turbo-peer-stanza:1.10.1 /s/span_predict.py /d/$set.jsonl /d/$set.stanza.jsonl
done
$G score --set masc --masc MASC-3.0.0 --peer spacy-trf=masc.spacy-trf.jsonl \
  --peer stanza=masc.stanza.jsonl --out masc.json
$G score --set reviewed --peer spacy-trf=reviewed.spacy-trf.jsonl \
  --peer stanza=reviewed.stanza.jsonl --out reviewed.json
```

The peer prediction files of these runs are in `predictions/`; the
receipts name each by SHA-256, so `score` reproduces the tables from them
without rerunning a peer. Both peer runs together took under three
minutes on a 16-core desktop CPU.
