# Peer analysis records (#1521)

Peer and turbo records scored by `turbo-eval analysis` against the
UD English-EWT test file: `spacy-en-ewt-test.json`,
`stanza-en-ewt-test.json`, `wtpsplit-en-ewt-test.json`,
`trankit-en-ewt-test.json`, and `turbo-sat-en-ewt-test.json`.

## Inputs and pins

- Gold: `en_ewt-ud-test.conllu`, SHA-256
  `fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a`
  (2077 sentences, 25094 tokens, 316 documents). The scorer refuses
  any other bytes under this pin.
- Peer: spaCy 3.8.7 with `en_core_web_sm` 3.8.0, run in the pinned
  container built from `analysis/eval/reference/peers/spacy/`
  (base image pinned by digest; local image id at record time
  `sha256:efad301914b0dc8f5be1f2f7fe02ebb011665aba7da00820055730108fecbb26`).
- spaCy predictions SHA-256:
  `a88fa76835915f87c8b0d7e75fd222383d6e0e81a683ac10c6ce44c8c4f1882e`.
- Peer: Stanza 1.10.1 with the English default models
  (`tokenize,mwt,pos,lemma,depparse`) on CPU torch 2.6.0, in the
  pinned container built from `analysis/eval/reference/peers/stanza/`
  (local image id at record time
  `sha256:cf9ee7acb0b86e8f96510481bc171dd9bac4ea4a67dab3ab61a3a443024d54ae`).
  Stanza's English default package trains on the UD EWT training
  split, so this treebank is in-domain for it.
- Stanza predictions SHA-256:
  `ae0e58aa2c5045709bae07f100556286eaccf6570ebe28cb54c5314f0a801f90`.
- Peer: wtpsplit 2.2.2 with the `sat-3l` SaT weights on CPU torch
  2.6.0, in the pinned container built from
  `analysis/eval/reference/peers/wtpsplit/` (local image id at record
  time `sha256:e9f66767399d34097ae89a0d6cb16af96dc799327f51fe4d1f045cc84cd14d32`).
  wtpsplit predicts sentences only; it is scored through the bench's
  sentence-list input (`--sents`), whose sentences must tile each
  gold document's joined text in order.
- wtpsplit predictions SHA-256:
  `1aa2702be1c99bcf1743337cfa5fc430ce4c173579f9265091e609a6e2c04841`.
  Sentence F1 85.30 in joined-text coordinates (our rule tier in the
  same record: 73.55).
- Peer: Trankit 1.1.2 with the English `xlm-roberta-base` pipeline
  on CPU torch 2.0.1, in the digest-pinned Python 3.10 container at
  `analysis/eval/reference/peers/trankit/` (local image id at record
  time `sha256:693b004911d6fb9bc1c70b4ed8f833e5b495bcc315e3f33cfde85ba9680cf540`).
  Trankit's own model host was unreachable from the recording
  machine, so the same model zip was baked in from the `uonlp`
  mirror pinned by revision; the Dockerfile records that revision.
- Trankit predictions SHA-256:
  `2c14aae8c6ae8d90a03996dcb3a48221916ccb0195c2be2dd8ff4891522314df`.
  Headlines: sentences 93.43 F1, tokens 98.42, UPOS 94.94, XPOS
  95.38, feats 84.51, lemma 93.17, UAS 86.82, LAS 83.39.
- Ours: the SaT sentence tier (`--sentences sat`) on CPU with model
  precision, bundle `segment-any-text/sat-1l-sm` revision 1, source
  commit `00a783439c201b20ee91ee1624dc817f1bf0e578`. Bundle weights
  SHA-256
  `81eb7a4fcd4dab85c50866c7e2cf58f3d848e1c830c9207d132277110489ea0c`;
  tokenizer SHA-256
  `a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c`;
  manifest SHA-256
  `c3a0dbcc13b9f5eec8154563eb3403b8f859254e69fcb202204296f734bcb506`.
  Sentence F1 82.64 in joined-text coordinates (1414 correct of 1661
  selected against 1761 target boundaries). This is the one-layer
  small SaT variant; the wtpsplit peer above used the three-layer
  weights, so the two sentence rows are not the same model in two
  runtimes. The record's throughput field is one in-process timing,
  not a 20-run receipt.

## Conditioning

Raw-text end to end: the peer receives each gold document's `# text`
sentences joined with one space and does its own tokenization and
sentence segmentation; no gold token or sentence boundaries reach it.
The scorer aligns predicted and gold token streams by form with
bounded resync; field metrics count over gold tokens (see the
`conditioning` list inside the record).

One reading note: the spaCy English pipeline predicts its own
dependency label scheme and attaches adpositions as heads, where UD
attaches the nominal. Its UAS/LAS here are therefore structurally
lower than a UD-native system's (Stanza, Trankit) and are not a
like-for-like parser ranking; tokens, UPOS, XPOS, feats, lemma, and
sentence boundaries are the comparable columns for this peer.

## Reproduce

```sh
docker build -t turbo-peer-spacy:3.8.7-sm-3.8.0 analysis/eval/reference/peers/spacy
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-spacy:3.8.7-sm-3.8.0 /gold/en_ewt-ud-test.conllu /out/spacy-sm.conllu
turbo-eval analysis --gold <gold-dir>/en_ewt-ud-test.conllu \
  --sha256 fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a \
  --pred spacy-3.8.7-en_core_web_sm-3.8.0=<out-dir>/spacy-sm.conllu \
  --sentences rule --out <new-record>.json

docker build -t turbo-peer-stanza:1.10.1-en analysis/eval/reference/peers/stanza
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-stanza:1.10.1-en /gold/en_ewt-ud-test.conllu /out/stanza.conllu

docker build -t turbo-peer-trankit:1.1.2-en analysis/eval/reference/peers/trankit
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-trankit:1.1.2-en /gold/en_ewt-ud-test.conllu /out/trankit.conllu

turbo-eval analysis --gold <gold-dir>/en_ewt-ud-test.conllu \
  --sha256 fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a \
  --system turbo-sat-1l-sm-cpu --sentences sat \
  --sat-bundle <sat-1l-sm-bundle-dir> --out <new-record>.json
```

Recorded 2026-10-28 under load average about 31 to 36 (accuracy
record; no timing claimed).
