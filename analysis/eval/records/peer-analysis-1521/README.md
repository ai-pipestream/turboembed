# Peer analysis records (#1521)

Peer and turbo records scored by `turbo-eval analysis` against the
UD English-EWT test file: `spacy-en-ewt-test.json`,
`spacy-lg-en-ewt-test.json`, `spacy-trf-en-ewt-test.json`,
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
- Peer: spaCy 3.8.7 with `en_core_web_lg` 3.8.0, same container with
  the model build arguments changed (local image id at record time
  `sha256:ac75f6a5866cad276de5aef167ccbe27cfb82d49a3f5a027e63090f94315ac7b`).
  Predictions SHA-256:
  `488c7b5e51c08e6f1b612e6f139514e472ca85e544f22cf52130f99f9c21f785`.
  Sentences 82.52 F1, UPOS 91.74. Per the issue, spaCy is compared
  on sentences and UPOS here (and NER once entity inputs land); the
  remaining columns stay in the JSON for completeness under the
  label-scheme caveat below.
- Peer: spaCy 3.8.7 with `en_core_web_trf` 3.8.0 on CPU torch 2.6.0
  (curated-transformers 0.1.1), in the pinned container at
  `analysis/eval/reference/peers/spacy-trf/` (local image id at
  record time
  `sha256:bd8fb6da4c3c2dc05f9477be3adc78be4ed5b332bb1552f45e0f2bcd2fe7a286`).
  Predictions SHA-256:
  `9c617b026c9b1489fda5e9653e17cd238d67c18ede50fbb1a3ceb5c0e5fa1126`.
  Sentences 85.41 F1, UPOS 92.96, under the same comparison rule.
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

## Multilingual UD splits

The same protocol on the four further UD test splits the issue
names, fetched from the UniversalDependencies repositories at the
pinned commits (gold files are not committed; the bench pins check
their hashes):

- `es_ancora-ud-test.conllu`, UD_Spanish-AnCora
  `20adddbfcdd773c6dc97ba48ea11ca9364e74185`, SHA-256
  `679f1c05ca654aa26801c80319df45f8f97606718fb534b91755b48f3e67fa6f`
  (1721 sentences, 53622 tokens).
- `ca_ancora-ud-test.conllu`, UD_Catalan-AnCora
  `7f3d23e85abc0656e8e88535d28174b030eb4d20`, SHA-256
  `7cd23807c1f8b090959b82fcab566c6312fc048c29bc9db5ff0bf794c9b98633`
  (1846 sentences, 59610 tokens).
- `fi_ftb-ud-test.conllu`, UD_Finnish-FTB
  `2dd197c1f7c4b9b6d69be1e3f826162a4443af96`, SHA-256
  `f7b738e86bf79c4c2db18ad8141649b175846d9f57c452ac1ca8671cc34f954e`
  (1867 sentences, 16286 tokens).
- `it_markit-ud-test.conllu`, UD_Italian-MarkIT
  `e210079df0aea331ec1dcaa90648d815ce94c3ce`, SHA-256
  `9048ed0064e3989e2c37377755a28c25c7e462b1ad916e5808dc0a7ce95fc3fc`
  (340 sentences, 10196 tokens).

Stanza runs as shipped per language, its default packages with no
treebank override, which matters for Finnish: the default Finnish
package is TDT while the gold split is FTB, so the Finnish XPOS and
parsing columns are cross-treebank numbers, not a like-for-like
parser ranking. Spanish and Italian defaults are their combined
packages. Prediction SHA-256s: es
`8c10d264e261371ce2d61f785a7e4c520cfb8daee8e03d3f7499922ba3b5325c`,
ca
`ed021a5eff707018e67e686ff267d9f6efee7ea04bd351399f7296e404d5babb`,
fi
`0fa9451cc32d087bffa8cb1c6090190562aae224d6e8393478c0271a266aba1d`,
it
`c464d04869a1c23b46acfbfec5e1a3485537eeb1968333197bb49740f7e50609`.
Headlines (Stanza LAS; our rule tier's sentence F1 in the same
records): es 86.71 LAS, rule sentences 97.19; ca 87.63, rule 99.64;
fi 60.06 cross-treebank, rule 82.19; it 77.93, rule 87.56 against
Stanza's own 72.90 sentence F1 on MarkIT.

Trankit records so far on the same splits (`trankit-fi-ftb-test.json`,
`trankit-it-markit-test.json`): one pinned image per language from
`analysis/eval/reference/peers/trankit/` with `TRANKIT_LANG` set
(finnish image
`sha256:1390e26ed600ed2c48e05b3c0a767cc71438d9b50fa6ea74b3aaf44a9682d644`,
italian image
`sha256:7854ecc137aee9f7b4967c07592041ed43e1263c5c878d0f62bfd614bf60de99`;
prediction SHA-256s fi
`f20892ea65cadf2c2e2aa6d2fba666bdba00a402fc4dc012879118ec40fa970f`,
it
`992ea3e6abf9e8f9c700f7eddd85961b183d1feaf6c093e7285131b57da3dbad`).
Finnish: sentences 95.97, LAS 63.85, with the same FTB XPOS
convention gap as Stanza (5.91). Italian: sentences 99.70, LAS
82.06; the segmentation gap against Stanza on MarkIT is the model,
not the protocol, since both peers saw the same joined text.

## Reproduce

```sh
docker build -t turbo-peer-spacy:3.8.7-sm-3.8.0 analysis/eval/reference/peers/spacy
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-spacy:3.8.7-sm-3.8.0 /gold/en_ewt-ud-test.conllu /out/spacy-sm.conllu
docker build --build-arg MODEL=en_core_web_lg --build-arg MODEL_VERSION=3.8.0 \
  -t turbo-peer-spacy:3.8.7-lg-3.8.0 analysis/eval/reference/peers/spacy
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-spacy:3.8.7-lg-3.8.0 /gold/en_ewt-ud-test.conllu /out/spacy-lg.conllu en_core_web_lg
docker build -t turbo-peer-spacy:3.8.7-trf-3.8.0 analysis/eval/reference/peers/spacy-trf
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-spacy:3.8.7-trf-3.8.0 /gold/en_ewt-ud-test.conllu /out/spacy-trf.conllu en_core_web_trf
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

# Multilingual Stanza (per language; es shown)
docker build --build-arg STANZA_LANG=es \
  -t turbo-peer-stanza:1.10.1-es analysis/eval/reference/peers/stanza
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-stanza:1.10.1-es /gold/es_ancora-ud-test.conllu /out/stanza-es.conllu es
turbo-eval analysis --gold <gold-dir>/es_ancora-ud-test.conllu \
  --sha256 679f1c05ca654aa26801c80319df45f8f97606718fb534b91755b48f3e67fa6f \
  --pred stanza-1.10.1-es=<out-dir>/stanza-es.conllu \
  --sentences rule --out <new-record>.json

# Multilingual Trankit (per language; fi shown)
docker build --build-arg TRANKIT_LANG=finnish \
  -t turbo-peer-trankit:1.1.2-finnish analysis/eval/reference/peers/trankit
docker run --rm --network none -v <gold-dir>:/gold:ro -v <out-dir>:/out \
  turbo-peer-trankit:1.1.2-finnish /gold/fi_ftb-ud-test.conllu /out/trankit-fi.conllu finnish
```

Recorded 2026-10-28 under load average about 31 to 36 (accuracy
record; no timing claimed).
