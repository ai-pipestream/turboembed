# Peer analysis records (#1521)

Peer and turbo records scored by `turbo-eval analysis` against the
UD English-EWT test file: `spacy-en-ewt-test.json`,
`spacy-lg-en-ewt-test.json`, `spacy-trf-en-ewt-test.json`,
`stanza-en-ewt-test.json`, `wtpsplit-en-ewt-test.json`,
`trankit-en-ewt-test.json`, and `turbo-sat-en-ewt-test.json`; the
multilingual splits and the WNUT-17 entity records are listed in
their own sections below.

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
  Sentences 80.05 F1, UPOS 91.74. Per the issue, spaCy is compared
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
  Sentences 83.48 F1, UPOS 92.96, under the same comparison rule.
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
  Headlines: sentences 92.66 F1, tokens 98.42, UPOS 94.94, XPOS
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
Spanish and Catalan complete the set (`trankit-es-ancora-test.json`,
`trankit-ca-ancora-test.json`; spanish image
`sha256:4cc6cfe20048a84e4d09b06d830e9a7c6c37b7466f882c97613e9d4f83bfd76f`,
catalan image
`sha256:4c11c004dbda85bd5e7d3d2a01b7782b64ed2262c1f6d4072875d5b3d5dda656`;
prediction SHA-256s es
`8f911076cb63e618eacf33990699fb7dc459e22c3a027300d8dcb42eaaa0a146`,
ca
`3940baab9db0ca7392785186bdf21d13b07250303238b9f65a6262823d109850`).
Spanish: sentences 99.71, LAS 83.43. Catalan: sentences 99.88, LAS
82.32. On both, Trankit's XPOS column repeats its UPOS while AnCora
gold XPOS is the EAGLES tagset, so XPOS scores exactly 0.00 there;
that is a tagset difference, not a tagging failure.

## Tokenizers: word segmentation (#1645)

The `turbo-tokenize-<split>-test.json` records score our own
tokenizers (the `--tokenize` tiers of `turbo-eval analysis`) on the
same five pinned splits and by the same form alignment as the peer
systems above: each tier receives only the per-document space-joined
`# text`. `turbo-uax29` is UAX #29 word segments with
whitespace-only segments dropped, so punctuation segments stand as
words, as UD counts them; `turbo-whitespace` is the deterministic
baseline of Unicode `White_Space` runs. Token F1 in points, as
scored by this bench (the peer numbers are the `tokens` scores of
the records named above, not the UD scorer's):

| split | gold words | turbo-uax29 | turbo-whitespace | spaCy | Stanza | Trankit |
|---|---:|---:|---:|---:|---:|---:|
| en-ewt | 25094 | 88.84 | 77.57 | 97.44 | 99.11 | 98.42 |
| es-ancora | 53622 | 95.83 | 78.02 | - | 99.94 | 97.00 |
| ca-ancora | 59610 | 88.52 | 70.69 | - | 99.96 | 95.47 |
| fi-ftb | 16286 | 98.31 | 99.77 | - | 99.77 | 99.50 |
| it-markit | 10196 | 87.60 | 91.24 | - | 99.41 | 99.75 |
| en-masc (held out) | 85292 | 94.37 | 73.00 | - | - | - |

Reading notes: the peers tokenize with models trained on UD-style
segmentation, so the gap to a rule segmenter is the price of
contractions and clitics UD splits and UAX #29 keeps whole (and, in
the other direction, of punctuation clusters UAX #29 splits). Where
the gold segmentation is itself whitespace-oriented (fi-ftb), the
whitespace baseline beats UAX #29 and matches Stanza's bench score;
it-markit goes the same way. The fi-ftb and it-markit test files are
single documents for this scorer, so their bootstrap intervals
degenerate to the point score. The token-split head's row lands
here when a trained bundle exists.

The en-masc row (`turbo-tokenize-en-masc-test.json`, via
`--gold-masc`) scores the same tiers against MASC 3.0.0's Penn
tokens on the trainer recipes' held-out split: every tenth document
in corpus order at 9 modulo 10, the Twitter genre left out, at the
recorded corpus digest `445449f5…`. Its 6384 gold sentences and
85292 gold words are the test counts the MASC POS receipts record,
which is the split-parity check. No peer system has a MASC token
row yet.

## Entities: WNUT-17

Entity records use the `--ner-gold` form on the WNUT-17 test split
(`wnut17-test.tsv`, SHA-256
`2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0`;
1287 sentences, 23394 tokens, 1079 gold entities over the six WNUT
kinds). Unlike the UD protocol, tokenization is given: each peer
labels the gold tokens and the scorer takes exact-span F1 from the
BIO labels with per-kind breakdowns and a paired bootstrap over
sentences. The spaCy and Stanza runners map their OntoNotes-style
labels to the WNUT kinds with the same documented map (PERSON to
person, ORG to corporation, GPE/LOC/FAC to location, PRODUCT to
product, WORK_OF_ART to creative-work, NORP to group; anything else
becomes O), recorded in each runner's versions output. The GLiNER
runner is zero-shot instead: it prompts with the six WNUT kinds as
words and maps its char spans back onto the gold tokens, both
documented in its runner (see below).

- `spacy-wnut17-test.json`: spaCy 3.8.7, `en_core_web_sm` 3.8.0,
  predictions SHA-256
  `480e09b72181755ba45dce59e7821a3ac5de1463b21ba088038545170170ab6a`.
  Entities 20.01 F1 (213 correct of 1050 selected against 1079
  target).
- `stanza-wnut17-test.json`: Stanza 1.10.1 English NER,
  pretokenized, predictions SHA-256
  `0d7dc83f5c9d9dd4d20c4d9e2bfcde55af0b199a25371f1e9e9ad98bb16a9acb`.
  Entities 40.09 F1 (360 of 717 selected).
- `gliner-wnut17-test.json`: GLiNER 0.2.21 with
  `urchade/gliner_small-v2.1` at
  `4e091416cf7c3481db542c2a3d26156916f3a47f` (Apache-2.0, local-only
  by its training data; the checkpoint is not shipped, only these
  predictions are recorded), zero-shot with the six kind names as
  prompt labels at the default threshold 0.5, in the pinned
  container at `analysis/eval/reference/peers/gliner/` (local image
  id at record time
  `sha256:614fedccce1528a481546dece6378f6344f1a8619c38987732266470158fda09`).
  Unlike the other two
  runners it tokenizes the joined text itself; its spans are mapped
  back onto the gold tokens by char overlap, highest score first,
  as its runner documents. Predictions SHA-256
  `9638a97e64eeeaba95d0d0287da7dc5c325296bbff4ce7efedc42d6bab2977a2`.
  Entities 42.38 F1 (562 correct of 1573 selected; person 51.45
  and location 50.13 carry it, creative-work 14.77 is its weakest
  kind). Recorded 2026-10-28.

Reading note: the spaCy and Stanza peers train on edited
newswire-style corpora while WNUT-17 is noisy user-generated text
with emerging entities, so these rows are the off-the-shelf peer
bar on a hard split, not tuned WNUT systems. GLiNER's row is
zero-shot with the gold kind names as its prompts, which is its
as-shipped use, not training on WNUT-17 either. There is no turbo
entity row yet: that waits on an entity bundle and label policy.
The GLiNER row is WNUT-17 only; a MultiCoNER row at 249980
sentences is a booked run, not a subset.

## UDPipe 2: published scores (cited, not run here)

UDPipe's code is MPL-2.0 and is not read in this repository, so
UDPipe 2 has no runner and no record JSON here; this row cites its
published numbers only. Source: the Model Performance tables of
the Universal Dependencies 2.17 models at
<https://ufal.mff.cuni.cz/udpipe/2/models> (read 2026-10-28),
models version `251125`, distributed under CC BY-NC-SA. The page
states the results are F1 scores measured by the official
`conll18_ud_eval.py` script on the testing portion of the UD 2.17
treebanks, on raw text (the rows below) and on gold tokenization.
That is a different scorer, a different protocol (the shared-task
evaluator's alignment, not this bench's), and UD 2.17 test bytes,
while the golds above are pinned repository commits; read these as
the published bar for the same treebanks, not as rows of this
bench. Raw-text rows:

| Model (UD 2.17, 251125) | Words | Sents | UPOS | XPOS | UFeats | AllTags | Lemma | UAS | LAS | MLAS | BLEX |
|---|---|---|---|---|---|---|---|---|---|---|---|
| english-ewt | 99.07 | 88.42 | 96.75 | 96.27 | 97.21 | 95.51 | 97.13 | 91.59 | 89.91 | 83.87 | 84.91 |
| spanish-ancora | 99.95 | 98.69 | 99.05 | 96.08 | 98.74 | 95.61 | 99.37 | 93.81 | 92.07 | 86.86 | 88.38 |
| catalan-ancora | 99.94 | 99.49 | 99.07 | 97.16 | 98.67 | 96.88 | 99.43 | 94.84 | 93.32 | 87.89 | 89.33 |
| finnish-ftb | 99.91 | 86.84 | 96.73 | 94.90 | 96.69 | 93.89 | 95.66 | 90.21 | 87.31 | 80.22 | 80.78 |
| italian-markit | 99.62 | 98.24 | 97.02 | 97.03 | 94.02 | 92.37 | 88.31 | 88.68 | 84.79 | 70.25 | 77.98 |

### MultiCoNER v2 (English)

The same entity form on MultiCoNER v2 English test
(`en_test.conll` from the `MultiCoNER/multiconer_v2` dataset at
revision `4be2d62c912977ee26ed14d2553a4fe17ca3d980`, source SHA-256
`be7e85f1542a9643de5f01cc99465ec0cb9d96a20bdc3c58418d391e194aff9a`:
249980 sentences, 3773671 tokens, 377805 entities). MultiCoNER
labels 33 fine-grained kinds on lowercased text; peers are compared
at the dataset's six coarse groups (PER, LOC, GRP, PROD, CW, MED)
because the peer label sets are coarse OntoNotes types. Gold is
converted by `analysis/eval/reference/peers/multiconer/convert_gold.py`
(fine-to-coarse taxonomy map, refusal on an unmapped kind) to
`multiconer-en-test.tsv`, SHA-256
`200c831eebb31201c19894149a3b4a1bb0a1f3a2651af67811970c485822a498`,
which is the pinned bench input. Peer runners take
`multiconer` as their label set: PERSON to PER, ORG and NORP to GRP,
GPE, LOC and FAC to LOC, PRODUCT to PROD, WORK_OF_ART to CW, the
rest dropped (so MED is unreachable for both peers and scores 0 by
construction). Gold tokens are given, as on WNUT-17.

- `spacy-multiconer-en-test.json`: spaCy 3.8.7, `en_core_web_sm`
  3.8.0, predictions SHA-256
  `2a51c1c4d1a468786607249068c5fe6113df703700531950948fa5a88b3949fc`.
  Entities 25.24 F1 (69265 correct of 170941 selected); PER 44.79
  and LOC 25.74 carry it, CW 0.03 (58 selected against 53164 gold)
  and PROD 0.50 show where the OntoNotes label set and the
  lowercased fine-grained gold do not meet.
- `stanza-multiconer-en-test.json`: Stanza 1.10.1 English NER,
  pretokenized, predictions SHA-256
  `5275d49d5b3492f971096bddcb4729ea4bd6684e97a3eb26fe95d82ef1ddd66e`.
  Entities 18.33 F1 (41233 correct of 72170 selected); PER 32.72
  and LOC 23.52 carry it, GRP 2.86 (5997 selected, where spaCy
  selects 39353) and CW 0.35 show the same label-set gap. The
  predictions were made in four 512-aligned shards and joined;
  the first shard's first 20480 sentences are byte-identical to
  an earlier unsharded partial run, so the sharding changed no
  output.

Reading note: these are off-the-shelf peers on a lowercased,
fine-grained benchmark whose tuned systems train on MultiCoNER
itself; the numbers are a peer bar under a coarse-label protocol,
not a MultiCoNER leaderboard entry.

## Language: FLORES+ (harness ready, record pending dataset access)

Language records use the `--lang-gold` form on a fixed subset of
the FLORES+ devtest split (openlanguagedata/flores_plus revision
`1147e2dc59f37fbb5c22fd8478770de02cd399f9`, CC BY-SA 4.0,
evaluation only; the files are never committed, see
`docs/licences.md`). The subset is the 73 ISO 639-3 codes both
FLORES+ devtest and lingua 2.2.0 cover, one languoid per code, at
macrolanguage grain where the code splits differ (Mandarin
`cmn_Hans` is gold `zho`, `arb_Arab` is `arb`, `pes` is `fas`,
`ekk` is `est`, `lvs` is `lav`, `khk_Cyrl` is `mon`, `zsm` is
`msa`, `swh` is `swa`, `azj` is `aze`, `als` is `sqi`). FLORES+ has
no Latin file and its `fil_Latn` is Filipino, not Tagalog, so
lingua's `lat` and `tgl` have no gold row.
`analysis/eval/reference/peers/flores/convert_gold.py` holds the
file map and writes `flores-plus-devtest-73.tsv` (`code<TAB>text`,
sorted by code then sentence id), printing every input digest and
the converted file's SHA-256 for the record's `--sha256` pin.

The short-text half of the gate uses the same converter's
`--short-out` file (`flores-plus-devtest-73-short.tsv`): each text
cut to its first four whitespace-separated words, or its first 20
characters when it has fewer words to cut, labels unchanged. It is
a second record from the same pinned rows, not a second corpus.

Peer: lingua 2.2.0 (`lingua-language-detector`, Apache-2.0), all
75 languages in accurate mode as shipped, in the pinned container
at `analysis/eval/reference/peers/lingua/`. Its Arabic code `ara`
is written as gold `arb` (its Arabic is Modern Standard Arabic);
every other code is written as lingua names it. Ours: the
script-based fallback (`--script-fallback`), a closed-set baseline
over the gold label set. Both are the rows #1525's language model
gate compares against, with gold labels and the majority baseline;
no OpenNLP model is a peer or a target here.

No language record is committed yet: the dataset download is gated
on accepting the FLORES+ terms, and the Hugging Face account on
the recording machine is not on the authorized list (the file
resolve answer is 403, checked 2026-10-28). The converter, the
lingua image and the scoring command below are verified end to end
on synthetic inputs; the record `language-flores-plus-devtest.json`
lands with its numbers as soon as access is granted, without any
code change. Nothing in this section is a measured accuracy.

## Throughput receipts

Full receipts (release form: 20 runs of the exact command, min,
mean, p50 and p99 wall seconds, the host's 1-minute load average
around every run, the commit and the core count) are written by
`analysis/eval/reference/throughput_receipt.py` into `throughput/`
beside these records. The timing is the whole command, model load
included. A receipt is only comparable at like host load, which is
why every run's load is in the file.

The first receipt is `throughput/gliner-wnut17.json`, the GLiNER
peer command above over the 1287 WNUT-17 sentences, 20 runs on
2026-10-29 from a clean tree at commit `e8e445c52db1`: min
34.41 s, mean 37.55 s, p50 37.16 s (34.63 sentences/s), p99 and
max 47.23 s. The run began at host 1-minute load 2.28 on 32 cores;
later per-run loads in the receipt rise to about 16 to 19 as the
timed command itself saturates its CPU threads. All 20 runs wrote
byte-identical predictions, SHA-256
`9638a97e64eeeaba95d0d0287da7dc5c325296bbff4ce7efedc42d6bab2977a2`,
the scored record's file. The mutable tag
`turbo-peer-gliner:0.2.21-small-v2.1` named image
`sha256:614fedccce1528a481546dece6378f6344f1a8619c38987732266470158fda09`
for this run. This replaces the 2026-10-28 receipt taken while
other lanes loaded the host; that historical file remains in git
history with the provenance correction recorded in #1655.

Three spaCy receipts over the 2077 English EWT sentences followed
from the same clean PR branch commit `2bd0bf648347`: `spacy-sm`
min 3.50 s, mean 3.55 s, p50 3.55 s (585.20 sentences/s), p99 and
max 3.60 s; `spacy-lg` min 4.07 s, mean 4.18 s, p50 4.16 s
(499.65 sentences/s), p99 and max 4.41 s; and `spacy-trf` min
14.62 s, mean 15.48 s, p50 15.31 s (135.66 sentences/s), p99 and
max 16.20 s. Before each receipt, one run was scored and reproduced
the committed record's sentence, UAS and LAS scores exactly; all
20 timed runs then wrote one byte-identical prediction file per
model. The mutable tags named image IDs
`sha256:c2b1beb463f6e98b99124fcd734a946d03344bf32063357c716df7d92eca6470`
(sm), `sha256:ac75f6a5866cad276de5aef167ccbe27cfb82d49a3f5a027e63090f94315ac7b`
(lg), and
`sha256:bd8fb6da4c3c2dc05f9477be3adc78be4ed5b332bb1552f45e0f2bcd2fe7a286`
(trf).

The Stanza English receipt (`throughput/stanza-en-ewt.json`, the
same 2077 sentences, clean tree at `915af0568f25`) is min 50.27 s,
mean 51.35 s, p50 51.33 s (40.46 sentences/s), p99 and max
52.94 s, beginning at host load 1.31; the scored-once output
reproduced the committed record's sentence, UAS and LAS scores
exactly, and all 20 timed runs are byte-identical (SHA-256
`ae0e58aa2c5045709bae07f100556286eaccf6570ebe28cb54c5314f0a801f90`).
The mutable tag `turbo-peer-stanza:1.10.1-en` named image
`sha256:e79ec7fc22750b287f7277980ae17797233e14b7e932572abeb39cd770675978`
for this run. An earlier capture the same evening is not used: an
external load spike to 39.9 crossed runs 10 to 12 of it.

The Trankit English receipt (`throughput/trankit-en-ewt.json`,
the same 2077 sentences, clean tree at `5d61b1f93d5f`, on
2026-10-30) is min 128.53 s, mean 131.65 s, p50 131.39 s
(15.81 sentences/s), p99 and max 134.66 s, beginning at host load
2.06; later per-run loads sit at about 8 to 13 as Trankit's own
CPU threads saturate, with no run's seconds leaving that band.
One run was scored first and reproduced the committed record's
headline scores exactly, and all 20 timed runs are byte-identical
to the recorded predictions file (SHA-256
`2c14aae8c6ae8d90a03996dcb3a48221916ccb0195c2be2dd8ff4891522314df`,
the digest the Trankit section above records). The mutable tag
`turbo-peer-trankit:1.1.2-en` named image
`sha256:693b004911d6fb9bc1c70b4ed8f833e5b495bcc315e3f33cfde85ba9680cf540`
for this run, the same image id as at record time.

The wtpsplit receipt (`throughput/wtpsplit-en-ewt.json`, the
same 2077 sentences as document text, clean tree at
`ec8dbfca1293`, on 2026-10-30) is min 7.15 s, mean 7.53 s, p50
7.46 s (278.48 sentences/s), p99 and max 8.14 s, beginning at
host load 1.42; later per-run loads rise to about 11 under the
timed command's own threads while its seconds stay in the same
band. All 20 runs are byte-identical to the recorded predictions
(SHA-256
`1aa2702be1c99bcf1743337cfa5fc430ce4c173579f9265091e609a6e2c04841`,
the digest the wtpsplit section above records). The mutable tag
`turbo-peer-wtpsplit:2.2.2-sat-3l` named image
`sha256:e9f66767399d34097ae89a0d6cb16af96dc799327f51fe4d1f045cc84cd14d32`
for this run, the same image id as at record time. An earlier
capture the same evening is not used: it began at host load
13.74 under other lanes' CI and its seconds followed that load
down from 10.15 s to 7.35 s.

Receipts for the other analysis systems land as their booked runs
complete; the records above still carry at most the single
in-process timing their own sections state.

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

# Tokenizer tiers on the same five splits (en shown; the other pins
# are in the inputs table above and in each record)
turbo-eval analysis --gold <gold-dir>/en_ewt-ud-test.conllu \
  --sha256 fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a \
  --tokenize uax29 --tokenize whitespace --out <new-record>.json

# Tokenizer tiers on the MASC held-out split (the extracted release
# directory; its corpus digest is checked against the recorded one)
turbo-eval analysis --gold-masc <masc-dir>/MASC-3.0.0 \
  --tokenize uax29 --tokenize whitespace --out <new-record>.json

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

# Entities on WNUT-17 with GLiNER (gold: emerging.test.annotated of
# leondz/emerging_entities_17, the bytes the pin below names)
docker build --build-arg TURBO_ACCEPT_TERMS=urchade/gliner_small-v2.1 \
  -t turbo-peer-gliner:0.2.21-small-v2.1 analysis/eval/reference/peers/gliner
docker run --rm --network none -v <out-dir>:/out \
  turbo-peer-gliner:0.2.21-small-v2.1 /out/wnut17-test.tsv /out/gliner-wnut17.txt
turbo-eval analysis --ner-gold <out-dir>/wnut17-test.tsv \
  --sha256 2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0 \
  --ner-pred gliner-0.2.21-small-v2.1=<out-dir>/gliner-wnut17.txt \
  --out <new-record>.json

# Language on FLORES+ devtest (after the dataset terms are accepted;
# <flores-dir> holds the devtest .jsonl files at the pinned revision)
python3 analysis/eval/reference/peers/flores/convert_gold.py \
  <flores-dir> <out-dir>/flores-plus-devtest-73.tsv \
  --short-out <out-dir>/flores-plus-devtest-73-short.tsv
docker build -t turbo-peer-lingua:2.2.0 analysis/eval/reference/peers/lingua
docker run --rm --network none -v <out-dir>:/out \
  turbo-peer-lingua:2.2.0 /out/flores-plus-devtest-73.tsv /out/lingua.txt
turbo-eval analysis --lang-gold <out-dir>/flores-plus-devtest-73.tsv \
  --sha256 <digest printed by convert_gold.py> \
  --lang-pred lingua-2.2.0=<out-dir>/lingua.txt \
  --script-fallback --out <new-record>.json
# Short-text record: the same three commands on
# flores-plus-devtest-73-short.tsv with its printed digest.

# Throughput receipt (20 runs of one system command; GLiNER shown)
python3 analysis/eval/reference/throughput_receipt.py \
  analysis/eval/records/peer-analysis-1521/throughput/gliner-wnut17.json 1287 \
  --output <out-dir>/gliner-wnut17.txt \
  -- docker run --rm --network none -v <out-dir>:/out \
  turbo-peer-gliner:0.2.21-small-v2.1 /out/wnut17-test.tsv /out/gliner-wnut17.txt
```

Recorded 2026-10-28 under load average about 31 to 36, except
the Stanza MultiCoNER predictions, which ran 2026-10-28 under
load average about 65 to 120 (accuracy record; no timing
claimed).

Scoring correction: sentence boundaries are within-document on
every bench path (a document's final sentence end is never a
boundary, for targets and selected alike). An earlier scorer let
the CoNLL-U path count file-final style boundaries the tiers never
counted, which also let truncated predictions earn unselected
credit and collapsed an unaligned governor to a root in UAS. All
CoNLL-U records above were re-scored from their existing
prediction files under the corrected scorer; EWT sentence scores
moved down by up to about two points (316 documents), the other
splits by hundredths, LAS unchanged to two decimals.
