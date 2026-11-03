# Peer rows for EWT and LitBank, and the Re-DocRED dependency-path rule (#1825)

The peer rows the gates of #1728 (EWT), #1813 (LitBank coreference) and
#1734 (Re-DocRED) name, each from an executed run on the recorded test
files, scored by `turbo-train-head peer` (or `baseline`) with the receipt
committed here. Hosts are machine kinds and commands name directories.
No published number is copied; every row's predictions file is pinned by
SHA-256 in its receipt.

## EWT (UD_English-EWT r2.18, 2,077 test sentences, 25,094 words)

Peers tag the gold words as given (pretokenized, no multiword expansion):
spaCy 3.8.7 `en_core_web_trf` 3.8.0, Stanza 1.10.1, Trankit 1.1.2
(xlm-roberta-base). Majority baselines: UPOS 0.8617, XPOS 0.8381, feats
0.8292, lemma 0.9503, LAS 0.3322 / UAS 0.3707.

| row | UPOS | XPOS | feats | lemma | LAS | UAS |
|---|---|---|---|---|---|---|
| spaCy trf | 0.9326 | 0.9532 | 0.6766 | 0.9531 | 0.3795 | 0.5266 |
| Stanza | 0.9677 | 0.9624 | 0.9622 | 0.9683 | 0.8835 | 0.9076 |
| Trankit | 0.9649 | 0.9700 | 0.8534 | 0.9138 | 0.8578 | 0.8935 |

LAS is strict (`nsubj:pass` counts only whole), as `train-parse` scores;
the receipt's `uas` is beside it. spaCy's parser writes ClearNLP-style
relations (`prep`, `pobj`, `ROOT`) against EWT's UD r2 labels, so its LAS
row is a conventions row, not a parser-quality row.

## LitBank coreference (the ten test works, singleton baseline CoNLL F1 0.3009)

| row | MUC | B-cubed | CEAF-e | CoNLL F1 | vs singleton [95%] |
|---|---|---|---|---|---|
| spaCy 3.4.4 `en_coreference_web_trf` 3.4.0a2 | 0.7768 | 0.5418 | 0.1989 | 0.5058 | +0.2049 [+0.0791, +0.3122] |
| Stanza 1.10.1 coref | 0.8939 | 0.8083 | 0.7130 | 0.8051 | +0.5041 [+0.3991, +0.5859] |

A span a peer gives two clusters is kept in the first; a span crossing a
sentence end is clipped to its sentence (the peer scripts note both).

## Re-DocRED (the #1734 test split: 500 documents, 17,448 gold triples)

The deterministic dependency-path rule (`baseline --corpus re-docred
--dependency-path`): each document's sentences parsed by our EWT
dependency head (ModernBERT-base, manifest `65a07202...`, BiLSTM 256x1,
test LAS 0.7850 on EWT; head.safetensors
`128dda9122e2ada03f669196d845d4f389129b82ae436ba9ef8e4f01dfd8c2ed`), an
ordered entity pair labelled by the most frequent training relation on
the shortest dependency path between the two entities' first mentions'
head words (8,339 paths mined), `none` otherwise. Two runs gave identical
predictions bytes.

| row | P | R | micro F1 | found | right |
|---|---|---|---|---|---|
| majority (`none`) | | | 0.0000 | 0 | 0 |
| dependency-path rule | 0.0655 | 0.0625 | 0.0640 | 16,630 | 1,090 |
| trained head (#1734 receipt) | 0.6177 | 0.3650 | 0.4589 | 10,311 | 6,369 |

The rule is a floor, as expected: it sees only same-sentence pairs whose
path the training split annotated, over predicted parses.

## Reproduce

```sh
# EWT words for the peers
turbo-train-head export --corpus ud-en-ewt --dir UD_English-EWT-r2.18 --out ewt-test.conllu
# ewt-test.conllu: sha256 72eb22e8b7a1f279947043aba375c9ff6565cf6d49a8f9ae14a8ae0294889091

# the pinned peer images (analysis/eval/reference/peers/)
docker build -t turbo-peer-spacy-trf:3.8.7 --build-arg MODEL=en_core_web_trf peers/spacy
docker build -t turbo-peer-stanza:1.10.1 peers/stanza
docker build -t turbo-peer-trankit:1.1.2 peers/trankit
docker build -f peers/spacy/Dockerfile.coref -t turbo-peer-spacy-coref:3.4.4 peers/spacy
docker build -f peers/stanza/Dockerfile.coref -t turbo-peer-stanza-coref:1.10.1 peers/stanza

docker run --rm -v $PWD:/data --entrypoint python turbo-peer-spacy-trf:3.8.7 \
  /tag_predict.py /data/ewt-test.conllu /data/spacy-trf-tags.conllu en_core_web_trf
docker run --rm -v $PWD:/data --entrypoint python turbo-peer-spacy-trf:3.8.7 \
  /full_predict.py /data/ewt-test.conllu /data/spacy-trf-full.conllu en_core_web_trf
docker run --rm -v $PWD:/data --entrypoint python turbo-peer-stanza:1.10.1 \
  /pretokenized_predict.py /data/ewt-test.conllu /data/stanza-full.conllu
docker run --rm -v $PWD:/data --entrypoint python turbo-peer-trankit:1.1.2 \
  /pretokenized_predict.py /data/ewt-test.conllu /data/trankit-full.conllu

turbo-train-head peer --corpus ud-en-ewt-<upos|xpos|feats> --dir UD_English-EWT-r2.18 \
  --peer <name> --predictions <file> --column <col> --out <dir>
turbo-train-head peer --corpus ud-en-ewt --dir UD_English-EWT-r2.18 \
  --peer <name> --predictions <file> --column <lemma|deps> --out <dir>

# LitBank coref
turbo-train-head export --corpus litbank --dir <litbank checkout> --out litbank-export
docker run --rm -v $PWD:/data turbo-peer-spacy-coref:3.4.4 /data/litbank-export /data/spacy-coref.conll
docker run --rm -v $PWD:/data turbo-peer-stanza-coref:1.10.1 /data/litbank-export /data/stanza-coref.conll
turbo-train-head peer --corpus litbank --dir <litbank checkout> --peer <name> \
  --predictions <file> --column coref --out <dir>
turbo-train-head baseline --corpus litbank --dir <litbank checkout> --out <dir>

# Re-DocRED rule row
turbo-train-head train-parse --corpus ud-en-ewt --dir UD_English-EWT-r2.18 \
  --bundle modernbert-base --out <parser dir> --bilstm 256x1
turbo-train-head baseline --corpus re-docred --dir <Re-DocRED checkout> \
  --dependency-path --parser <parser dir> --bundle modernbert-base --out <dir>
```
