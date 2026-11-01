# Sentence boundary heads, test split (#1725)

Held-out sentence exact-span F1 (`evaluate_spans`) for `train-sentences`
across MASC 3.0.0 and the five pinned Universal Dependencies treebanks
(`ud-en-ewt`, `ud-es-ancora`, `ud-ca-ancora`, `ud-fi-ftb`, `ud-it-markit`),
all receipts of `turbo-train-head` at commit `f59cb5ba`:

| corpus | receipt | encoder | test sentences | newline F1 | `RuleSplitter` F1 | head F1 | vs newline (95% CI) | vs `RuleSplitter` (95% CI) |
|---|---|---|---|---|---|---|---|---|
| `masc-pos` (`en`) | `masc-pos.json` | `answerdotai/ModernBERT-base` | 6,384 | 0.2558 | 0.6520 | 0.7781 | +0.5223 [+0.3522, +0.7319] | +0.1262 [-0.0288, +0.2701] |
| `ud-en-ewt` (`en`) | `ud-en-ewt.json` | `answerdotai/ModernBERT-base` | 2,077 | 0.2900 | 0.5970 | 0.7681 | +0.4781 [+0.4042, +0.5491] | +0.1711 [+0.1436, +0.2032] |
| `ud-es-ancora` (`es`) | `ud-es-ancora.json` | `FacebookAI/xlm-roberta-base` | 1,721 | 0.0021 | 0.9414 | 0.9755 | +0.9734 [+0.9618, +0.9840] | +0.0341 [-0.0035, +0.0800] |
| `ud-ca-ancora` (`ca`) | `ud-ca-ancora.json` | `FacebookAI/xlm-roberta-base` | 1,846 | 0.0020 | 0.9943 | 0.9908 | +0.9888 [+0.9801, +0.9964] | -0.0035 [-0.0132, +0.0049] |
| `ud-fi-ftb` (`fi`) | `ud-fi-ftb.json` | `FacebookAI/xlm-roberta-base` | 1,867 | 0.0000 | 0.6720 | 0.9224 | +0.9224 [+0.9040, +0.9385] | +0.2504 [+0.2216, +0.2810] |
| `ud-it-markit` (`it`) | `ud-it-markit.json` | `FacebookAI/xlm-roberta-base` | 340 | 0.0000 | 0.7136 | 0.9912 | +0.9912 [+0.9778, +1.0000] | +0.2776 [+0.2232, +0.3323] |

## Reading note

- Each document (`# newdoc` run in UD, or MASC document) is reconstructed
  into a single text with exact token byte offsets. When a UD treebank lacks
  `# newdoc` markers (`ud-ca-ancora`, `ud-fi-ftb`, `ud-it-markit`), consecutive
  sentences are grouped into 10-sentence blocks separated by single spaces so
  boundary placement cannot rely on synthetic document breaks.
- The newline baseline splits at `\n` or `\r` between consecutive tokens, or
  emits one whole-block sentence when no line break appears.
- The `RuleSplitter` baseline runs `turbo-sentdetect`'s UAX #29 + language
  abbreviation inventory splitter (`Rules::english()`, `Rules::spanish()`,
  `Rules::catalan()`, `Rules::finnish()`, `Rules::italian()`) with
  `LineBreaks::AsWhitespace` on the reconstructed block text and maps each
  predicted character span back to token indices.
- On conversational and unpunctuated genres (`masc-pos` spoken transcripts
  and emails, `ud-en-ewt` web reviews and newsgroups, `ud-fi-ftb` clause
  examples, and `ud-it-markit` student essays), a 1-layer BiLSTM adapter
  (`--bilstm 64x1`) with 64-dim hashed word identity (`--hash-embed 64`) and
  first-piece pooling (`--pooling first`) lets the boundary classifier see the
  following token's shape and capitalization across unpunctuated boundaries.
- Historical provenance note on `ud-en-ewt.json`: `ud-en-ewt.json` is the
  historical receipt recorded at commit `3e1f1f366d5c2ec63af235e7fe8ef1e2363b4a05`
  (with embedded run `--commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46`),
  prior to the authoritative `# text` reconstruction fix in `ud.rs` (where test
  document 52's no-break space had been rebuilt from `FORM`/`MISC` as an ASCII
  space). It is preserved as historical run provenance rather than a post-fix
  re-run.
- Remaining comparisons under #1725 (still incomplete): these six receipts
  record the `train-sentences` head, the newline baseline, and the
  `turbo-sentdetect` `RuleSplitter` baseline. Comparison rows against
  `turbo-sat` (SaT) and external `spaCy` / `Stanza` sentence segmenters on the
  same document reconstructions remain open under #1725.

## Reproduce

```sh
turbo-train-head train-sentences --corpus masc-pos --dir MASC-3.0.0 \
  --bundle modernbert-base --device cuda --pooling first --hash-embed 64 \
  --bilstm 64x1 --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-masc-pos-bilstm

turbo-train-head train-sentences --corpus ud-en-ewt --dir ud-en-ewt \
  --bundle modernbert-base --device cuda --pooling first --hash-embed 64 \
  --bilstm 64x1 --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-ud-en-ewt-bilstm

turbo-train-head train-sentences --corpus ud-es-ancora --dir ud-es-ancora \
  --bundle xlm-roberta-base --device cuda --pooling first --hash-embed 64 \
  --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-ud-es-ancora

turbo-train-head train-sentences --corpus ud-ca-ancora --dir ud-ca-ancora \
  --bundle xlm-roberta-base --device cuda --pooling first --hash-embed 64 \
  --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-ud-ca-ancora

turbo-train-head train-sentences --corpus ud-fi-ftb --dir ud-fi-ftb \
  --bundle xlm-roberta-base --device cuda --pooling first --hash-embed 64 \
  --bilstm 64x1 --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-ud-fi-ftb-bilstm

turbo-train-head train-sentences --corpus ud-it-markit --dir ud-it-markit \
  --bundle xlm-roberta-base --device cuda --pooling first --hash-embed 64 \
  --patience 6 --tolerance 0.0005 --threads 16 \
  --commit f59cb5ba49be9963503caaa54bd3fa7f31bbdc46 --out out-it-markit
```
