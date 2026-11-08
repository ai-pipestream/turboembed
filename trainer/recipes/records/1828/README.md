# MASC POS and LitBank coreference sweeps (#1828)

Every run is one `turbo-train-head` receipt at commit `a3f73995`, on the
RTX 4080 SUPER (CUDA, F32), on the same recorded corpus, ModernBERT-base
bundle (manifest `65a07202...`) and device as the head it would replace.
Each run is given `--against` that head's run directory, so its receipt
carries the shipped head's score on the same test split and the paired
bootstrap of the new head against it (1000 resamples, the run's seed).
Commands name directories in place of the training machine's paths; no
receipt has a host.

## MASC POS, against head B (0.9581, unseen words 0.7287)

| run | flags beyond head B's | test | unseen | vs head B [95%] | best epoch |
|---|---|---|---|---|---|
| `pos-shipped` | none: head B's command at this commit | 0.9617 | 0.7433 | +0.0036 [+0.0026, +0.0047] | 59 |
| `pos-crf` | `--crf` | 0.9616 | 0.7353 | +0.0036 [+0.0026, +0.0046] | 53 |
| `pos-hash128` | `--hash-embed 128` | 0.9613 | 0.7305 | +0.0032 [+0.0021, +0.0042] | 60 |
| `pos-hashrows` | `--hash-rows 5000,2000,5000,2500` | 0.9595 | 0.7335 | +0.0014 [+0.0004, +0.0026] | 35 |
| `pos-bilstm256x2` | `--bilstm 256x2` | 0.9640 | 0.7325 | +0.0059 [+0.0049, +0.0070] | 59 |

Head B already has prefix and suffix rows (5000/1000/2500/2500), so the
`--hash-rows` run doubles the prefix and suffix tables. `pos-shipped` is
the control: head B's command, retrained, gains as much as `--crf` or
`--hash-embed 128`. It ran to epoch 59 where head B stopped at 32, so
most of each gain is the longer run. Only the two-layer BiLSTM is above
the control.

## LitBank coreference, against the first head (CoNLL F1 0.7094)

| shape | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| the first head's (`coref-shipped`) | 0.7094 | 0.7348 | 0.7216 | 0.7219 |
| `--max-antecedents 50` | 0.7318 | 0.7243 | 0.7315 | 0.7292 |
| `--max-antecedents 100` | 0.7408 | 0.7252 | 0.7212 | 0.7291 |
| `--max-antecedents 250` | 0.7362 | 0.7354 | 0.7220 | 0.7312 |
| `--max-words 10` | 0.7356 | 0.7316 | 0.7253 | 0.7308 |
| `--width-dim 20` | 0.7335 | 0.7213 | 0.7276 | 0.7275 |
| `--max-words 20` | 0.7192 | | | |
| `--width-dim 40` | 0.7272 | | | |
| `--mlp 512` | 0.7166 | | | |

Seed 1's run of the first head's shape reproduces it exactly (the same
test score, best epoch 7). Seeds 2 and 3 of the same shape score 0.7348
and 0.7216, so the first head is the low draw of its shape. The variants'
three-seed means are 0.005 to 0.009 above the shape's own mean, inside
its seed spread of 0.025, so no span or ranking setting is shown to help.
MUC, B-cubed and CEAF-e for every run are in its receipt.

## In the zoo

- `turbo-en-masc-pos-modernbert-base-2`: `pos-shipped` (0.9617), head
  B's shape, so no speed change. It replaces `-1` and is the English
  `COMPONENT_POS` entry `PosTagger::from_zoo_language` picks.
- `turbo-en-masc-pos-modernbert-base-bilstm256x2-1`: `pos-bilstm256x2`
  (0.9640), opt-in by id (`PosTagger::from_zoo`). It is listed after
  `-2`, so the language lookup never picks it. Through the annotator
  (`pos_annotator_eval`, CPU, the test split, three interleaved pairs)
  both bundles give their receipts' scores (0.9617 and 0.9640). The
  machine's CI runners were busy throughout (load average 19 rising to
  119), so the times are indicative only: the two-layer bundle took
  1.10x and 1.11x the time of `-2` in the first two pairs (143.9 s
  against 157.8 s, 333.9 s against 372.1 s), and the third pair (450.4 s
  against 364.5 s) is load noise.
- `turbo-en-litbank-modernbert-base-coref-2`: `coref-shipped-s2`
  (0.7348, +0.0254 [+0.0106, +0.0411] against the first head), the first
  head's shape, so no speed change. It replaces `-1`.

Each head file's SHA-256 is pinned in `analysis/pins/models.txt`. The
bundles are `turbo-train-head assemble` of each run over the
ModernBERT-base bundle, as in `masc-pos-1649/README.md`.

## Reproduce

```sh
turbo-train-head train --corpus masc-pos --dir MASC-3.0.0 --bundle modernbert-base --out pos-shipped \
  --device cuda --pooling first --threads 16 --commit a3f73995 --against masc-pos-B \
  --hash-embed 64 --bilstm 256x1
turbo-train-head train-coref --corpus litbank --dir litbank --bundle modernbert-base --out coref-shipped-s2 \
  --device cuda --commit a3f73995 --seed 2 --against turbo-en-litbank-modernbert-base-coref-1
```

Each other row is the same command with its own flags and seed, as its
receipt's `command` gives.
