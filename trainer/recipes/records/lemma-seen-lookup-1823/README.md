# Lemma head `--min-tree-freq`, `--seen-lookup`, and `--bilstm` sweep (#1823)

Test-split lemma accuracy across `ud-en-ewt` (ModernBERT-base), `ud-it-markit`
(XLM-RoBERTa-base), `ud-de-gsd` (XLM-RoBERTa-base), and `ud-es-ancora`
(XLM-RoBERTa-base), sweeping `--min-tree-freq` in `{3, 2, 1}` with and without
unambiguous seen-form lookup (`--seen-lookup 2` and `--seen-lookup 1`) and a
1-layer 256-hidden BiLSTM adapter (`--bilstm 256x1`), all scored end-to-end
inside `turbo-train-head train-lemma` (`seed = 42`, `epochs = 5`, `lr = 0.05`,
1,000 paired sentence bootstrap resamples against the majority baseline).

## Winning configurations vs majority baseline and Stanza

| treebank | encoder | variant | trees | lookup entries | test words | overall | seen | unseen | majority | delta vs majority (95% CI) | Stanza overall (seen / unseen) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `ud-en-ewt` | `answerdotai/ModernBERT-base` | `mtf2-lookup1` | 469 | 18,697 (`k=1`) | 25,094 | **0.9583** | **0.9813** | 0.7299 | 0.9503 | **+0.0080 [+0.0057, +0.0102]** | 0.9598 (`stanza-1.10.1-en`) |
| `ud-it-markit` | `FacebookAI/xlm-roberta-base` | `mtf2-bilstm-lookup1` | 185 | 3,980 (`k=1`) | 10,196 | **0.8530** | **0.8732** | **0.7152** | 0.8172 | **+0.0358 [+0.0317, +0.0403]** | 0.8695 (0.8732 / 0.8308, `stanza-1.15.0-it_markit`) |
| `ud-de-gsd` | `FacebookAI/xlm-roberta-base` | `mtf2-bilstm-lookup1` | 616 | 48,968 (`k=1`) | 16,499 | **0.9670** | **0.9920** | **0.7780** | 0.9424 | **+0.0246 [+0.0221, +0.0271]** | 0.9743 (0.9803 / 0.8468, `stanza-1.15.0-de_gsd`) |
| `ud-es-ancora` | `FacebookAI/xlm-roberta-base` | `mtf2-bilstm-lookup1` | 695 | 36,065 (`k=1`) | 53,622 | **0.9883** | **0.9949** | **0.8711** | 0.9677 | **+0.0206 [+0.0193, +0.0220]** | 0.9900 (0.9922 / 0.9505, `stanza-1.15.0-es_ancora`) |

All four treebanks clear the majority baseline with the lower bound of the 95%
paired bootstrap interval strictly above zero (`low > 0`) under both `k = 2`
and `k = 1`.

## Full sweep tables

### `ud-en-ewt` (`answerdotai/ModernBERT-base`, 25,094 test words: 22,824 seen, 2,270 unseen; majority baseline `0.9503`: seen `0.9756`, unseen `0.6960`)

| variant | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mtf3-nolookup` | 3 | none | none | 337 | 0 | 4 | 0.9373 | 0.9380 | 0.9590 | 0.7286 | -0.0124 | [-0.0155, -0.0094] |
| `mtf2-nolookup` | 2 | none | none | 469 | 0 | 4 | 0.9363 | 0.9372 | 0.9581 | 0.7299 | -0.0131 | [-0.0161, -0.0101] |
| `mtf1-nolookup` | 1 | none | none | 1,064 | 0 | 4 | 0.9371 | 0.9375 | 0.9588 | 0.7256 | -0.0128 | [-0.0157, -0.0100] |
| `mtf3-lookup2` | 3 | none | `k=2` | 337 | 8,896 | 4 | 0.9542 | 0.9549 | 0.9777 | 0.7286 | +0.0046 | [+0.0020, +0.0071] |
| `mtf2-lookup2` | 2 | none | `k=2` | 469 | 8,896 | 4 | 0.9537 | 0.9552 | 0.9779 | 0.7299 | +0.0049 | [+0.0025, +0.0073] |
| `mtf1-lookup2` | 1 | none | `k=2` | 1,064 | 8,896 | 4 | 0.9540 | 0.9545 | 0.9775 | 0.7256 | +0.0042 | [+0.0017, +0.0066] |
| `mtf3-lookup1` | 3 | none | `k=1` | 337 | 18,697 | 4 | 0.9576 | 0.9579 | 0.9809 | 0.7286 | +0.0076 | [+0.0052, +0.0099] |
| `mtf2-lookup1` | 2 | none | `k=1` | 469 | 18,697 | 4 | 0.9572 | **0.9583** | **0.9813** | **0.7299** | **+0.0080** | **[+0.0057, +0.0102]** |
| `mtf1-lookup1` | 1 | none | `k=1` | 1,064 | 18,697 | 4 | 0.9572 | 0.9577 | 0.9810 | 0.7256 | +0.0074 | [+0.0050, +0.0095] |
| `mtf2-bilstm-nolookup` | 2 | `256x1` | none | 469 | 0 | 4 | 0.9312 | 0.9307 | 0.9513 | 0.7260 | -0.0196 | [-0.0228, -0.0164] |
| `mtf2-bilstm-lookup2` | 2 | `256x1` | `k=2` | 469 | 8,896 | 4 | 0.9521 | 0.9517 | 0.9744 | 0.7260 | +0.0014 | [-0.0011, +0.0038] |
| `mtf2-bilstm-lookup1` | 2 | `256x1` | `k=1` | 469 | 18,697 | 4 | 0.9555 | 0.9552 | 0.9783 | 0.7260 | +0.0049 | [+0.0026, +0.0072] |
| `stanza-1.10.1-en` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9598 | n/a | n/a | +0.0095 | n/a |

### `ud-it-markit` (`FacebookAI/xlm-roberta-base`, 10,196 test words: 8,890 seen, 1,306 unseen; majority baseline `0.8172`: seen `0.8654`, unseen `0.4893`)

| variant | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mtf3-nolookup` | 3 | none | none | 145 | 0 | 4 | 0.9132 | 0.8280 | 0.8535 | 0.6539 | +0.0108 | [+0.0065, +0.0153] |
| `mtf2-nolookup` | 2 | none | none | 185 | 0 | 1 | 0.9186 | 0.8321 | 0.8543 | 0.6807 | +0.0149 | [+0.0103, +0.0192] |
| `mtf1-nolookup` | 1 | none | none | 295 | 0 | 3 | 0.9203 | 0.8323 | 0.8571 | 0.6631 | +0.0151 | [+0.0107, +0.0194] |
| `mtf3-lookup2` | 3 | none | `k=2` | 145 | 1,391 | 4 | 0.9207 | 0.8345 | 0.8611 | 0.6539 | +0.0174 | [+0.0132, +0.0216] |
| `mtf2-lookup2` | 2 | none | `k=2` | 185 | 1,391 | 1 | 0.9242 | 0.8373 | 0.8603 | 0.6807 | +0.0201 | [+0.0157, +0.0241] |
| `mtf1-lookup2` | 1 | none | `k=2` | 295 | 1,391 | 3 | 0.9251 | 0.8371 | 0.8627 | 0.6631 | +0.0199 | [+0.0159, +0.0239] |
| `mtf3-lookup1` | 3 | none | `k=1` | 145 | 3,980 | 1 | 0.9295 | 0.8450 | 0.8737 | 0.6501 | +0.0279 | [+0.0243, +0.0316] |
| `mtf2-lookup1` | 2 | none | `k=1` | 185 | 3,980 | 1 | 0.9323 | 0.8481 | 0.8727 | 0.6807 | +0.0309 | [+0.0272, +0.0346] |
| `mtf1-lookup1` | 1 | none | `k=1` | 295 | 3,980 | 3 | 0.9314 | 0.8459 | 0.8728 | 0.6631 | +0.0287 | [+0.0250, +0.0325] |
| `mtf2-bilstm-nolookup` | 2 | `256x1` | none | 185 | 0 | 4 | 0.9271 | 0.8368 | 0.8547 | 0.7152 | +0.0196 | [+0.0149, +0.0244] |
| `mtf2-bilstm-lookup2` | 2 | `256x1` | `k=2` | 185 | 1,391 | 4 | 0.9331 | 0.8441 | 0.8630 | 0.7152 | +0.0269 | [+0.0224, +0.0317] |
| `mtf2-bilstm-lookup1` | 2 | `256x1` | `k=1` | 185 | 3,980 | 4 | 0.9397 | **0.8530** | **0.8732** | **0.7152** | **+0.0358** | **[+0.0317, +0.0403]** |
| `stanza-1.15.0-it_markit` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.8695 | 0.8732 | 0.8308 | +0.0523 | [+0.0475, +0.0573] |

### `ud-de-gsd` (`FacebookAI/xlm-roberta-base`, 16,499 test words: 14,567 seen, 1,932 unseen; majority baseline `0.9424`: seen `0.9901`, unseen `0.5823`)

| variant | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mtf3-nolookup` | 3 | none | none | 506 | 0 | 4 | 0.9417 | 0.9367 | 0.9642 | 0.7298 | -0.0056 | [-0.0091, -0.0021] |
| `mtf2-nolookup` | 2 | none | none | 616 | 0 | 4 | 0.9413 | 0.9369 | 0.9642 | 0.7314 | -0.0055 | [-0.0088, -0.0021] |
| `mtf1-nolookup` | 1 | none | none | 929 | 0 | 4 | 0.9410 | 0.9378 | 0.9653 | 0.7308 | -0.0045 | [-0.0079, -0.0011] |
| `mtf3-lookup2` | 3 | none | `k=2` | 506 | 14,634 | 4 | 0.9590 | 0.9535 | 0.9831 | 0.7298 | +0.0111 | [+0.0085, +0.0137] |
| `mtf2-lookup2` | 2 | none | `k=2` | 616 | 14,634 | 4 | 0.9581 | 0.9530 | 0.9824 | 0.7314 | +0.0106 | [+0.0080, +0.0133] |
| `mtf1-lookup2` | 1 | none | `k=2` | 929 | 14,634 | 4 | 0.9583 | 0.9535 | 0.9830 | 0.7308 | +0.0111 | [+0.0085, +0.0137] |
| `mtf3-lookup1` | 3 | none | `k=1` | 506 | 48,968 | 4 | 0.9652 | 0.9611 | 0.9918 | 0.7298 | +0.0187 | [+0.0166, +0.0209] |
| `mtf2-lookup1` | 2 | none | `k=1` | 616 | 48,968 | 4 | 0.9646 | 0.9611 | 0.9916 | 0.7314 | +0.0188 | [+0.0166, +0.0211] |
| `mtf1-lookup1` | 1 | none | `k=1` | 929 | 48,968 | 4 | 0.9647 | 0.9612 | 0.9918 | 0.7308 | +0.0188 | [+0.0167, +0.0212] |
| `mtf2-bilstm-nolookup` | 2 | `256x1` | none | 616 | 0 | 4 | 0.9544 | 0.9526 | 0.9758 | 0.7780 | +0.0102 | [+0.0069, +0.0136] |
| `mtf2-bilstm-lookup2` | 2 | `256x1` | `k=2` | 616 | 14,634 | 4 | 0.9639 | 0.9606 | 0.9848 | 0.7780 | +0.0182 | [+0.0154, +0.0210] |
| `mtf2-bilstm-lookup1` | 2 | `256x1` | `k=1` | 616 | 48,968 | 4 | 0.9697 | **0.9670** | **0.9920** | **0.7780** | **+0.0246** | **[+0.0221, +0.0271]** |
| `stanza-1.15.0-de_gsd` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9743 | 0.9803 | 0.8468 | +0.0319 | [+0.0290, +0.0351] |

### `ud-es-ancora` (`FacebookAI/xlm-roberta-base`, 53,622 test words: 50,775 seen, 2,847 unseen; majority baseline `0.9677`: seen `0.9897`, unseen `0.5753`)

| variant | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mtf3-nolookup` | 3 | none | none | 574 | 0 | 4 | 0.9754 | 0.9744 | 0.9837 | 0.8089 | +0.0068 | [+0.0051, +0.0084] |
| `mtf2-nolookup` | 2 | none | none | 695 | 0 | 4 | 0.9763 | 0.9750 | 0.9840 | 0.8142 | +0.0073 | [+0.0057, +0.0090] |
| `mtf1-nolookup` | 1 | none | none | 961 | 0 | 4 | 0.9761 | 0.9750 | 0.9839 | 0.8149 | +0.0073 | [+0.0057, +0.0089] |
| `mtf3-lookup2` | 3 | none | `k=2` | 574 | 17,513 | 4 | 0.9830 | 0.9818 | 0.9915 | 0.8089 | +0.0142 | [+0.0128, +0.0156] |
| `mtf2-lookup2` | 2 | none | `k=2` | 695 | 17,513 | 4 | 0.9836 | 0.9822 | 0.9917 | 0.8142 | +0.0146 | [+0.0131, +0.0160] |
| `mtf1-lookup2` | 1 | none | `k=2` | 961 | 17,513 | 4 | 0.9835 | 0.9822 | 0.9916 | 0.8149 | +0.0146 | [+0.0131, +0.0160] |
| `mtf3-lookup1` | 3 | none | `k=1` | 574 | 36,065 | 4 | 0.9853 | 0.9836 | 0.9934 | 0.8089 | +0.0159 | [+0.0147, +0.0173] |
| `mtf2-lookup1` | 2 | none | `k=1` | 695 | 36,065 | 4 | 0.9856 | 0.9839 | 0.9934 | 0.8142 | +0.0162 | [+0.0148, +0.0176] |
| `mtf1-lookup1` | 1 | none | `k=1` | 961 | 36,065 | 4 | 0.9855 | 0.9838 | 0.9932 | 0.8149 | +0.0161 | [+0.0148, +0.0174] |
| `mtf2-bilstm-nolookup` | 2 | `256x1` | none | 695 | 0 | 4 | 0.9869 | 0.9857 | 0.9921 | 0.8711 | +0.0180 | [+0.0165, +0.0195] |
| `mtf2-bilstm-lookup2` | 2 | `256x1` | `k=2` | 695 | 17,513 | 4 | 0.9891 | 0.9876 | 0.9942 | 0.8711 | +0.0200 | [+0.0186, +0.0213] |
| `mtf2-bilstm-lookup1` | 2 | `256x1` | `k=1` | 695 | 36,065 | 4 | 0.9899 | **0.9883** | **0.9949** | **0.8711** | **+0.0206** | **[+0.0193, +0.0220]** |
| `stanza-1.15.0-es_ancora` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9900 | 0.9922 | 0.9505 | +0.0224 | [+0.0207, +0.0240] |

## Winning artifact digests

| treebank | variant | `head.safetensors` SHA-256 | `head.json` SHA-256 | assembled `manifest.json` SHA-256 |
|---|---|---|---|---|
| `ud-en-ewt` | `mtf2-lookup1` | `be3a9eb71e8709c61cd8224fdc571f509a236be5ef3ba2abd1600058981ea652` | `f7c5a62808396833afccc44487a6385251a96783fa4e8868219a8bbc263deeff` | `7f3923cfa1d5e24d9b0b34a02c5faa0ccd16656e21f25b170d6cd91436bdb456` |
| `ud-it-markit` | `mtf2-bilstm-lookup1` | `0c305b865db5e34440e7d7ea20d1f233723c5d2af5711e496921f43bdb38fcc9` | `d8e8f70560ba855a886cc1146979daf35c19812b531332398de7ee9a3e2ce461` | `1020c9912319c4d1f4586c1f3109d05de854ac6a94c4167f7c2a18b64867164c` |
| `ud-de-gsd` | `mtf2-bilstm-lookup1` | `918a79d49ec12466c28f719a7401e36eed8de4749a84dfb39802c5f3a4bb22dc` | `1701aa7bf53e5d6eef5845248151a119215c70a3292be1db4094bb59ac177251` | `2cd3c57ec7dcd8c2d87cc8173473e00044bc5724457bd401a06ed7ff0433139f` |
| `ud-es-ancora` | `mtf2-bilstm-lookup1` | `b1a569fb8d1a25dc29a729e0c7c4ce97cd657657400de5812afa7e8cc43731e8` | `c3530dfe640df8da2cbc60833a782253cd42c3fe2f3b39272e03c23e4a1397b4` | `066ad7700fa147b1cfd5dd938d1dcb9ad9c4caf9d910d423d92c9ff6c697972c` |

## Reproduce

```sh
turbo-train-head train-lemma --bundle modernbert-base \
  --corpus ud-en-ewt --dir UD_English-EWT-r2.18 \
  --out out-en-ewt --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-it-markit --dir UD_Italian-MarkIT-r2.16 \
  --out out-it-markit --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-de-gsd --dir UD_German-GSD-r2.16 \
  --out out-de-gsd --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-es-ancora --dir UD_Spanish-AnCora-r2.16 \
  --out out-es-ancora --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda
```
