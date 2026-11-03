# Lemma head `--min-tree-freq`, `--seen-lookup`, and `--bilstm` sweep (#1823)

Test-split lemma accuracy across `ud-en-ewt` (ModernBERT-base), `ud-it-markit`
(XLM-RoBERTa-base), `ud-de-gsd` (XLM-RoBERTa-base), and `ud-es-ancora`
(XLM-RoBERTa-base), sweeping `--min-tree-freq` in `{3, 2, 1}` with and without
unambiguous seen-form lookup (`--seen-lookup 2` and `--seen-lookup 1`) and a
1-layer 256-hidden BiLSTM adapter (`--bilstm 256x1`), all scored end-to-end
inside `turbo-train-head train-lemma` (`seed = 42`, `epochs = 5`, `lr = 0.05`,
1,000 paired sentence bootstrap resamples against the majority baseline).

Every variant (48 training runs at commit `2a05cf523424b06981f8281c819f4e54189a0c66`),
the 4 majority baseline receipts (`<treebank>/majority.json`, generated with
`turbo-train-head baseline --corpus ud-<bank>-lemma` at commit
`4873eb828a13345e1a41ab6a23c0b71103d89c63`), and the 4 Stanza peer receipts
(`<treebank>/stanza.json`, generated at `bc426f41b7ef3714d36e2f24d4f1ca8ef1c0c56b`
and `f48a400a21417d9a712fa2de765e54e84c5d7b11`) are committed beside this README
under `en-ewt/`, `it-markit/`, `de-gsd/`, and `es-ancora/`.

## Dev-selected winning configurations vs majority baseline and Stanza

For each treebank, the winning configuration is selected strictly by highest
dev-split accuracy (`training.dev[training.best_epoch]`):

| treebank | encoder | dev-selected winner | dev | trees | lookup entries | test words | overall | seen | unseen | majority (seen / unseen) | delta vs majority (95% CI) | Stanza overall (seen / unseen) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `ud-en-ewt` | `answerdotai/ModernBERT-base` | `en-ewt/mtf3-lookup1.json` | 0.9576 | 337 | 18,697 (`k=1`) | 25,094 | **0.9579** | **0.9809** | **0.7286** | 0.9503 (0.9812 / 0.6431) | **+0.0076 [+0.0052, +0.0099]** | 0.9683 (0.9849 / 0.8028, `stanza-1.10.1`) |
| `ud-it-markit` | `FacebookAI/xlm-roberta-base` | `it-markit/mtf2-bilstm-lookup1.json` | 0.9397 | 185 | 3,980 (`k=1`) | 10,196 | **0.8530** | **0.8732** | **0.7152** | 0.8172 (0.8654 / 0.4893) | **+0.0358 [+0.0317, +0.0403]** | 0.8695 (0.8751 / 0.8308, `stanza-1.15.0-it_markit`) |
| `ud-de-gsd` | `FacebookAI/xlm-roberta-base` | `de-gsd/mtf2-bilstm-lookup1.json` | 0.9697 | 616 | 48,968 (`k=1`) | 16,499 | **0.9670** | **0.9920** | **0.7780** | 0.9424 (0.9901 / 0.5823) | **+0.0246 [+0.0221, +0.0271]** | 0.9743 (0.9912 / 0.8468, `stanza-1.15.0-de_gsd`) |
| `ud-es-ancora` | `FacebookAI/xlm-roberta-base` | `es-ancora/mtf2-bilstm-lookup1.json` | 0.9899 | 695 | 36,065 (`k=1`) | 53,622 | **0.9883** | **0.9949** | **0.8711** | 0.9677 (0.9897 / 0.5753) | **+0.0206 [+0.0193, +0.0220]** | 0.9900 (0.9922 / 0.9505, `stanza-1.15.0-es_ancora`) |

All four treebanks clear the majority baseline with the lower bound of the 95%
paired bootstrap interval strictly above zero (`low > 0`) under both `k = 2`
and `k = 1`.

## Full sweep tables

### `ud-en-ewt` (`answerdotai/ModernBERT-base`, 25,094 test words: 22,802 seen, 2,292 unseen; majority baseline `en-ewt/majority.json`: `0.9503`, seen `0.9812`, unseen `0.6431`)

| receipt | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `en-ewt/mtf3-nolookup.json` | 3 | none | none | 337 | 0 | 4 | 0.9373 | 0.9380 | 0.9590 | 0.7286 | -0.0124 | [-0.0155, -0.0094] |
| `en-ewt/mtf2-nolookup.json` | 2 | none | none | 469 | 0 | 4 | 0.9363 | 0.9372 | 0.9581 | 0.7299 | -0.0131 | [-0.0161, -0.0101] |
| `en-ewt/mtf1-nolookup.json` | 1 | none | none | 1,064 | 0 | 4 | 0.9371 | 0.9375 | 0.9588 | 0.7256 | -0.0128 | [-0.0157, -0.0100] |
| `en-ewt/mtf3-lookup2.json` | 3 | none | `k=2` | 337 | 8,896 | 4 | 0.9542 | 0.9549 | 0.9777 | 0.7286 | +0.0046 | [+0.0020, +0.0071] |
| `en-ewt/mtf2-lookup2.json` | 2 | none | `k=2` | 469 | 8,896 | 4 | 0.9537 | 0.9552 | 0.9779 | 0.7299 | +0.0049 | [+0.0025, +0.0073] |
| `en-ewt/mtf1-lookup2.json` | 1 | none | `k=2` | 1,064 | 8,896 | 4 | 0.9540 | 0.9545 | 0.9775 | 0.7256 | +0.0042 | [+0.0017, +0.0066] |
| `en-ewt/mtf3-lookup1.json` | 3 | none | `k=1` | 337 | 18,697 | 4 | **0.9576** | **0.9579** | **0.9809** | **0.7286** | **+0.0076** | **[+0.0052, +0.0099]** |
| `en-ewt/mtf2-lookup1.json` | 2 | none | `k=1` | 469 | 18,697 | 4 | 0.9572 | 0.9583 | 0.9813 | 0.7299 | +0.0080 | [+0.0057, +0.0102] |
| `en-ewt/mtf1-lookup1.json` | 1 | none | `k=1` | 1,064 | 18,697 | 4 | 0.9572 | 0.9577 | 0.9810 | 0.7256 | +0.0074 | [+0.0050, +0.0095] |
| `en-ewt/mtf2-bilstm-nolookup.json` | 2 | `256x1` | none | 469 | 0 | 4 | 0.9312 | 0.9307 | 0.9513 | 0.7260 | -0.0196 | [-0.0228, -0.0164] |
| `en-ewt/mtf2-bilstm-lookup2.json` | 2 | `256x1` | `k=2` | 469 | 8,896 | 4 | 0.9521 | 0.9517 | 0.9744 | 0.7260 | +0.0014 | [-0.0011, +0.0038] |
| `en-ewt/mtf2-bilstm-lookup1.json` | 2 | `256x1` | `k=1` | 469 | 18,697 | 4 | 0.9555 | 0.9552 | 0.9783 | 0.7260 | +0.0049 | [+0.0026, +0.0072] |
| `en-ewt/stanza.json` (`stanza-1.10.1`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9683 | 0.9849 | 0.8028 | +0.0180 | [+0.0158, +0.0202] |

### `ud-it-markit` (`FacebookAI/xlm-roberta-base`, 10,196 test words: 8,890 seen, 1,306 unseen; majority baseline `it-markit/majority.json`: `0.8172`, seen `0.8654`, unseen `0.4893`)

| receipt | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `it-markit/mtf3-nolookup.json` | 3 | none | none | 145 | 0 | 4 | 0.9132 | 0.8280 | 0.8535 | 0.6539 | +0.0108 | [+0.0065, +0.0153] |
| `it-markit/mtf2-nolookup.json` | 2 | none | none | 185 | 0 | 1 | 0.9186 | 0.8321 | 0.8543 | 0.6807 | +0.0149 | [+0.0103, +0.0192] |
| `it-markit/mtf1-nolookup.json` | 1 | none | none | 295 | 0 | 3 | 0.9203 | 0.8323 | 0.8571 | 0.6631 | +0.0151 | [+0.0107, +0.0194] |
| `it-markit/mtf3-lookup2.json` | 3 | none | `k=2` | 145 | 1,391 | 4 | 0.9207 | 0.8345 | 0.8611 | 0.6539 | +0.0174 | [+0.0132, +0.0216] |
| `it-markit/mtf2-lookup2.json` | 2 | none | `k=2` | 185 | 1,391 | 1 | 0.9242 | 0.8373 | 0.8603 | 0.6807 | +0.0201 | [+0.0157, +0.0241] |
| `it-markit/mtf1-lookup2.json` | 1 | none | `k=2` | 295 | 1,391 | 3 | 0.9251 | 0.8371 | 0.8627 | 0.6631 | +0.0199 | [+0.0159, +0.0239] |
| `it-markit/mtf3-lookup1.json` | 3 | none | `k=1` | 145 | 3,980 | 1 | 0.9295 | 0.8450 | 0.8737 | 0.6501 | +0.0279 | [+0.0243, +0.0316] |
| `it-markit/mtf2-lookup1.json` | 2 | none | `k=1` | 185 | 3,980 | 1 | 0.9323 | 0.8481 | 0.8727 | 0.6807 | +0.0309 | [+0.0272, +0.0346] |
| `it-markit/mtf1-lookup1.json` | 1 | none | `k=1` | 295 | 3,980 | 3 | 0.9314 | 0.8459 | 0.8728 | 0.6631 | +0.0287 | [+0.0250, +0.0325] |
| `it-markit/mtf2-bilstm-nolookup.json` | 2 | `256x1` | none | 185 | 0 | 4 | 0.9271 | 0.8368 | 0.8547 | 0.7152 | +0.0196 | [+0.0149, +0.0244] |
| `it-markit/mtf2-bilstm-lookup2.json` | 2 | `256x1` | `k=2` | 185 | 1,391 | 4 | 0.9331 | 0.8441 | 0.8630 | 0.7152 | +0.0269 | [+0.0224, +0.0317] |
| `it-markit/mtf2-bilstm-lookup1.json` | 2 | `256x1` | `k=1` | 185 | 3,980 | 4 | **0.9397** | **0.8530** | **0.8732** | **0.7152** | **+0.0358** | **[+0.0317, +0.0403]** |
| `it-markit/stanza.json` (`stanza-1.15.0-it_markit`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.8695 | 0.8751 | 0.8308 | +0.0523 | [+0.0475, +0.0573] |

### `ud-de-gsd` (`FacebookAI/xlm-roberta-base`, 16,499 test words: 14,567 seen, 1,932 unseen; majority baseline `de-gsd/majority.json`: `0.9424`, seen `0.9901`, unseen `0.5823`)

| receipt | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `de-gsd/mtf3-nolookup.json` | 3 | none | none | 506 | 0 | 4 | 0.9417 | 0.9367 | 0.9642 | 0.7298 | -0.0056 | [-0.0091, -0.0021] |
| `de-gsd/mtf2-nolookup.json` | 2 | none | none | 616 | 0 | 4 | 0.9413 | 0.9369 | 0.9642 | 0.7314 | -0.0055 | [-0.0088, -0.0021] |
| `de-gsd/mtf1-nolookup.json` | 1 | none | none | 929 | 0 | 4 | 0.9410 | 0.9378 | 0.9653 | 0.7308 | -0.0045 | [-0.0079, -0.0011] |
| `de-gsd/mtf3-lookup2.json` | 3 | none | `k=2` | 506 | 14,634 | 4 | 0.9590 | 0.9535 | 0.9831 | 0.7298 | +0.0111 | [+0.0085, +0.0137] |
| `de-gsd/mtf2-lookup2.json` | 2 | none | `k=2` | 616 | 14,634 | 4 | 0.9581 | 0.9530 | 0.9824 | 0.7314 | +0.0106 | [+0.0080, +0.0133] |
| `de-gsd/mtf1-lookup2.json` | 1 | none | `k=2` | 929 | 14,634 | 4 | 0.9583 | 0.9535 | 0.9830 | 0.7308 | +0.0111 | [+0.0085, +0.0137] |
| `de-gsd/mtf3-lookup1.json` | 3 | none | `k=1` | 506 | 48,968 | 4 | 0.9652 | 0.9611 | 0.9918 | 0.7298 | +0.0187 | [+0.0166, +0.0209] |
| `de-gsd/mtf2-lookup1.json` | 2 | none | `k=1` | 616 | 48,968 | 4 | 0.9646 | 0.9611 | 0.9916 | 0.7314 | +0.0188 | [+0.0166, +0.0211] |
| `de-gsd/mtf1-lookup1.json` | 1 | none | `k=1` | 929 | 48,968 | 4 | 0.9647 | 0.9612 | 0.9918 | 0.7308 | +0.0188 | [+0.0167, +0.0212] |
| `de-gsd/mtf2-bilstm-nolookup.json` | 2 | `256x1` | none | 616 | 0 | 4 | 0.9544 | 0.9526 | 0.9758 | 0.7780 | +0.0102 | [+0.0069, +0.0136] |
| `de-gsd/mtf2-bilstm-lookup2.json` | 2 | `256x1` | `k=2` | 616 | 14,634 | 4 | 0.9639 | 0.9606 | 0.9848 | 0.7780 | +0.0182 | [+0.0154, +0.0210] |
| `de-gsd/mtf2-bilstm-lookup1.json` | 2 | `256x1` | `k=1` | 616 | 48,968 | 4 | **0.9697** | **0.9670** | **0.9920** | **0.7780** | **+0.0246** | **[+0.0221, +0.0271]** |
| `de-gsd/stanza.json` (`stanza-1.15.0-de_gsd`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9743 | 0.9912 | 0.8468 | +0.0319 | [+0.0290, +0.0351] |

### `ud-es-ancora` (`FacebookAI/xlm-roberta-base`, 53,622 test words: 50,775 seen, 2,847 unseen; majority baseline `es-ancora/majority.json`: `0.9677`, seen `0.9897`, unseen `0.5753`)

| receipt | `--min-tree-freq` | `--bilstm` | `--seen-lookup` | trees | lookup entries | best epoch | dev | test | seen | unseen | delta vs majority | 95% CI |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `es-ancora/mtf3-nolookup.json` | 3 | none | none | 574 | 0 | 4 | 0.9754 | 0.9744 | 0.9837 | 0.8089 | +0.0068 | [+0.0051, +0.0084] |
| `es-ancora/mtf2-nolookup.json` | 2 | none | none | 695 | 0 | 4 | 0.9763 | 0.9750 | 0.9840 | 0.8142 | +0.0073 | [+0.0057, +0.0090] |
| `es-ancora/mtf1-nolookup.json` | 1 | none | none | 961 | 0 | 4 | 0.9761 | 0.9750 | 0.9839 | 0.8149 | +0.0073 | [+0.0057, +0.0089] |
| `es-ancora/mtf3-lookup2.json` | 3 | none | `k=2` | 574 | 17,513 | 4 | 0.9830 | 0.9818 | 0.9915 | 0.8089 | +0.0142 | [+0.0128, +0.0156] |
| `es-ancora/mtf2-lookup2.json` | 2 | none | `k=2` | 695 | 17,513 | 4 | 0.9836 | 0.9822 | 0.9917 | 0.8142 | +0.0146 | [+0.0131, +0.0160] |
| `es-ancora/mtf1-lookup2.json` | 1 | none | `k=2` | 961 | 17,513 | 4 | 0.9835 | 0.9822 | 0.9916 | 0.8149 | +0.0146 | [+0.0131, +0.0160] |
| `es-ancora/mtf3-lookup1.json` | 3 | none | `k=1` | 574 | 36,065 | 4 | 0.9853 | 0.9836 | 0.9934 | 0.8089 | +0.0159 | [+0.0147, +0.0173] |
| `es-ancora/mtf2-lookup1.json` | 2 | none | `k=1` | 695 | 36,065 | 4 | 0.9856 | 0.9839 | 0.9934 | 0.8142 | +0.0162 | [+0.0148, +0.0176] |
| `es-ancora/mtf1-lookup1.json` | 1 | none | `k=1` | 961 | 36,065 | 4 | 0.9855 | 0.9838 | 0.9932 | 0.8149 | +0.0161 | [+0.0148, +0.0174] |
| `es-ancora/mtf2-bilstm-nolookup.json` | 2 | `256x1` | none | 695 | 0 | 4 | 0.9869 | 0.9857 | 0.9921 | 0.8711 | +0.0180 | [+0.0165, +0.0195] |
| `es-ancora/mtf2-bilstm-lookup2.json` | 2 | `256x1` | `k=2` | 695 | 17,513 | 4 | 0.9891 | 0.9876 | 0.9942 | 0.8711 | +0.0200 | [+0.0186, +0.0213] |
| `es-ancora/mtf2-bilstm-lookup1.json` | 2 | `256x1` | `k=1` | 695 | 36,065 | 4 | **0.9899** | **0.9883** | **0.9949** | **0.8711** | **+0.0206** | **[+0.0193, +0.0220]** |
| `es-ancora/stanza.json` (`stanza-1.15.0-es_ancora`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9900 | 0.9922 | 0.9505 | +0.0224 | [+0.0207, +0.0240] |

## Dev-selected winning artifact digests

| treebank | receipt | `head.safetensors` SHA-256 | `head.json` SHA-256 | encoder `manifest_sha256` |
|---|---|---|---|---|
| `ud-en-ewt` | `en-ewt/mtf3-lookup1.json` | `1cba58c78ee009cfe0f866d0e585d54f694e293904e164bf3b25dba72f45b93b` | `be8c81065f89830793fa8cd28aadaddcd8428c1fe475478b0a919423d7625729` | `65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8` |
| `ud-it-markit` | `it-markit/mtf2-bilstm-lookup1.json` | `0c305b865db5e34440e7d7ea20d1f233723c5d2af5711e496921f43bdb38fcc9` | `d8e8f70560ba855a886cc1146979daf35c19812b531332398de7ee9a3e2ce461` | `4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52` |
| `ud-de-gsd` | `de-gsd/mtf2-bilstm-lookup1.json` | `918a79d49ec12466c28f719a7401e36eed8de4749a84dfb39802c5f3a4bb22dc` | `1701aa7bf53e5d6eef5845248151a119215c70a3292be1db4094bb59ac177251` | `4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52` |
| `ud-es-ancora` | `es-ancora/mtf2-bilstm-lookup1.json` | `b1a569fb8d1a25dc29a729e0c7c4ce97cd657657400de5812afa7e8cc43731e8` | `c3530dfe640df8da2cbc60833a782253cd42c3fe2f3b39272e03c23e4a1397b4` | `4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52` |

## Reproduce

```sh
turbo-train-head baseline --corpus ud-en-ewt-lemma \
  --dir UD_English-EWT-r2.18 --out en-ewt/majority

turbo-train-head train-lemma --bundle modernbert-base \
  --corpus ud-en-ewt --dir UD_English-EWT-r2.18 \
  --out en-ewt/mtf3-lookup1 --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 3 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-it-markit --dir UD_Italian-MarkIT-r2.16 \
  --out it-markit/mtf2-bilstm-lookup1 --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-de-gsd --dir UD_German-GSD-r2.16 \
  --out de-gsd/mtf2-bilstm-lookup1 --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda

turbo-train-head train-lemma --bundle xlm-roberta-base \
  --corpus ud-es-ancora --dir UD_Spanish-AnCora-r2.16 \
  --out es-ancora/mtf2-bilstm-lookup1 --seed 42 --epochs 5 --lr 0.05 \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1 --device cuda
```
