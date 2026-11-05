# Italian MarkIT token split head, the #1803 sweep

The `ud-it-markit` split head against its per-token majority baseline
(test split, 340 sentences, 9,654 tokens, baseline accuracy 0.99254),
closing #1803. Every run at commit
`e97a161b5f1c8f903e2a64d5cf199952dda308ea` on the sealed
`FacebookAI/xlm-roberta-base` bundle (manifest
`4c5a35e444094f547df03c0e1745a0d13375f17e06430f1d2dd0c185a643ad52`), CPU,
f32, defaults (seed 1, batch 16, lr 0.002, patience 6), on the recorded
UD_Italian-MarkIT r2.16 files (`recorded: true`; 611 training sentences,
18,759 tokens). Each receipt is one `turbo-train-head train-split` run;
the commands name directories in place of the training machine's paths,
as the other records do. Epoch numbers in the table are counted from 1.

| run | flags | labels | seen table | dev best (epoch) | test model | delta vs baseline [95%] |
|---|---|---|---|---|---|---|
| `mtf2` | `--min-tree-freq 2` | 42 | | 0.99120 (20) | 0.99016 | -0.00238 [-0.00380, -0.00107] |
| `bilstm` | `--bilstm 256x1` | 34 | | 0.99563 (9) | 0.99316 | +0.00062 [-0.00063, +0.00193] |
| `mtf2-bilstm` | both | 42 | | 0.99502 (8) | 0.99358 | +0.00104 [-0.00020, +0.00227] |
| `seen1` | `--seen-lookup 1` | 34 | 4,113 | 0.99454 (17) | 0.99192 | -0.00062 [-0.00166, +0.00041] |
| `seen2` | `--seen-lookup 2` | 34 | 1,500 | 0.99345 (17) | 0.99078 | -0.00176 [-0.00299, -0.00061] |
| `mtf2-seen2` | `--min-tree-freq 2 --seen-lookup 2` | 42 | 1,501 | 0.99321 (22) | 0.99140 | -0.00114 [-0.00230, +0.00000] |
| `bilstm-seen1` | `--bilstm 256x1 --seen-lookup 1` | 34 | 4,113 | 0.99662 (10) | **0.99389** | **+0.00135 [+0.00021, +0.00245]** |
| `full1` | `--min-tree-freq 2 --bilstm 256x1 --seen-lookup 1` | 42 | 4,122 | 0.99618 (13) | **0.99389** | **+0.00135 [+0.00021, +0.00248]** |
| `full2` | `--min-tree-freq 2 --bilstm 256x1 --seen-lookup 2` | 42 | 1,501 | 0.99576 (13) | 0.99368 | +0.00114 [+0.00000, +0.00227] |

Item 1 alone does not clear the gate: `--min-tree-freq 2` loses (the
extra labels are rare splits the head still misses) and the BiLSTM beats
the baseline but its interval straddles zero. Item 2 alone does not
either: the lookup is the baseline's equal on unambiguous seen tokens
and the head still decides the rest, which is where it had trailed.
Together they clear it: with the lookup at k=1 the BiLSTM head's errors
on seen tokens are the tokens' unambiguous splits, so the model keeps
the head's gain on unseen tokens (1,292 of 1,335 where the baseline has
1,280) and stops losing on seen ones. `full1` is the recorded
configuration; a second run of the same command at the same commit gave
the same receipt's eval and head and the same `head.safetensors` bytes.

Output digests (sha256) of the clearing runs:

| run | head.json | head.safetensors |
|---|---|---|
| `bilstm-seen1` | `1c9f684aed844876ba721767d944b3ba71109434b35f15d8cd63bf13621377d2` | `50b28e1af303e00af15e5eff7f41c3a2bb17234c0e51157930b4437d95627e8b` |
| `full1` | `38a302aab4e353c641267427411aed2f60112d669b03d59ad146710fd886ee51` | `28abab2a002967e6fc441a81b875af5fc1dca98c55c50eddc800659513d3b320` |

## Reproduce

```sh
turbo-train-head train-split --corpus ud-it-markit --dir UD_Italian-MarkIT-r2.16 \
  --bundle xlm-roberta-base --out <new dir> --commit e97a161b5f1c8f903e2a64d5cf199952dda308ea \
  --min-tree-freq 2 --bilstm 256x1 --seen-lookup 1
```

Each other row is the same command with its own flags. The seen-token
lookup is a decode option carried in the head's spec (`seen` in
head.json and the bundle's manifest, docs/bundle.md): a token whose
every training occurrence split the same way at least `min_count` times
takes that split's label, the scorer decides the rest, and the receipt's
model row is scored with it on.
