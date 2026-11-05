# UD English EWT heads (#1523)

Heads trained by `turbo-train-head` on UD_English-EWT r2.18 (2,077 test
sentences, 25,094 words), each over frozen ModernBERT-base (bundle
manifest `65a07202`, 22 layers), first-piece pooling and a BiLSTM 256x1
adapter (the tag heads also a 64-wide hashed word identity), on CUDA
(RTX 4080 SUPER) at F32. EWT is CC BY-SA 4.0, so every head here is
share-alike: offered for download after acceptance, never in a default
bundle. Receipts name directories, not paths, and no host.

| head | receipt | commit | test | majority | vs majority (paired bootstrap, 95%) | head.safetensors |
|---|---|---|---|---|---|---|
| UPOS | `upos.json` | `aeab6c0c` | 0.9457 | 0.8617 | +0.0840 [+0.0789, +0.0890] | `3fde6d9b...` |
| XPOS | `xpos.json` | `990fa0a5` | 0.9408 | 0.8381 | +0.1027 [+0.0976, +0.1077] | `78bac119...` |
| feats | `feats.json` | `990fa0a5` | 0.9531 | 0.8292 | +0.1239 [+0.1188, +0.1289] | `a1f98dfd...` |
| lemma (edit trees) | `lemma.json` | `75b4af00` | 0.9543 | 0.9503 | +0.0039 [+0.0010, +0.0068] | `620111de...` |
| dependencies, LAS (UAS 0.8403) | `deps.json` | `75b4af00` | 0.7901 | 0.3322 | +0.4579 [+0.4486, +0.4675] | `45cce91e...` |

A variant, the UPOS head over the 12-layer cut of ModernBERT-base
(manifest `3c520dec`, `upos-l12.json`, `75b4af00`): 0.9427, +0.0811
[+0.0762, +0.0859], under the 22-layer head's 0.9457, so the full
encoder stays the default for EWT tagging.

Peer rows on the same gold words (trainer/recipes/records/1825):

| row | UPOS | XPOS | feats | lemma | LAS |
|---|---|---|---|---|---|
| spaCy trf | 0.9326 | 0.9532 | 0.6766 | 0.9531 | 0.3795 |
| Stanza | 0.9677 | 0.9624 | 0.9622 | 0.9683 | 0.8835 |
| Trankit | 0.9649 | 0.9700 | 0.8534 | 0.9138 | 0.8578 |
| ours | 0.9457 | 0.9408 | 0.9531 | 0.9543 | 0.7901 |

The frozen-encoder heads beat spaCy on every column but XPOS, and
Trankit on feats and lemma, and sit under Stanza on all five; LAS is the
widest gap (0.79 against 0.88), which is R2 fine-tuning's to close. The
full SHA-256 of each head file is pinned in `analysis/pins/models.txt`.
