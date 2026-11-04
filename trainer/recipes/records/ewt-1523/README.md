# UD English EWT heads (#1523)

Heads trained by `turbo-train-head` on UD_English-EWT r2.18 (2,077 test
sentences, 25,094 words), each over frozen ModernBERT-base (bundle
manifest `65a07202`, 22 layers), first-piece pooling, a 64-wide hashed
word identity and a BiLSTM 256x1 adapter, on CUDA (RTX 4080 SUPER) at
F32. EWT is CC BY-SA 4.0, so every head here is share-alike: offered for
download after acceptance, never in a default bundle. Receipts name
directories, not paths, and no host.

| head | receipt | commit | test | majority | vs majority (paired bootstrap, 95%) | head.safetensors |
|---|---|---|---|---|---|---|
| UPOS | `upos.json` | `aeab6c0c` | 0.9457 | 0.8617 | +0.0840 [+0.0789, +0.0890] | `3fde6d9b...` |
| XPOS | `xpos.json` | `990fa0a5` | 0.9408 | 0.8381 | +0.1027 [+0.0976, +0.1077] | `78bac119...` |
| feats | `feats.json` | `990fa0a5` | 0.9531 | 0.8292 | +0.1239 [+0.1188, +0.1289] | `a1f98dfd...` |

Peer rows on the same gold words (trainer/recipes/records/1825): UPOS
spaCy trf 0.9326, Stanza 0.9677, Trankit 0.9649; XPOS 0.9532, 0.9624,
0.9700; feats 0.6766, 0.9622, 0.8534. The heads beat spaCy on all three
and Trankit on feats, and sit under Stanza on all three (a frozen
encoder; R2 fine-tuning is the next step for the gap).

Lemma and dependency heads follow as their runs finish. The full SHA-256
of each head file is pinned in `analysis/pins/models.txt`.
