# UD heads in the CC BY 4.0 default tier (#1522)

Heads trained by `turbo-train-head` on UD r2.16 treebanks (`ud_banks`),
each over frozen xlm-roberta-base (bundle manifest `4c5a35e4`, 12
layers), first-piece pooling and a BiLSTM 256x1 adapter (UPOS heads also
a 64-wide hashed word identity), on CUDA (RTX 4080 SUPER) at F32, commit
`990fa0a5`. Receipts name directories, not paths, and no host.

| treebank | head | receipt | test | majority | vs majority (paired bootstrap, 95%) | head.safetensors |
|---|---|---|---|---|---|---|
| Spanish AnCora | UPOS | `es-ancora-upos.json` | 0.9878 | 0.9176 | +0.0703 [+0.0678, +0.0728] | `3b764672...` |

The remaining heads (lemma and parse for Spanish, Catalan, Finnish and
Italian) follow as their runs finish. The full SHA-256 of each head file
is pinned in `analysis/pins/models.txt`.
