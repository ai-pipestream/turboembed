# UD heads in the CC BY 4.0 default tier (#1522)

Heads trained by `turbo-train-head` on UD r2.16 treebanks (`ud_banks`),
each over frozen xlm-roberta-base (bundle manifest `4c5a35e4`, 12
layers), first-piece pooling and a BiLSTM 256x1 adapter (UPOS heads also
a 64-wide hashed word identity), on CUDA (RTX 4080 SUPER) at F32. The
Spanish UPOS head ran at commit `990fa0a5`, the rest at `75b4af00`. The
treebanks are CC BY 4.0 (Finnish FTB under its CC BY 4.0 option), so the
heads are permissive. Receipts name directories, not paths, and no host.

Test scores, each against the treebank's majority row, with the paired
bootstrap over documents (95%); dependencies are LAS, with UAS in the
receipt's `eval.uas`:

| treebank | head | receipt | test | majority | vs majority | head.safetensors |
|---|---|---|---|---|---|---|
| Spanish AnCora | UPOS | `es-ancora-upos.json` | 0.9878 | 0.9176 | +0.0703 [+0.0678, +0.0728] | `3b764672...` |
| Spanish AnCora | lemma | `es-ancora-lemma.json` | 0.9864 | 0.9677 | +0.0187 [+0.0171, +0.0204] | `a6d99a19...` |
| Spanish AnCora | dependencies (UAS 0.9246) | `es-ancora-deps.json` | 0.8993 | 0.3873 | +0.5120 [+0.5074, +0.5162] | `04b1f024...` |
| Catalan AnCora | UPOS | `ca-ancora-upos.json` | 0.9882 | 0.9198 | +0.0684 [+0.0657, +0.0709] | `3a2665a2...` |
| Catalan AnCora | lemma | `ca-ancora-lemma.json` | 0.9861 | 0.9701 | +0.0160 [+0.0145, +0.0174] | `88492b26...` |
| Catalan AnCora | dependencies (UAS 0.9347) | `ca-ancora-deps.json` | 0.9128 | 0.4087 | +0.5041 [+0.4990, +0.5088] | `78e4a834...` |
| Finnish FTB | UPOS | `fi-ftb-upos.json` | 0.9463 | 0.8200 | +0.1263 [+0.1201, +0.1321] | `28884343...` |
| Finnish FTB | lemma | `fi-ftb-lemma.json` | 0.8208 | 0.7928 | +0.0280 [+0.0220, +0.0343] | `e7871fca...` |
| Finnish FTB | dependencies (UAS 0.9001) | `fi-ftb-deps.json` | 0.8537 | 0.3849 | +0.4688 [+0.4594, +0.4783] | `7ec1c08a...` |
| Italian MarkIT | UPOS | `it-markit-upos.json` | 0.9555 | 0.8441 | +0.1114 [+0.1052, +0.1183] | `2f5ac640...` |
| Italian MarkIT | lemma | `it-markit-lemma.json` | 0.8310 | 0.8172 | +0.0138 [+0.0086, +0.0188] | `4b30ac68...` |
| Italian MarkIT | dependencies (UAS 0.8325) | `it-markit-deps.json` | 0.7733 | 0.3305 | +0.4428 [+0.4302, +0.4562] | `c51ad012...` |

Each head clears its majority row with the whole interval above zero.
MarkIT trains on 611 sentences, and Finnish lemmas are the hardest
column here; the lemma heads gain least over their baselines. Stanza and
spaCy peer rows for these test files are not recorded yet. The full
SHA-256 of each head file is pinned in `analysis/pins/models.txt`.
