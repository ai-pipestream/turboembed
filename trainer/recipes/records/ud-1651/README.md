# UD heads on CC BY-SA treebanks, download after acceptance (#1651)

Heads trained by `turbo-train-head` on the UD r2.16 treebanks the
retired OpenNLP UD models used (`ud_banks`): German, French and
Portuguese GSD, Dutch Alpino, Russian GSD. Each is over frozen
xlm-roberta-base (bundle manifest `4c5a35e4`, 12 layers), first-piece
pooling and a BiLSTM 256x1 adapter (UPOS heads also a 64-wide hashed word
identity), on CUDA (RTX 4080 SUPER) at F32, commit `75b4af00`. The
treebanks are CC BY-SA 4.0, so every head here is share-alike: offered
for download after acceptance, never in a default bundle. Receipts name
directories, not paths, and no host.

| treebank | head | receipt | test | majority | vs majority (paired bootstrap, 95%) | head.safetensors |
|---|---|---|---|---|---|---|
| German GSD | UPOS | `de-gsd-upos.json` | 0.9620 | 0.8865 | +0.0755 [+0.0708, +0.0807] | `d22b7d7c...` |
| German GSD | lemma | `de-gsd-lemma.json` | 0.9516 | 0.9424 | +0.0093 [+0.0059, +0.0127] | `94b8a57a...` |
| French GSD | UPOS | `fr-gsd-upos.json` | 0.9803 | 0.9132 | +0.0672 [+0.0613, +0.0734] | `4b9f2600...` |
| French GSD | lemma | `fr-gsd-lemma.json` | 0.9741 | 0.9593 | +0.0149 [+0.0115, +0.0183] | `f84c1e5e...` |
| Portuguese GSD | UPOS | `pt-gsd-upos.json` | 0.9738 | 0.8803 | +0.0935 [+0.0894, +0.0976] | `6ced685b...` |
| Portuguese GSD | lemma | `pt-gsd-lemma.json` | 0.9033 | 0.8761 | +0.0271 [+0.0227, +0.0317] | `cb11be50...` |
| Dutch Alpino | UPOS | `nl-alpino-upos.json` | 0.9624 | 0.8806 | +0.0818 [+0.0754, +0.0881] | `bcb46229...` |
| Dutch Alpino | lemma | `nl-alpino-lemma.json` | 0.9160 | 0.9076 | +0.0084 [+0.0041, +0.0128] | `23ff601d...` |
| Russian GSD | UPOS | `ru-gsd-upos.json` | 0.9721 | 0.8072 | +0.1649 [+0.1565, +0.1726] | `49e91628...` |
| Russian GSD | lemma | `ru-gsd-lemma.json` | 0.8996 | 0.7969 | +0.1027 [+0.0948, +0.1109] | `8366b0b7...` |

Each head clears its majority row with the whole interval above zero.
Sentence, token-split and dependency heads for these languages, the
other 19 languages of #1651, and Stanza peer rows follow. The full
SHA-256 of each head file is pinned in `analysis/pins/models.txt`.
