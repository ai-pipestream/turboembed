# UD heads on CC BY-SA treebanks, download after acceptance (#1651)

Heads trained by `turbo-train-head` on the UD r2.16 treebanks
docs/licences.md lists for these languages (`ud_banks`): German, French and
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

## The other 18 treebanks

The same recipe on the remaining CC BY-SA r2.16 treebanks docs/licences.md
lists for #1651, at commit `a3f73995` (the same trainer, with `--against`
added, #1828), same encoder bundle, device and settings:

| treebank | head | receipt | test | majority | vs majority (paired bootstrap, 95%) | head.safetensors |
|---|---|---|---|---|---|---|
| Afrikaans AfriBooms | UPOS | `af-afribooms-upos.json` | 0.9758 | 0.9043 | +0.0714 [+0.0652, +0.0779] | `34a3cb87...` |
| Afrikaans AfriBooms | lemma | `af-afribooms-lemma.json` | 0.9460 | 0.9450 | +0.0010 [-0.0041, +0.0057] | `6115be18...` |
| Armenian BSUT | UPOS | `hy-bsut-upos.json` | 0.9437 | 0.8158 | +0.1278 [+0.1209, +0.1347] | `c4de994e...` |
| Armenian BSUT | lemma | `hy-bsut-lemma.json` | 0.8750 | 0.7586 | +0.1164 [+0.1075, +0.1255] | `4193b518...` |
| Croatian SET | UPOS | `hr-set-upos.json` | 0.9785 | 0.8871 | +0.0914 [+0.0865, +0.0959] | `6d0b6f5e...` |
| Croatian SET | lemma | `hr-set-lemma.json` | 0.9158 | 0.8797 | +0.0362 [+0.0317, +0.0404] | `6b1dccd3...` |
| Danish DDT | UPOS | `da-ddt-upos.json` | 0.9721 | 0.8546 | +0.1174 [+0.1105, +0.1247] | `51c08d0b...` |
| Danish DDT | lemma | `da-ddt-lemma.json` | 0.9341 | 0.9003 | +0.0337 [+0.0276, +0.0399] | `f463a812...` |
| Georgian GLC | UPOS | `ka-glc-upos.json` | 0.9528 | 0.8642 | +0.0886 [+0.0806, +0.0968] | `cadf10c4...` |
| Georgian GLC | lemma | `ka-glc-lemma.json` | 0.8610 | 0.8140 | +0.0470 [+0.0393, +0.0551] | `f95e8844...` |
| Icelandic IcePaHC | UPOS | `is-icepahc-upos.json` | 0.9609 | 0.8912 | +0.0697 [+0.0678, +0.0715] | `0a1da632...` |
| Icelandic IcePaHC | lemma | `is-icepahc-lemma.json` | 0.9142 | 0.9045 | +0.0096 [+0.0080, +0.0112] | `116a9d40...` |
| Indonesian GSD | UPOS | `id-gsd-upos.json` | 0.9411 | 0.8421 | +0.0989 [+0.0909, +0.1071] | `bde482b6...` |
| Indonesian GSD | lemma | `id-gsd-lemma.json` | 0.9593 | 0.8932 | +0.0661 [+0.0578, +0.0738] | `7e67707a...` |
| Irish IDT | UPOS | `ga-idt-upos.json` | 0.9495 | 0.8807 | +0.0688 [+0.0612, +0.0768] | `d80a19ca...` |
| Irish IDT | lemma | `ga-idt-lemma.json` | 0.9082 | 0.9020 | +0.0062 [+0.0001, +0.0121] | `6c533635...` |
| Korean Kaist | UPOS | `ko-kaist-upos.json` | 0.9164 | 0.7816 | +0.1349 [+0.1299, +0.1403] | `c9c09aa2...` |
| Korean Kaist | lemma | `ko-kaist-lemma.json` | 0.8359 | 0.7269 | +0.1090 [+0.1037, +0.1145] | `ef583ecd...` |
| Latvian LVTB | UPOS | `lv-lvtb-upos.json` | 0.9700 | 0.8834 | +0.0865 [+0.0833, +0.0897] | `246d751c...` |
| Latvian LVTB | lemma | `lv-lvtb-lemma.json` | 0.9046 | 0.8646 | +0.0400 [+0.0363, +0.0435] | `5696ed81...` |
| Norwegian Bokmaal | UPOS | `no-bokmaal-upos.json` | 0.9796 | 0.8942 | +0.0853 [+0.0818, +0.0891] | `085dcd24...` |
| Norwegian Bokmaal | lemma | `no-bokmaal-lemma.json` | 0.9597 | 0.9405 | +0.0191 [+0.0162, +0.0218] | `805a1e17...` |
| Persian PerDT | UPOS | `fa-perdt-upos.json` | 0.9734 | 0.9371 | +0.0363 [+0.0331, +0.0397] | `c154bd32...` |
| Persian PerDT | lemma | `fa-perdt-lemma.json` | 0.9838 | 0.9776 | +0.0063 [+0.0043, +0.0082] | `af245a1c...` |
| Romanian RRT | UPOS | `ro-rrt-upos.json` | 0.9751 | 0.9050 | +0.0701 [+0.0656, +0.0746] | `2749c883...` |
| Romanian RRT | lemma | `ro-rrt-lemma.json` | 0.9449 | 0.8998 | +0.0451 [+0.0399, +0.0505] | `6e788284...` |
| Serbian SET | UPOS | `sr-set-upos.json` | 0.9817 | 0.8804 | +0.1013 [+0.0943, +0.1079] | `13dd971a...` |
| Serbian SET | lemma | `sr-set-lemma.json` | 0.9069 | 0.8691 | +0.0378 [+0.0309, +0.0444] | `544aa93b...` |
| Slovak SNK | UPOS | `sk-snk-upos.json` | 0.9629 | 0.6194 | +0.3434 [+0.3351, +0.3528] | `ec5ae843...` |
| Slovak SNK | lemma | `sk-snk-lemma.json` | 0.8770 | 0.6935 | +0.1835 [+0.1759, +0.1912] | `4b4805cb...` |
| Slovenian SSJ | UPOS | `sl-ssj-upos.json` | 0.9821 | 0.8819 | +0.1001 [+0.0961, +0.1041] | `40ce3ff5...` |
| Slovenian SSJ | lemma | `sl-ssj-lemma.json` | 0.9272 | 0.8714 | +0.0558 [+0.0509, +0.0605] | `f2a97e59...` |
| Swedish Talbanken | UPOS | `sv-talbanken-upos.json` | 0.9796 | 0.8736 | +0.1061 [+0.1016, +0.1103] | `2f6a74f1...` |
| Swedish Talbanken | lemma | `sv-talbanken-lemma.json` | 0.9319 | 0.9042 | +0.0278 [+0.0236, +0.0319] | `a786c54b...` |
| Turkish BOUN | UPOS | `tr-boun-upos.json` | 0.9265 | 0.8274 | +0.0991 [+0.0921, +0.1059] | `accbc11c...` |
| Turkish BOUN | lemma | `tr-boun-lemma.json` | 0.8423 | 0.7785 | +0.0638 [+0.0564, +0.0714] | `d60d6ea0...` |

Every UPOS head and 17 of the 18 lemma heads clear their majority rows
with the whole interval above zero. Afrikaans AfriBooms lemma does not:
its majority row is already 0.9450 and the head's gain of 0.0010 is
inside the noise, so its receipt is kept but the head is not pinned and
is not offered. Afrikaans keeps the no-model lemma tier until a head
clears it. Irish IDT lemma clears by the narrowest margin of the set
(lower bound +0.0001). Kazakh, the 24th language of the ticket, is not
trained: KTB r2.16 has 31 training sentences and no dev split
(docs/licences.md).

Sentence, token-split and dependency heads for these languages and
Stanza peer rows follow. The full SHA-256 of each pinned head file is in
`analysis/pins/models.txt`.
