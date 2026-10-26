# Finnish-FTB dependency recipe, UD v2.14

This is a reproducible training/evaluation starting point, not a released model
or an accuracy claim. Both model types retain crossing trees through SWAP.
The fixed parameters have not been tuned or evaluated for this treebank.
Issue #922 still requires trained artifacts, original comparison gates and
release/model-card evidence. Do not publish a model based on oracle replay.

## Source and license

Tag `r2.14` of [UD Finnish-FTB](https://github.com/UniversalDependencies/UD_Finnish-FTB/tree/r2.14)
resolves to commit `948615ba5289828a8d5cf7161d9cedf803e9efe6`.
`SHA256SUMS` pins the exact train, development, test, license and README files.
[LICENSE.txt](https://github.com/UniversalDependencies/UD_Finnish-FTB/blob/948615ba5289828a8d5cf7161d9cedf803e9efe6/LICENSE.txt)
offers LGPLv3+ or CC BY 4.0; this recipe chooses the **CC BY 4.0 alternative**.
It applies to the treebank including text. Attribution: Copyright 2014
FIN-CLARIN, University of Helsinki. Unannotated text is adapted from VISK
examples; the FIN-CLARIN projects produced and revised annotations.

VISK: Auli Hakulinen, Maria Vilkuna, Riitta Korhonen, Vesa Koivisto,
Tarja Riitta Heinonen and Irja Alho (2004), *Iso suomen kielioppi*.
Helsinki: Suomalaisen Kirjallisuuden Seura, online edition,
URN:ISBN:978-952-5446-35-7. UD contributors: Jussi Piitulainen, Hanna Nurmi.
The [primary license grant](https://www.kielipankki.fi/lic/uralic-ud2-13-eng/)
also lists Finnish-FTB under CC BY 4.0. Preserve attribution, license links,
source hashes and descriptions of any changes with distributed derivatives.
This selection says nothing about other UD treebanks' licenses.

The genre is Finnish grammar examples, not English or general news accuracy.
The upstream split and its overlap-removal history are in the pinned README.
Use train only for fitting, development only for selecting settings/floors,
and test only for the frozen final comparison. No train+dev merge is implicit.

## Commands (from repository root)

```sh
analysis/train-recipes/dependency/fi-ftb-r2.14/prepare.sh target/fi-ftb-r2.14
cargo run -p turbo-depparse --example audit_treebank -- \
  target/fi-ftb-r2.14/fi_ftb-ud-train.conllu \
  target/fi-ftb-r2.14/fi_ftb-ud-dev.conllu \
  target/fi-ftb-r2.14/fi_ftb-ud-test.conllu
cargo build --release -p turbo-depparse --bin turbo-depparse
# These next commands start full training; preparation never runs them.
target/release/turbo-depparse train linear fi \
  target/fi-ftb-r2.14/fi_ftb-ud-train.conllu \
  analysis/train-recipes/dependency/fi-ftb-r2.14/linear.properties target/ftb-linear.bin
target/release/turbo-depparse train feedforward fi \
  target/fi-ftb-r2.14/fi_ftb-ud-train.conllu \
  analysis/train-recipes/dependency/fi-ftb-r2.14/neural.properties target/ftb-neural.turbo
# Set both floors from the approved comparison protocol before seeing test scores.
: "${MIN_UAS:?Set approved UAS floor}" "${MIN_LAS:?Set approved LAS floor}"
target/release/turbo-depparse evaluate feedforward target/ftb-neural.turbo \
  target/fi-ftb-r2.14/fi_ftb-ud-test.conllu target/ftb-neural-test.tsv "$MIN_UAS" "$MIN_LAS"
```

Replace `target/release` when setting a separate `CARGO_TARGET_DIR`. Use `linear`
and its model path for the other parser. Retain stdout/stderr, source commit,
compiler/build information, parameters, wall time and peak RSS. The CLI prints
model/corpus/parameter SHA256; evaluation records model/corpus hashes and floors.
A gate failure exits nonzero **after writing the complete report**.
No Python is needed by these operations. Full training has not been run for
this recipe; a small compatibility pilot is not its accuracy result.

## Fixed-input scoring protocol

Both parsers consume the same integer-ID syntactic words (`FORM`) and gold
`UPOS`. MWT range rows and empty nodes stay in source provenance but are not
basic dependency nodes. Gold `HEAD` and `DEPREL` are scoring targets, never
inference features. LAS uses exact relation strings, including subtypes;
UAS includes root attachment. Report token-weighted scores both including
punctuation and excluding precisely `UPOS=PUNCT`. Keep every sentence,
including non-projective ones. A failed/invalid parse receives zero credit
on all its words; it never shrinks the denominator and always fails the gate.
Report projective and non-projective subsets alongside the complete set.

Future MaltParser and Stanza comparisons must use these exact words, POS,
labels, punctuation policy and split hashes. Disable their automatic token/POS
replacement, record versions/model hashes and include failures. A downloaded
model with overlapping training data is not an independent held-out baseline.
No Malt/Stanza result is claimed by this recipe, and the absolute-floor CLI
is not a substitute for the issue's paired external-reference comparison.

## Structural audit (not learned accuracy)

Real Rust CoNLL parsing and oracle replay matched every gold head and label:

| Split | Sentences | Words | Non-projective sentences | Non-projective words | SWAP actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 14981 | 127602 | 1637 | 20590 | 5139 |
| dev | 1875 | 15724 | 210 | 2761 | 652 |
| test | 1867 | 16286 | 200 | 2436 | 596 |

Largest action counts were 250, 100 and 146 respectively. This audit uses gold
trees to verify the transition system, and therefore is **not** model UAS/LAS.
It makes no throughput or trained-model quality claim.
