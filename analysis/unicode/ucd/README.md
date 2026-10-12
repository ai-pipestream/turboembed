# Unicode Character Database inputs

These files are the inputs of `turbo-unicode-gen`, which generates the
tables in `analysis/unicode/src/tables/`. They are copied unchanged from
the Unicode 18.0.0 release and are used under the Unicode License v3
(`LICENSE` in this directory). `SHA256SUMS` lists each file's digest.

| File | Source |
|---|---|
| `UnicodeData.txt`, `CaseFolding.txt`, `SpecialCasing.txt`, `DerivedNormalizationProps.txt`, `DerivedCoreProperties.txt`, `NormalizationTest.txt`, `PropList.txt`, `Scripts.txt`, `CompositionExclusions.txt` | UCD 18.0.0, `https://www.unicode.org/Public/18.0.0/ucd/` |
| `DerivedGeneralCategory.txt`, `DerivedCombiningClass.txt` | UCD 18.0.0, `ucd/extracted/` |
| `emoji-data.txt`, `emoji-variation-sequences.txt` | UCD 18.0.0, `ucd/emoji/` |
| `emoji-sequences.txt`, `emoji-zwj-sequences.txt`, `emoji-test.txt` | Unicode Emoji 18.0, `https://www.unicode.org/Public/emoji/18.0/` |
| `confusables.txt` | Unicode Security Mechanisms 18.0.0 (UTS #39), `https://www.unicode.org/Public/security/18.0.0/` |

They were fetched on 2026-10-12 from the `unicode-org/unicodetools`
repository on GitHub (`unicodetools/data/`), which carries the same
release files, because the container that did the work could not reach
`unicode.org` directly.

To regenerate the tables after replacing these files with a newer
release:

```sh
cargo run -p turbo-unicode-gen -- analysis/unicode/ucd analysis/unicode/src/tables
```

A test in `turbo-unicode` runs the generator over this directory and
fails when the committed tables differ from its output.
