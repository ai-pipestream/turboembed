# turbo-hunspell

OpenNLP's Hunspell stemmer, ported to Rust: a reader of Hunspell `.aff` and
`.dic` files (`HunspellDictionary`, with `AffixCondition`,
`HunspellAffixTable`, `HunspellCompoundRule` and `HunspellConversion`), the
stemmer (`HunspellStemmer`) and an analysis step, `HunspellAnnotator`, that
reads `opennlp:tokens` and writes `turbo:stems` (the layer is in
`turbo-stem-layers`). The crate documentation (`src/lib.rs`) describes the
modules and where the port differs from OpenNLP.

It follows the classes of the `OPENNLP-1927-hunspell-compatibility` branch of
the OpenNLP fork at commit `fb299cf32ccf6f5dbc81b39448d2d7bea1cf66dc`, which
reads more of the format than the `HunspellStemmer` of the commit the other
crates follow. The reader is written for this crate from OpenNLP's classes;
it has no dependency beyond the workspace's own crates.

## Dictionaries and their licences

No dictionary is checked in or shipped with the crate. The goldens use three
of LibreOffice's dictionaries, which `turbo-goldens fetch-models <dir>`
downloads into `<dir>/hunspell/` from the `LibreOffice/dictionaries`
repository at commit `67032ee957631908098a72edf48dc81cac27e369` and checks
against the SHA-512 in `HUNSPELL_FILES` of
`analysis/goldens-tool/src/recipe/models.rs`:

| Dictionary | Licence (the README beside it in that repository) |
|---|---|
| `en_US` | SCOWL's notice (Kevin Atkinson and the word lists it names): permissive, MIT-like |
| `de_DE_frami` | igerman98: GPL-2.0 or GPL-3.0 |
| `hu_HU` | Magyar Ispell 1.9: MPL-2.0 or LGPL-3.0 or later |

Only the goldens made with them are checked in: the words of the inputs and
the stems and analyses OpenNLP returned. The `ci` workflow caches the
download directory under the hash of the recipe directory
`analysis/goldens-tool/src/recipe/`, which holds the digests, and of
`analysis/goldens-tool/src/models.rs`, which fetches them, and runs this
crate's tests with the dictionaries required.

## Snapshots and unit tests

`tests/snapshots.rs` pins Dictionary/Stemmer output for each fixture and
published dictionary against the checked-in files under `tests/snapshots/`.
`tests/units.rs` covers listed-form stemming, circumfix/forbidden/needaffix,
`COMPOUNDRULE`, and the annotator's per-token stems layer. Fixtures live in
`analysis/goldens/data/fixtures/hunspell` (and inputs under
`tests/snapshots/inputs/` plus `inputs/stem` for published dictionaries).

The fixtures are written for this project and may be redistributed with it:

| Fixture | What it covers |
|---|---|
| `affixes` | Conditions, strip text, continuation classes, cross products, `NEEDAFFIX`, `ONLYINCOMPOUND`, `FORBIDDENWORD`, `KEEPCASE`, `CIRCUMFIX`, `WARN` with `FORBIDWARN`, `FULLSTRIP`, `IGNORE`, `ICONV`, `OCONV`, morphological fields |
| `flag-long`, `flag-num`, `flag-utf8`, `raw-flags` | The flag formats, with `AF` and `AM` aliases, supplementary characters and raw flag bytes in a UTF-8 file |
| `compound` | Positional compound flags, `COMPOUNDMIN`, `COMPOUNDWORDMAX`, the `CHECKCOMPOUND*` checks, `COMPOUNDPERMITFLAG`, `COMPOUNDFORBIDFLAG`, `FORCEUCASE`, `COMPOUNDROOT`, `CHECKCOMPOUNDPATTERN`, `CHECKCOMPOUNDREP` and `BREAK` |
| `triple`, `compound-rule`, `complex` | `SIMPLIFIEDTRIPLE`, `COMPOUNDRULE`, `COMPLEXPREFIXES` |
| `latin1`, `turkic`, `hungarian` | An ISO 8859-1 file with `CHECKSHARPS`, Turkic case mapping, Hungarian compounding (`COMPOUNDSYLLABLE`, `SYLLABLENUM`) |
| `bom-crlf`, `unsupported-directives` | A byte order mark, CRLF lines and comments; a directive the reader refuses |
| `error-*` | One load error each: the flag mode, a truncated block, a condition, the encoding, an alias, a morphology alias, a long flag, a compound rule, bytes that are not UTF-8 |



## Speed

`examples/hunspell_bench.rs` loads a dictionary and stems its language's
Snowball sample and the same words capitalized:

```
cargo run --release -p turbo-hunspell --example hunspell_bench -- en_US
```

On one core of the machine the goldens were made on, stemming each sample
(about 2,000 words, half of them capitalized) with `stems` ran at about
530,000 words a second with `en_US`, 17,000 with `de_DE_frami` and 6,400
with `hu_HU`. OpenNLP's `stemAll` over the same words, in the goldens'
container after a warm up, ran at about 290,000, 13,000 and 400. Most words
of the German and Hungarian samples that are not in the dictionary go
through the whole compound search, which is what those numbers measure.
