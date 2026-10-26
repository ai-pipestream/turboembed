# Recognizers-Text specifications

The `*Model.json` files under this directory are copied unchanged from
the `Specs/` directory of the microsoft/Recognizers-Text repository on
GitHub, at commit `da7edcff59f669b2a460ab9d400e36298f0d658e`. They are
used under the MIT License in [LICENSE](LICENSE), copyright Microsoft
Corporation.

| File here | Upstream path |
|---|---|
| `English/Number/NumberModel.json` | `Specs/Number/English/NumberModel.json` |
| `English/Number/OrdinalModel.json` | `Specs/Number/English/OrdinalModel.json` |
| `English/Number/PercentModel.json` | `Specs/Number/English/PercentModel.json` |
| `English/DateTime/DateTimeModel.json` | `Specs/DateTime/English/DateTimeModel.json` |
| `English/NumberWithUnit/AgeModel.json` | `Specs/NumberWithUnit/English/AgeModel.json` |
| `English/NumberWithUnit/CurrencyModel.json` | `Specs/NumberWithUnit/English/CurrencyModel.json` |
| `English/NumberWithUnit/DimensionModel.json` | `Specs/NumberWithUnit/English/DimensionModel.json` |
| `English/NumberWithUnit/TemperatureModel.json` | `Specs/NumberWithUnit/English/TemperatureModel.json` |
| `Dutch/Number/NumberModel.json` | `Specs/Number/Dutch/NumberModel.json` |
| `Dutch/Number/OrdinalModel.json` | `Specs/Number/Dutch/OrdinalModel.json` |
| `Dutch/Number/PercentModel.json` | `Specs/Number/Dutch/PercentModel.json` |
| `Dutch/DateTime/DateTimeModel.json` | `Specs/DateTime/Dutch/DateTimeModel.json` |
| `Spanish/Number/NumberModel.json` | `Specs/Number/Spanish/NumberModel.json` |
| `Spanish/Number/OrdinalModel.json` | `Specs/Number/Spanish/OrdinalModel.json` |
| `Spanish/Number/PercentModel.json` | `Specs/Number/Spanish/PercentModel.json` |
| `Spanish/DateTime/DateTimeModel.json` | `Specs/DateTime/Spanish/DateTimeModel.json` |

`tests/spec.rs` reads them; docs/grammar.md says how a case is judged.

The Spanish NumberWithUnit files below are also unchanged copies at the same
pinned revision and MIT license. `tests/spanish_unit_specs.rs` checks every case.

| File here | Upstream path |
|---|---|
| `Spanish/NumberWithUnit/AgeModel.json` | `Specs/NumberWithUnit/Spanish/AgeModel.json` |
| `Spanish/NumberWithUnit/CurrencyModel.json` | `Specs/NumberWithUnit/Spanish/CurrencyModel.json` |
| `Spanish/NumberWithUnit/DimensionModel.json` | `Specs/NumberWithUnit/Spanish/DimensionModel.json` |
| `Spanish/NumberWithUnit/TemperatureModel.json` | `Specs/NumberWithUnit/Spanish/TemperatureModel.json` |

The Japanese NumberWithUnit files are unchanged copies at the same revision
and MIT license. All 188 cases are run, including upstream unsupported entries.
The two incorrect upstream unit labels are explicitly tested as native policy
differences in `japanese_unit_specs.rs`, not counted as parity matches.

| File here | Upstream path | SHA256 |
|---|---|---|
| `Japanese/NumberWithUnit/AgeModel.json` | `Specs/NumberWithUnit/Japanese/AgeModel.json` | `130278bcce4feca64b3a66e62cd1eb8d6581661584f3347234684a67dff55a00` |
| `Japanese/NumberWithUnit/CurrencyModel.json` | `Specs/NumberWithUnit/Japanese/CurrencyModel.json` | `bb47bab4033b335dffc8f5f17ddc9c74b8eb1bf5718625a0a568c0350c18d679` |
| `Japanese/NumberWithUnit/DimensionModel.json` | `Specs/NumberWithUnit/Japanese/DimensionModel.json` | `c1122b55305825f172ba949a5b18026570fe2f14c23796a9a7f8e244d7719820` |
| `Japanese/NumberWithUnit/TemperatureModel.json` | `Specs/NumberWithUnit/Japanese/TemperatureModel.json` | `8d88e5679f477f667f21ba16771518c3837ca76b14e1b966afd21cb9ea6f642a` |
