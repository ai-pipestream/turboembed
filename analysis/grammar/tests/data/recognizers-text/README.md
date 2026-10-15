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

`tests/spec.rs` reads them; docs/grammar.md says how a case is judged.
