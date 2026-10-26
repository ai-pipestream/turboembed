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

## Chinese NumberWithUnit

The four `Chinese/NumberWithUnit/*Model.json` files are unchanged copies from
revision `da7edcff59f669b2a460ab9d400e36298f0d658e` of
[Microsoft Recognizers-Text](https://github.com/microsoft/Recognizers-Text/tree/da7edcff59f669b2a460ab9d400e36298f0d658e/Specs/NumberWithUnit/Chinese),
under the existing vendored MIT license. `chinese_unit_specs.rs` checks all
91 cases, including upstream unsupported cases, with original offsets and
all supplied resolution fields. Optional absent subtype fields are unspecified.
See `docs/grammar-units-chinese.md` for exact unit conventions and native tests.
