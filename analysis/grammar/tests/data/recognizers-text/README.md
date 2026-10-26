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
| `Dutch/NumberWithUnit/AgeModel.json` | `Specs/NumberWithUnit/Dutch/AgeModel.json` |
| `Dutch/NumberWithUnit/CurrencyModel.json` | `Specs/NumberWithUnit/Dutch/CurrencyModel.json` |
| `Dutch/NumberWithUnit/DimensionModel.json` | `Specs/NumberWithUnit/Dutch/DimensionModel.json` |
| `Dutch/NumberWithUnit/TemperatureModel.json` | `Specs/NumberWithUnit/Dutch/TemperatureModel.json` |
| `Dutch/Number/NumberModel.json` | `Specs/Number/Dutch/NumberModel.json` |
| `Dutch/Number/OrdinalModel.json` | `Specs/Number/Dutch/OrdinalModel.json` |
| `Dutch/Number/PercentModel.json` | `Specs/Number/Dutch/PercentModel.json` |
| `Dutch/DateTime/DateTimeModel.json` | `Specs/DateTime/Dutch/DateTimeModel.json` |
| `Spanish/Number/NumberModel.json` | `Specs/Number/Spanish/NumberModel.json` |
| `Spanish/Number/OrdinalModel.json` | `Specs/Number/Spanish/OrdinalModel.json` |
| `Spanish/Number/PercentModel.json` | `Specs/Number/Spanish/PercentModel.json` |
| `Spanish/DateTime/DateTimeModel.json` | `Specs/DateTime/Spanish/DateTimeModel.json` |

`tests/spec.rs` reads them; docs/grammar.md says how a case is judged.

`tests/dutch_unit_specs.rs` checks the Dutch NumberWithUnit files. It reports
unchanged-oracle agreement and the explicit Dutch policy checks separately;
see `docs/grammar-units.md` for every difference. Where upstream omits numeric
coordinates, the harness derives them from the expected text, independently
of parser results. One Currency case is marked unsupported by upstream.
