## Golden files

JVM goldens for money come from the goldens tool (`analysis/goldens-tool`),
as for the other OpenNLP ports ([analysis.md](analysis.md#golden-files)).

- Inputs: the `money` set, `analysis/goldens/data/inputs/money/`, 64 texts.
- Rates: `analysis/goldens/data/rates/ecb-snapshot.csv`, a small hand made
  file in the format of the ECB's `eurofxref-hist.csv` with invented
  values (columns USD, JPY, GBP, CHF, CNY, INR, RUB; 12 weekdays from
  2026-03-02 to 2026-03-17; `N/A` where there is no quote).
- Extraction: `analysis/goldens/data/money/<notation>/money/<id>.golden`,
  `latin-us` or `latin-eu`, from `MoneyAnnotator` over
  `CursorMoneyExtractor`. Layer `turbo:money`.
- Conversion: `analysis/goldens/data/money-convert/<notation>/money/<id>.golden`,
  from `MoneyConversionAnnotator` over `EcbFxRates` into EUR as of
  2026-03-15 (a Sunday, so the Friday 2026-03-13 row applies). Layer
  `turbo:money-converted`; a mention without a rate has no row.

Both layers have the columns `start`, `end` (UTF-8 byte offsets of the
whole mention, sign and symbol included), `coefficient` (unscaled value
without sign), `negative` (0 or 1), `scale`, `plain` (`toPlainString()`)
and `currency` (ISO 4217). The value is
`(-1)^negative * coefficient / 10^scale`. The `config` records are
`notation` (`LATIN_US` or `LATIN_EU`) and, for conversions, `target`,
`as_of` and `snapshot` (`rates/ecb-snapshot.csv`, relative to
`analysis/goldens/data`).
