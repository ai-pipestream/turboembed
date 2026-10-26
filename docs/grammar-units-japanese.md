# Japanese units and money

`JAPANESE_UNITS` reads Japanese ages, dimensions, temperatures and currencies.
Use `GrammarAnnotator::new(&JAPANESE_UNITS).bound()` to write the existing
`grammar` Money and Quantities instances. Its provenance name is `ja-units`.
It does not enable the separate Number, Ordinal or Percent output families.

The grammar retains source byte ranges. Its lexicon splitter takes the longest
known unit term before considering numeric prefixes; unmatched characters stay
in the input. Shared CJK arithmetic handles Arabic and fullwidth digits,
ideographic digits and scales, Japanese spoken number atoms, negative words,
decimal points and slash fractions with `Dec`, without converting through f64.
The largest pinned Chinese amount, 202107211321363588510367, is checked exactly
by the shared test. Source strings are borrowed rather than reconstructed.

A grammar-specific boundary hook permits Han/kana quantities inside unspaced
sentences. Existing grammars retain the original written-token boundary rule.
Latin identifiers still prevent partial quantities. The shared CJK reader does
not concatenate a Japanese particle such as `に` with an Arabic count.

Supported combinations include yen and sen, yuan/jiao/fen, dollar/euro cents,
repeated currency labels, spelled numbers, `時速` and `毎秒` speed prefixes,
`華氏`/`摂氏` temperature prefixes and `マイナス` temperatures. Bare temperature
and unambiguous currency names remain parser results; only quantities with a
value become layer records. Standalone `角`/`分` are not emitted as money, since
these also occur in ordinary geometric and temporal text. `日` and `週` age
readings require birth context. An age range keeps its final positive endpoint.

`キロ` defaults to kilometers; explicit weight/sales context (`体重`, `重量`,
`出荷`, `完売`) selects kilograms. This lexical policy cannot resolve every
meaning in arbitrary prose. `ムー` denotes the modern Chinese land-area unit,
2000/3 square meters, in the Chinese agricultural passage in the upstream
Japanese corpus; it is not a Japanese traditional land-area unit. The shared
unit conventions are documented in [Chinese units](grammar-units-chinese.md).

## Pinned upstream comparison

The unchanged MIT-licensed Recognizers-Text files are pinned at
`da7edcff59f669b2a460ab9d400e36298f0d658e` under
`analysis/grammar/tests/data/recognizers-text/Japanese/NumberWithUnit/`.
All 188 entries run, including those marked unsupported by upstream runtimes.
No case is removed or skipped. Japanese Dimension fixtures omit subtype names;
the comparator adds the corresponding unit subtype without changing the unit,
value, or source span.

| File | Cases | Exact upstream matches | Explicit native corrections |
| --- | ---: | ---: | ---: |
| AgeModel | 18 | 17 | 1 |
| CurrencyModel | 68 | 68 | 0 |
| DimensionModel | 64 | 63 | 1 |
| TemperatureModel | 38 | 38 | 0 |
| Total | 188 | 186 | 2 |

The two differences are checked separately and are not parity passes:

- AgeModel case 14 expects `三歳` to mean three weeks. The native result is
  three years, consistent with `歳` and the other age cases.
- DimensionModel case 46 expects `100メガバイト` to mean megabits. The native
  result is 100 megabytes. `ビット` remains bits.

`japanese_unit_specs` checks exact UTF-16-derived source spans, values, unit
names and currency codes. `japanese_units` exercises the real Analyzer,
round-trips every emitted money/quantity span across a markup-removal view for
all 188 inputs, and checks UTF-8/UTF-16/code-point endpoints. A separate
fullwidth-digit conversion checks source-to-child and child-to-source ranges
when byte widths change. Native edge cases cover negative ages, range signs,
foreign currency subunits, ambiguous minor-unit words and code escapes.

The pinned Japanese Number (764), Ordinal (101) and Percent (204) files were
also inspected when choosing the numeric helper. This unit change does not
claim conformance to those three output families. No model or training run is part of this change. A bounded existing-English
comparison and its shared-host limits are recorded in
[the receipt](receipts/grammar-929-japanese/README.md).

Bare day/week ages require a birth cue in the same sentence or clause. A cue
before a sentence mark, comma, semicolon or line separator does not turn a
later duration into an age. Leading periods on bare days/weeks are ambiguous
with sentence punctuation and do not inherit the birth cue; explicit age-unit
spellings remain available. Unicode minus signs retain negative money values
and are refused for ages, including positive suffix candidates.
