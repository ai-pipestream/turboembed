# Numbers

`NumberScanner` in `turbo-numeric` (`analysis/numeric/`) finds the
numbers written in a text view and writes them as the `turbo:numbers`
layer. Spelled-out numbers with exact values ("one hundred and
three") are read by the rule grammar in `turbo-grammar`
(`analysis/grammar/`), not by this scanner.

## The layer

`Numbers`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The whole mention: the minus sign, the digits and the scale marker |
| `numeral_end` | Where the digits, group separators and fraction end, before a scale marker; the span's end when there is none. The numeral starts at the span's start, or one byte later when the mention is negative |
| `negative` | Whether the mention starts with a minus |
| `coefficient` | The value's digits without sign, decimal point or leading zeros, `0` for zero |
| `scale` | How many of those digits are the fraction |
| `multiplier` | `One`, `Thousand`, `Million`, `Billion` or `Trillion` |
| `marker` | `None`, `Suffix` (`k`, `m`, `b`, `bn`) or `Word` (`thousand` to `trillion`) |
| `notation` | The notation the mention was read in |

`coefficient` and `scale` store the unscaled integer value and decimal
scale with the multiplier applied: `1.5k` has the coefficient `15000`
and the scale 1. `Numbers::value` returns a `DecimalRef`, which formats
the value as a plain decimal string (`-1500.0`, `0.05`) and converts it
to the nearest `f64`. A negated zero is zero, though `negative` still
records the sign in the text.

## The scan rules

Offsets are UTF-8 byte offsets into the view, on character boundaries.

- A mention starts at a decimal digit (`Nd` in Unicode, such as ASCII
  `0..=9`, Arabic-Indic `U+0660..=U+0669`, Extended Arabic-Indic
  `U+06F0..=U+06F9`, Devanagari `U+0966..=U+096F`, or Fullwidth
  `U+FF10..=U+FF19`), or at a minus directly before one. All digits in a
  single numeral must belong to the same 10-digit Unicode block, and the
  value is normalized into canonical ASCII digits in `coefficient`.
  The character before the mention, skipping combining marks, must not be
  a letter, a number of any kind, a mark or a connector such as `_`. A
  minus also must not follow a currency symbol or `%`, so `5%-10%` is two
  positive numbers.
- A digit right after a group or decimal separator that follows a digit,
  or after an exponent marker (`1e-3`), never starts a mention: it goes
  on with the number before it.
- `NumberNotation::LatinUs` groups with `,` and marks the fraction with
  `.`; `LatinEu` the other way round. Callers can choose a fixed notation,
  resolve one from a `RegionCode` (`region_notation` /
  `NumberScanner::for_region`) or a BCP-47 locale tag (`locale_notation` /
  `NumberScanner::for_locale`), or run `RegionAwareNumberAnnotator`
  (`turbo-numeric:region-aware-numbers`) over `(View, Regions)`. When a
  document's `Regions` ballot carries votes for both `LatinEu` and
  `LatinUs` countries, `RegionAwareNumberAnnotator` tries the winning
  region's notation first and falls back to the secondary notation when a
  candidate numeral is written in the other convention (`1.234,5`
  alongside `1,234.5`).
- Grouping conventions supported by default:
  - Standard 3-digit grouping in the active notation (`1,234,567.89` in
    `LatinUs`, `1.234.567,89` in `LatinEu`).
  - Indian lakh and crore comma grouping in `LatinUs` (`1,00,000`,
    `12,34,567.89`, `1,23,45,678`, with 1 or 2 leading digits, one or
    more 2-digit comma groups, and a final 3-digit comma group).
  - Thin and no-break space 3-digit grouping (`U+00A0`, `U+2009`,
    `U+202F`, such as `1\u{00A0}000`, `1\u{2009}234\u{2009}567`,
    `1\u{202F}350,75`).
  - Swiss apostrophe 3-digit grouping (`'` and `U+2019`, such as `1'000`,
    `1\u{2019}234.56`, `1'234,56`).
  - Arabic thousands (`U+066C`) and decimal (`U+066B`) separators
    (`1\u{066C}234\u{066B}56` and `\u{0661}\u{066C}\u{0662}\u{0663}\u{0664}\u{066B}\u{0665}\u{0666}`).
- Scientific notation exponents (`1e3`, `1E3`, `1e+3`, `1e-3`, `1E-03`,
  `2.5e-3`, `2,500e+3`, and `2,5e-3` in `LatinEu`) adjust `coefficient`
  and `scale` directly. Malformed exponents (`1.e-3`, `1,e3`, `1e-3e-4`)
  are rejected whole.
- The scan fails, rather than returning the prefix it read, when it stops
  at a separator followed by a digit that does not form a valid group or
  fraction (`1.2.3`, `1,23`, `1'23`, `1\u{2009}3`, `1.234,5` read in
  `LatinUs`), or when a separator precedes an exponent (`1.e-3`).
- With scaling on, the default, a `k`, `m`, `b` or `bn` in either case
  directly after the digits multiplies the value and must be followed by
  a word boundary; any other letter there rejects the number (`5x`). One
  `White_Space` character and then `thousand`, `million`, `billion` or
  `trillion` in any case also multiplies it, when the word ends at a
  boundary; otherwise the number ends before the whitespace.
- After a mention the scan goes on at its end; after a failed candidate,
  at the next character.

## Performance and boundary edge cases

By default, `NumberScanner`, `QuantityExtractor`, `MoneyExtractor`, and
`parse` accept the extended groupings, exponents, and non-ASCII `Nd`
digit scripts listed above. A strict ASCII-only `LatinUs`/`LatinEu` mode
is also available for ASCII-only fixture suites.

Candidate scanning uses a bounded AVX2 byte prefilter when available, with
non-ASCII candidates confirmed by the existing Unicode digit classifier.
Other CPUs keep the scalar candidate path. The default scanner avoids
regional fallback dispatch when none was requested and keeps its candidate
matcher compact. At text start or after an ASCII space, the word/sign
boundary is already known; all other positions use the general boundary
rules.

The [paired release receipt](receipts/numeric-931/README.md) retains the
original regression, a rejected intermediate implementation, and all final
samples. Code commit `14c0f541c2c55086139dc262195ea89ed338c8ab` was compared
with main `6f7a6ea6a16d1fc36972d57a883c67255146887e`, using 24 alternating
pairs per corpus and 1,000 timed iterations per sample on one CPU of a
shared AMD Ryzen 9 9950X3D host:

- Existing Latin output: 6,000 mentions per iteration on both; median
  209.18 ms on main and 200.90 ms on the candidate.
- Existing non-Latin text with ASCII numerals: 15,000 mentions per
  iteration on both; median 933.63 ms on main and 925.12 ms on the candidate.
- Non-Latin text with Nd numerals: 7,500 mentions on main versus 15,000 on
  the candidate; median 636.94 ms versus 1,200.01 ms. This performs new work
  and is not a same-output speed comparison.

The two existing-output corpora have matching value/span checksums,
including numeral_start. No
median slowdown was observed in this run; the small differences are not
claimed as statistically significant speedups. Raw ordering, ambient load,
input hashes, binary hashes, and reproduction steps are in the receipt.
These measurements do not establish performance on other architectures.

Boundary edge-case rules:

- A number with no scale marker is not checked for a boundary after it:
  `5` followed by U+0301 gives `5`, and without scaling `5x` gives `5`.
- Scale words are lowercased code point by code point, so `mİllion`
  (with U+0130) reads as `million`.

## Named entities (`percent`, `money`, `date`, `time`)

Named entity types `date`, `time`, `money` and `percent` (`percentage` in
legacy 7-type NER; `PERCENT` in OntoNotes) are grammar problems backed by the
deterministic numeric, money and temporal extractors rather than statistical
sequence NER models:

- `quantities_to_entities` and `RuleEntityProjector::quantities` in
  `turbo-namefind` (`analysis/namefind/src/rules/mod.rs`) project percentage
  mentions (`quantities.is_percent(ix)`, where `unit == "%"`) from
  `Quantities` (`turbo:quantities`, written by `QuantityExtractor` or
  `GrammarAnnotator` with `GRAMMAR_PERCENTS`; see
  [quantities.md](quantities.md) and [grammar.md](grammar.md)) into `Entities`
  spans typed `"percent"`, aligned to `Tokens`. Non-percentage unit quantities
  (`80 kg`, `2.5 km`) are left out.
- `RuleEntityProjector::money`, `RuleEntityProjector::temporals` and
  `RuleEntityProjector::times` project `Money` (`turbo:money`), `Temporals`
  (`turbo:temporal`) and `Times` (`turbo:times`) into `"money"`, `"date"` and
  `"time"` spans ([money.md](money.md), [temporal.md](temporal.md)), and
  `RuleEntitiesAnnotator` (`turbo-namefind:rule-entities`) projects all
  present rule layers into one `Entities` instance.
- Each `RuleEntityProjector` preserves its upstream layer's provenance
  (`annotator`, `version`, `config`, `model` and `execution`) while recording
  both the source layer and `Tokens` in `reads`, so `ConsensusEntityMerger`
  (`analysis/namefind/src/consensus.rs`) combines rule-backed entities with
  learned NER heads and attributes winning rule spans to their original
  deterministic producer in `Evidence<Entities>`.

## Testing

`analysis/numeric/tests/scanner.rs` and `analysis/numeric/tests/extended_numbers.rs`
test `NumberScanner` across `LatinUs` and `LatinEu` notations, Indian,
thin-space, apostrophe, and Arabic-Indic groupings, scientific notation,
and region-aware fallback, alongside the `num` snapshot suite in both notations.
`analysis/namefind/tests/rule_entities.rs`,
`analysis/namefind/tests/rule_entities_eval.rs` and
`analysis/query/tests/rule_entities_gold.rs` test rule-to-`Entities`
projection, provenance preservation, consensus merging with learned NER
heads, and exact-span F1 against reviewed gold spans.
