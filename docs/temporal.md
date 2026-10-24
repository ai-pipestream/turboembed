# Temporal expressions

The `temporal` module of `turbo-numeric` (`analysis/numeric/src/temporal/`)
finds calendar mentions in a text view, writes them as the `turbo:temporal`
layer, and picks the date a document is dated with, the
`turbo:document-date` layer. It is a port of the temporal package of the
OpenNLP numeric add-on:

| Rust | OpenNLP |
|---|---|
| `TemporalExtractor` | `CursorTemporalExtractor` and `TemporalAnnotator` |
| `Temporals`, `TemporalMention`, `TemporalValue` | `TemporalExpression` |
| `Granularity`, `Origin` | `TemporalExpression.Granularity`, `TemporalExpression.Origin` |
| `DocumentDateSelector` | `DocumentDateAnnotator` |
| `DocumentDate` | the `LocalDate` annotation `DocumentDateAnnotator` adds |
| `DatedMoneyConverter` | `MoneyConversionAnnotator`, document-date mode ([money.md](money.md)) |

`Date`, the calendar date of [money.md](money.md), lives in
`turbo-analysis-layers` beside the layers, and has the `java.time`
arithmetic the relative mentions need.

## The layers

`Temporals`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The mention as written; for a timestamp, only its date |
| `granularity` | `Day`, `Week`, `Month`, `Quarter` or `Year` |
| `origin` | `Absolute` when the text names the period, `Relative` when it is resolved against a reference date |
| `year` | The year; for a week, the ISO week-based year |
| `period` | The month (1 to 12), ISO week (1 to 53) or quarter (1 to 4); 0 for a year |
| `day` | The day of the month for a day; 0 otherwise |

`Temporals::value` gives a `TemporalValue`, an enum with one variant per
granularity, whose `Display` writes what `TemporalExpression.value()`
holds: `2026-07-14`, `2026-W29`, `2026-07`, `2024-Q3` or `2026`. A year
outside 0 to 9999, which only a relative mention can reach, is written as
`java.time` writes it: at least four digits, with a sign below zero or
above 9999 (`-0004-02-29`, `+10000-01-01`).

`DocumentDate` is anchored on the document and holds no date or one:
`date`, and `mention`, the index of the `Temporals` item it came from.
Its `indexes` point into the default `Temporals` instance.

## Extraction

Offsets are UTF-8 byte offsets into the view, on character boundaries.
The scan tries a mention at each character that is an ASCII digit or a
letter and follows a word boundary; after a mention it goes on at the
mention's end. A digit that goes on with a number before it never starts
one, as in [numbers.md](numbers.md), so `1,14 July 2026` gives nothing.

At a digit it tries, in order:

1. An ISO date, `2026-07-14`, exactly four, two and two digits, followed
   by a word boundary or by a complete time of day, `T09:30`,
   `t09:30:00.125+05:30` or `T24:00Z`. The mention is then the date's ten
   bytes; an incomplete or invalid time of day makes the date no mention.
2. A day-first date: one or two digits, an optional ordinal suffix (`st`,
   `nd`, `rd`, `th`), the month and the year: `14 July 2026`,
   `14th Jul 2026`.
3. With a reference date, a count of up to nine digits (one or two digits
   in OpenNLP parity mode), a unit and `ago`: `3 days ago`, `100 days ago`,
   `1 week ago`.

At a letter it tries:

1. `Q` and a digit: a quarter, `Q3 2024`. Nothing else is tried there.
2. A month first: `July 2026` (a month), or `July 14, 2026`,
   `Jul 14 2026`, `July 14th, 2026` (a day).
3. With a reference date, a relative keyword: `today`, `yesterday`,
   `tomorrow`; `last`, `this` or `next` and a unit (singular or plural by
   default, singular only in OpenNLP parity mode); `in`, a count and a
   unit.

The parts of a mention are separated by exactly one `White_Space`
character. Months are their full English names or first three letters,
in any case. A word is a run of at most ten UTF-16 units of letters
followed by a word boundary. An absolute mention needs a four-digit year
from 0001 to 9999 (`1000..=2999` in OpenNLP parity mode) and a date that
exists. Units are `day`, `week`, `month`, `quarter` and `year`; a unit may
have one final `s`.

A relative mention moves the reference by the count of the unit (a
quarter is three months) with the `java.time` rules: a month or year step
that lands past the end of a month falls back to its last day. The
mention then has the unit's granularity: `next week` is the ISO week of
the reference plus seven days, `next quarter` the quarter three months on.
A result outside the `LocalDate` range (years -999999999 to 999999999) is
no mention.

The reference date is chosen when the extractor is made:

- `TemporalExtractor::new()` takes it from the text: the first absolute
  day mention, its dateline. A text without one has no relative mentions;
  the clock is never read.
- `TemporalExtractor::with_reference(date)` fixes it, for a document
  whose date is known from metadata. A dateline in the text then does not
  count.

`TemporalExtractor::extract` and the free functions `scan` and `dateline`
do the same outside a pipeline.

## Document date

`DocumentDateSelector` reads `turbo:temporal` and `turbo:times` (from
`turbo-grammar`, when present) and writes `turbo:document-date`:

- When `turbo:temporal` holds an absolute day mention, it selects the one
  that starts first and records its `mention` index.
- Otherwise, when `turbo:times` holds a resolved `Date` or `DateTime`
  mention, it selects the earliest one's calendar date (with `mention:
  None`), so Spanish, Dutch and other non-English documents annotated by
  `TimeAnnotator` can still be dated and converted by
  `DatedMoneyConverter` ([money.md](money.md)).
- Relative mentions, months, quarters, durations and times of day never
  date a document.

## Differences from OpenNLP

By default `TemporalExtractor` fixes four OpenNLP cursor quirks while
keeping OpenNLP parity behind `TemporalExtractor::with_opennlp_parity(true)`
(and `scan_with_parity` / `dateline_with_parity`):

- **Four-digit years (`0001..=9999`)**: by default any four-digit year in
  `0001..=9999` is accepted (`0999-12-31`, `3000-01-01`, `Q1 0999`); parity
  mode restricts absolute mentions to `1000..=2999`.
- **Multi-digit relative counts**: by default counts of 1 to 9 digits are
  accepted (`100 days ago`, `in 365 days`); parity mode accepts 1 or 2
  digits only.
- **Plurals after `last`, `this` and `next`**: by default `next weeks`,
  `last months` and `this years` are accepted alongside the singular forms;
  parity mode requires the singular after `last`, `this` and `next`.
- **Supplementary-plane letters**: by default `java::is_letter` is tested on
  the full scalar value rather than a single UTF-16 code unit; parity mode
  excludes supplementary characters from entering the initial word branch.

With `.with_opennlp_parity(true)`, none in the results differ from OpenNLP.
The tests port the cases of the add-on's temporal tests (the `CursorTemporal*`,
`TemporalAnnotator*`, `DocumentDateAnnotator*` and
`TemporalExtractorContractTest` classes, and the dated cases of the money
conversion tests), with relative shifts across month ends,
leap years and the ends of the `LocalDate` range checked against values
the JVM computed. A JVM oracle (outside the repository) compared the port
with the Java on 2,250,000 generated texts in three modes (no reference,
a random reference from the whole `LocalDate` range with its ends and
year boundaries weighted, and the dateline with the document date), about
950,000 mentions, with no difference.

Where the APIs differ:

- A value is a typed `TemporalValue`, not a string with a separate
  granularity. The checks the Java makes on a `TemporalExpression` (a
  blank value, a value that does not parse for its granularity) cannot
  fail, and the remaining ones (column lengths, spans in order, ranges of
  the parts) are `Temporals::check`.
- The extractor is not pluggable: `TemporalAnnotator` takes any
  `TemporalExtractor` implementation, the port has the cursor extractor
  only.
- `DocumentDateAnnotator` checks that each mention's span is its
  annotation's and that the chosen value parses as a date; a `Temporals`
  item has one span and a typed date, so neither can fail.
- The character classes come from the Unicode 18.0 tables of
  `turbo-unicode`, as in numbers.md.

## OpenNLP behaviour kept

Behaviour that comes from the Java and is kept, though it looks wrong:

- The dateline is the first absolute day even when a relative mention
  comes before it: in `Yesterday we wrote this. Dateline 2025-11-30`,
  `Yesterday` is 2025-11-29.
- A month or a quarter is never a dateline and never a document date:
  `In July 2026 ... next week` has no relative mention.
- An impossible day leaves the month: `31 April 2026` gives `April 2026`,
  a month, and `30 February 2024` gives `February 2024`.
- `Q` and a digit are tried as a quarter only, so `Q3 July 2026` gives
  `July 2026` and `Q5 2024` gives nothing.
- Words are lowercased code point by code point with
  `Character.toLowerCase`, so U+0130 (`İ`) reads as `i` and the Kelvin
  sign U+212A as `k`: `İn 3 days`, and `next wee` followed by the Kelvin
  sign, are mentions. Those are
  the only two characters outside ASCII that lowercase into it.
- A letter outside the Basic Multilingual Plane never starts a mention:
  the Java tests one UTF-16 unit, half of a surrogate pair, with
  `Character.isLetter`.
- After `last`, `this` and `next` only the singular is read
  (`next weeks` gives nothing), while a counted unit drops one final `s`
  whatever the count: `in 1 weeks` and `in 2 day` both read, `in 2 dayss`
  does not.
- A count has one or two digits: `100 days ago` and `in 100 days` give
  nothing.
- The gap is exactly one `White_Space` character, a line break or an
  ideographic space included; two spaces end the mention.
- In a timestamp, hour 24 needs zero minutes, seconds and fraction, and
  second 60 is taken without a leap second table. A timestamp whose time
  of day is invalid gives no date at all.
- The year range 1000 to 2999 holds for absolute mentions only; relative
  mentions reach the whole `LocalDate` range, and a step past it is
  dropped without an error.

## Golden files

JVM goldens for temporal mentions come from the goldens tool
(`analysis/goldens-tool`), as for the other OpenNLP ports
([analysis.md](analysis.md#golden-files)).

- Inputs: the `temporal` set, `analysis/goldens/data/inputs/temporal/`,
  65 texts.
- Goldens: `analysis/goldens/data/temporal/<mode>/temporal/<id>.golden`,
  `dateline` or `fixed`, from `TemporalAnnotator` over
  `CursorTemporalExtractor` followed by `DocumentDateAnnotator`. Layers
  `turbo:temporal` and `turbo:document-date`.

`turbo:temporal` has the columns `start`, `end` (UTF-8 byte offsets),
`value` (`TemporalExpression.value()`), `granularity` and `origin` (the
enum constants' names). `turbo:document-date` has `date` and `mention`,
the index of the mention whose span the date annotation has. The `config`
record is `reference`: `dateline`, or `2026-03-15` for the fixed mode.
`analysis/numeric/tests/temporal_goldens.rs` runs `TemporalExtractor` and
`DocumentDateSelector` over the set in both modes and compares both
layers with the goldens row for row.
