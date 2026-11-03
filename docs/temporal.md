# Temporal expressions

The `temporal` module of `turbo-numeric` (`analysis/numeric/src/temporal/`)
finds calendar mentions in a text view, writes them as the `turbo:temporal`
layer, and picks the date a document is dated with, the
`turbo:document-date` layer. Dates, times, durations and sets written
as words ("tomorrow at 8am", "every Monday") are also read by the
rule grammar in `turbo-grammar` (`analysis/grammar/`):

| Type | Role |
|---|---|
| `TemporalExtractor` | Extracts calendar mentions into `Temporals` (`turbo:temporal`) |
| `Temporals`, `TemporalMention`, `TemporalValue` | Typed layer, mention accessor, and resolved calendar value |
| `Granularity`, `Origin` | `Day` / `Week` / `Month` / `Quarter` / `Year` and `Absolute` / `Relative` |
| `DocumentDateSelector` | Selects the earliest absolute day mention as `DocumentDate` (`turbo:document-date`) |
| `DatedMoneyConverter` | Converts `Money` using `DocumentDate` ([money.md](money.md)) |

`Date`, the calendar date of [money.md](money.md), lives in
`turbo-analysis-layers` beside the layers and provides proleptic ISO
calendar arithmetic for relative mentions.

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
3. With a reference date, a count of up to nine digits, a unit and `ago`:
   `3 days ago`, `100 days ago`, `1 week ago`.

At a letter it tries:

1. `Q` and a digit: a quarter, `Q3 2024`. Nothing else is tried there.
2. A month first: `July 2026` (a month), or `July 14, 2026`,
   `Jul 14 2026`, `July 14th, 2026` (a day).
3. With a reference date, a relative keyword: `today`, `yesterday`,
   `tomorrow`; `last`, `this` or `next` and a unit (singular or plural);
   `in`, a count and a unit.

The parts of a mention are separated by exactly one `White_Space`
character. Months are their full English names or first three letters,
in any case. A word is a run of at most ten UTF-16 units of letters
followed by a word boundary. An absolute mention needs a four-digit year
from 0001 to 9999 and a date that exists. Units are `day`, `week`,
`month`, `quarter` and `year`; a unit may have one final `s`.

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

`DocumentDateSelector` reads `turbo:temporal` and writes `turbo:document-date`,
and `DocumentDateSelector::from_times()` (`TimesDocumentDateSelector`) reads
`turbo:times` (with optional priority to `turbo:temporal` when present) and
writes `turbo:document-date`:

- `DocumentDateSelector` selects the absolute day mention in `turbo:temporal`
  that starts first and records its `mention` index. Because it reads only
  `turbo:temporal`, pipelines combining `TemporalExtractor`,
  `DocumentDateSelector`, and `TimeAnnotator` always schedule
  `DocumentDateSelector` before `TimeAnnotator` regardless of the order in
  which steps are added.
- `DocumentDateSelector::from_times()` (`TimesDocumentDateSelector`) selects
  the earliest absolute day from `turbo:temporal` when present, or else the
  earliest resolved `Date` or `DateTime` mention from `turbo:times` (with
  `mention: None`), so Spanish, Dutch and other non-English documents
  annotated by `TimeAnnotator` can be dated and converted by
  `DatedMoneyConverter` ([money.md](money.md)).
- Relative mentions, months, quarters, durations and times of day never
  date a document.

## Named entities (`date` and `time`)

The `"date"` and `"time"` named entity types in `Entities` come from
deterministic rules in `turbo-numeric` and `turbo-grammar`, projected by
`turbo-namefind` (`analysis/namefind/src/rules/mod.rs`):

- `temporals_to_entities` and `RuleEntityProjector::temporals` project every
  mention in `Temporals` (`turbo:temporal`: `Day`, `Week`, `Month`, `Quarter`
  and `Year`, whether `Absolute` or `Relative`) to `"date"`, aligned to
  `Tokens`.
- `times_to_entities`, `times_to_entities_with`, `times_to_entities_in_text`
  and `RuleEntityProjector::times` project `Times` (`turbo:times`, written by
  `TimeAnnotator` in `analysis/grammar/src/times.rs`) according to
  `TimeEntityFilter`:
  - `TimeEntityFilter::DatesAndTimes` (default) maps `TimeKind::Date` and
    `TimeKind::DateRange` to `"date"`, `TimeKind::Time` and
    `TimeKind::TimeRange` to `"time"`, and `TimeKind::DateTime` and
    `TimeKind::DateTimeRange` to `"time"` (or `"date"` / skipped via
    `DateTimeEntityKind`), while skipping `TimeKind::Duration` and
    `TimeKind::Set`. When source text is available (`times_to_entities_in_text`
    and `RuleEntityProjector::times`), composite `<date> at <time>`
    `DateTime` / `DateTimeRange` spans (`2026-07-14 at 9:30am`) are split at
    `" at "` into a `"date"` prefix and a `"time"` suffix.
  - `TimeEntityFilter::TimesOnly` and `TimeEntityFilter::ClockTimesOnly`
    project only time-of-day mentions to `"time"` when `"date"` spans are
    already supplied by `Temporals`.
  - `TimeEntityFilter::DatesOnly` projects only `TimeKind::Date` and
    `TimeKind::DateRange` to `"date"`.
- `RuleEntityProjector` preserves the upstream layer's provenance
  (`"turbo-numeric:temporal"` or `"turbo-grammar:times"`) and records both the
  source layer and `Tokens` in `reads`.
- When a learned NER head (such as the MASC `date` head) and the deterministic
  rule both produce `"date"`, placing the rule layer ahead of the learned head
  in `ConsensusEntityMerger` (`analysis/namefind/src/consensus.rs`) under
  `EntityMergePolicy::PriorityFallback` keeps the rule's exact byte and token
  boundaries and attributes `Evidence<Entities>` to the rule producer, while
  absorbing the overlapping learned `"date"` prediction as a corroborating
  vote that supplies the merged entity's model confidence and raw score.
  Disjoint `"date"` spans predicted only by the learned head (such as named
  holidays not in `Temporals`) are retained as fallback entities.

## Edge cases and testing

Default extraction behavior:

- **Four-digit years (`0001..=9999`)**: any four-digit year in
  `0001..=9999` is accepted (`0999-12-31`, `3000-01-01`, `Q1 0999`; note
  that a month name followed by any four-digit numeral such as `March 4000`
  is read as a `Month` mention).
- **Multi-digit relative counts**: counts of 1 to 9 digits are accepted
  (`100 days ago`, `in 365 days`).
- **Plurals after `last`, `this` and `next`**: `next weeks`, `last months`
  and `this years` are accepted alongside the singular forms.
- **Supplementary-plane letters**: Unicode letter classification is tested
  on the full scalar value.
- The dateline is the first absolute day even when a relative mention
  comes before it: in `Yesterday we wrote this. Dateline 2025-11-30`,
  `Yesterday` is 2025-11-29.
- A month or a quarter is never a dateline and never a document date:
  `In July 2026 ... next week` has no relative mention.
- An impossible day leaves the month: `31 April 2026` gives `April 2026`,
  a month, and `30 February 2024` gives `February 2024`.
- `Q` and a digit are tried as a quarter only, so `Q3 July 2026` gives
  `July 2026` and `Q5 2024` gives nothing.
- The gap is exactly one `White_Space` character, a line break or an
  ideographic space included; two spaces end the mention.
- In a timestamp, hour 24 needs zero minutes, seconds and fraction, and
  second 60 is taken without a leap second table. A timestamp whose time
  of day is invalid gives no date at all.

`analysis/numeric/tests/temporal_extract.rs`,
`analysis/numeric/tests/temporal_fixes.rs`,
`analysis/numeric/tests/temporal_relative.rs`,
`analysis/numeric/tests/temporal_timestamp.rs`, and
`analysis/numeric/tests/temporal_annotate.rs` test absolute and relative
mentions, timestamps, dateline selection, `DocumentDateSelector`, and the
`temporal` snapshot suite in both `dateline` and `fixed` reference modes.
`analysis/namefind/tests/rule_entities.rs`,
`analysis/namefind/tests/rule_entities_eval.rs`, and
`analysis/query/tests/rule_entities_gold.rs` test `"date"` and `"time"`
projection into `Entities`, MASC `"date"` consensus merging, and exact-span
gold F1.
