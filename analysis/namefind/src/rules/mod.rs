//! Deterministic projection of structured numeric, money, and temporal layers
//! into [`Entities`] (`turbo:entities`).
//!
//! Named entity types `"date"`, `"time"`, `"money"`, and `"percent"` are
//! grammar problems backed by the deterministic extractors in `turbo-numeric`
//! and `turbo-grammar`:
//!
//! - [`Temporals`] (`turbo:temporal`) maps calendar mentions (`Day`, `Week`,
//!   `Month`, `Quarter`, `Year`, both absolute and relative) to `"date"`.
//! - [`Times`] (`turbo:times`) maps [`TimeKind::Date`] and
//!   [`TimeKind::DateRange`] mentions to `"date"`, and [`TimeKind::Time`],
//!   [`TimeKind::TimeRange`], [`TimeKind::DateTime`], and
//!   [`TimeKind::DateTimeRange`] mentions to `"time"` (configurable via
//!   [`TimeEntityFilter`] and [`DateTimeEntityKind`]).
//! - [`Money`] (`turbo:money`) maps currency mentions (`$5.50`, `EUR 100`,
//!   `1.2 million dollars`) to `"money"`.
//! - [`Quantities`] (`turbo:quantities`) maps percentage mentions
//!   (`quantities.is_percent(ix)`, where `unit == "%"`) to `"percent"`,
//!   skipping non-percentage unit quantities (`80 kg`, `2.5 km`).
//!
//! Each projected [`Entities`] span preserves the source mention's exact byte
//! [`Span`], aligns its [`IxRange<Tokens>`] to the overlapping tokens of
//! [`Tokens`] (`turbo:tokens`) via binary search, and carries score `0.0`
//! (as dictionary and pattern finders do when no statistical model ran).
//!
//! [`RuleEntityProjector`] runs as a pipeline [`Step`](turbo_analysis::annotate::Step)
//! that preserves the upstream rule layer's provenance (`annotator`, `version`,
//! `config`, `model`, `execution`) while recording both the source layer and
//! [`Tokens`] in `reads`, so [`ConsensusEntityMerger`](crate::ConsensusEntityMerger)
//! attributes each rule entity to its original producer in
//! [`Evidence<Entities>`](turbo_analysis::Evidence).

mod step;

use std::cmp::Reverse;
use std::sync::Arc;

use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, Tag, TagSet};
use turbo_analysis_layers::{Entities, Money, Quantities, Temporals, TimeKind, Times, Tokens};

pub use step::{RuleEntitiesAnnotator, RuleEntityProjector, RuleEntitySource};

/// Canonical entity label for calendar date mentions (`"date"`).
pub const DATE_ENTITY: &str = "date";

/// Canonical entity label for clock and time-of-day mentions (`"time"`).
pub const TIME_ENTITY: &str = "time";

/// Canonical entity label for monetary mentions (`"money"`).
pub const MONEY_ENTITY: &str = "money";

/// Canonical entity label for percentage mentions (`"percent"`).
pub const PERCENT_ENTITY: &str = "percent";

/// All four deterministic rule-backed entity labels in canonical tag order.
pub const RULE_ENTITY_KINDS: [&str; 4] = [DATE_ENTITY, TIME_ENTITY, MONEY_ENTITY, PERCENT_ENTITY];

const NER_TAGSET: &str = "ner";
const RULE_SCORE: f64 = 0.0;

/// How [`TimeKind::DateTime`] and [`TimeKind::DateTimeRange`] mentions in
/// [`Times`] (`turbo:times`) map to [`Entities`] kinds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum DateTimeEntityKind {
    /// Map date-time points and ranges to [`TIME_ENTITY`] (`"time"`).
    ///
    /// In [`Times`], pure calendar expressions without a time-of-day component
    /// are [`TimeKind::Date`] or [`TimeKind::DateRange`]; [`TimeKind::DateTime`]
    /// and [`TimeKind::DateTimeRange`] always include a clock time, part-of-day
    /// (`this morning`, `tonight`), `now`, or sub-day clock unit.
    #[default]
    Time,
    /// Map date-time points and ranges to [`DATE_ENTITY`] (`"date"`).
    Date,
    /// Omit [`TimeKind::DateTime`] and [`TimeKind::DateTimeRange`] mentions.
    Skip,
}

impl DateTimeEntityKind {
    /// Deterministic configuration code for provenance hashing.
    pub const fn code(self) -> u64 {
        match self {
            DateTimeEntityKind::Time => 1,
            DateTimeEntityKind::Date => 2,
            DateTimeEntityKind::Skip => 3,
        }
    }
}

/// Selects which [`TimeKind`] variants from [`Times`] (`turbo:times`) are
/// projected into [`Entities`] (`turbo:entities`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimeEntityFilter {
    /// Project [`TimeKind::Date`] and [`TimeKind::DateRange`] to `"date"`,
    /// [`TimeKind::Time`] and [`TimeKind::TimeRange`] to `"time"`, and
    /// [`TimeKind::DateTime`] / [`TimeKind::DateTimeRange`] to `"time"`
    /// ([`DateTimeEntityKind::Time`]). [`TimeKind::Duration`] and
    /// [`TimeKind::Set`] are skipped.
    #[default]
    DatesAndTimes,
    /// Project dates and times with an explicit [`DateTimeEntityKind`] mapping
    /// for [`TimeKind::DateTime`] and [`TimeKind::DateTimeRange`].
    DatesAndTimesWith(DateTimeEntityKind),
    /// Project only time-of-day and date-time mentions ([`TimeKind::Time`],
    /// [`TimeKind::TimeRange`], [`TimeKind::DateTime`], and
    /// [`TimeKind::DateTimeRange`]) to `"time"`, leaving calendar dates to
    /// [`Temporals`].
    TimesOnly,
    /// Project strictly pure time-of-day mentions ([`TimeKind::is_time_of_day`]:
    /// [`TimeKind::Time`] and [`TimeKind::TimeRange`]) to `"time"`.
    ClockTimesOnly,
    /// Project only calendar date mentions ([`TimeKind::Date`] and
    /// [`TimeKind::DateRange`]) to `"date"`.
    DatesOnly,
}

impl TimeEntityFilter {
    /// Returns the target entity kind (`"date"` or `"time"`) for `kind`, or
    /// `None` when `kind` is excluded by this filter.
    pub const fn entity_kind(self, kind: TimeKind) -> Option<&'static str> {
        match self {
            TimeEntityFilter::DatesAndTimes => Self::DatesAndTimesWith(DateTimeEntityKind::Time).entity_kind(kind),
            TimeEntityFilter::DatesAndTimesWith(dt) => match kind {
                TimeKind::Date | TimeKind::DateRange => Some(DATE_ENTITY),
                TimeKind::Time | TimeKind::TimeRange => Some(TIME_ENTITY),
                TimeKind::DateTime | TimeKind::DateTimeRange => match dt {
                    DateTimeEntityKind::Time => Some(TIME_ENTITY),
                    DateTimeEntityKind::Date => Some(DATE_ENTITY),
                    DateTimeEntityKind::Skip => None,
                },
                TimeKind::Duration | TimeKind::Set => None,
            },
            TimeEntityFilter::TimesOnly => match kind {
                TimeKind::Time | TimeKind::TimeRange | TimeKind::DateTime | TimeKind::DateTimeRange => {
                    Some(TIME_ENTITY)
                }
                TimeKind::Date | TimeKind::DateRange | TimeKind::Duration | TimeKind::Set => None,
            },
            TimeEntityFilter::ClockTimesOnly => match kind {
                TimeKind::Time | TimeKind::TimeRange => Some(TIME_ENTITY),
                _ => None,
            },
            TimeEntityFilter::DatesOnly => match kind {
                TimeKind::Date | TimeKind::DateRange => Some(DATE_ENTITY),
                _ => None,
            },
        }
    }

    /// Deterministic configuration code for provenance hashing.
    pub const fn code(self) -> u64 {
        match self {
            TimeEntityFilter::DatesAndTimes => 0x10 | DateTimeEntityKind::Time.code(),
            TimeEntityFilter::DatesAndTimesWith(dt) => 0x10 | dt.code(),
            TimeEntityFilter::TimesOnly => 0x20,
            TimeEntityFilter::ClockTimesOnly => 0x30,
            TimeEntityFilter::DatesOnly => 0x40,
        }
    }

    fn tag_set(self) -> Arc<TagSet> {
        match self {
            TimeEntityFilter::DatesAndTimes | TimeEntityFilter::DatesAndTimesWith(_) => {
                TagSet::new(NER_TAGSET, &[DATE_ENTITY, TIME_ENTITY])
            }
            TimeEntityFilter::TimesOnly | TimeEntityFilter::ClockTimesOnly => TagSet::new(NER_TAGSET, &[TIME_ENTITY]),
            TimeEntityFilter::DatesOnly => TagSet::new(NER_TAGSET, &[DATE_ENTITY]),
        }
    }
}

/// Finds the bounding [`IxRange<Tokens>`] of tokens in `tokens` that overlap
/// the non-empty byte `span`, or `None` when `span` is empty or falls
/// strictly between tokens.
///
/// Uses binary search (`partition_point`) over the sorted token start and end
/// offset slices in $O(\log N)$ time with zero allocations.
pub fn span_token_range(tokens: &Tokens, span: Span) -> Option<IxRange<Tokens>> {
    if span.start >= span.end {
        return None;
    }
    let ts = tokens.spans.text_spans();
    let first = ts.end.partition_point(|&e| e <= span.start);
    let end = ts.start.partition_point(|&s| s < span.end);
    (first < end).then(|| IxRange::new(first as u32, end as u32))
}

/// Projects [`Temporals`] (`turbo:temporal`) mentions into [`Entities`]
/// (`turbo:entities`) typed [`DATE_ENTITY`] (`"date"`), aligned to `tokens`.
pub fn temporals_to_entities(temporals: &Temporals, tokens: &Tokens) -> Result<Entities, String> {
    if temporals.spans.view() != tokens.spans.view() {
        return Err("temporals and tokens are in different views".into());
    }
    let kinds = TagSet::new(NER_TAGSET, &[DATE_ENTITY]);
    let mut out = empty_entities(tokens, temporals.len(), kinds);
    for (_, span) in temporals.spans.iter() {
        if let Some(range) = span_token_range(tokens, span) {
            push_entity(&mut out, span, range, Tag(0));
        }
    }
    Ok(out)
}

/// Projects [`Times`] (`turbo:times`) mentions into [`Entities`]
/// (`turbo:entities`) using [`TimeEntityFilter::DatesAndTimes`], aligned to
/// `tokens`.
pub fn times_to_entities(times: &Times, tokens: &Tokens) -> Result<Entities, String> {
    times_to_entities_with(times, tokens, TimeEntityFilter::DatesAndTimes)
}

/// Projects [`Times`] (`turbo:times`) mentions into [`Entities`]
/// (`turbo:entities`) selected by `filter`, aligned to `tokens`.
pub fn times_to_entities_with(times: &Times, tokens: &Tokens, filter: TimeEntityFilter) -> Result<Entities, String> {
    if times.spans.view() != tokens.spans.view() {
        return Err("times and tokens are in different views".into());
    }
    let kinds = filter.tag_set();
    let mut out = empty_entities(tokens, times.len(), Arc::clone(&kinds));
    for (ix, span) in times.spans.iter() {
        let kind = times.kind[ix.get()];
        if let Some(label) = filter.entity_kind(kind)
            && let Some(range) = span_token_range(tokens, span)
        {
            let tag = kinds.tag(label).expect("filter tag_set includes every emitted label");
            push_entity(&mut out, span, range, tag);
        }
    }
    Ok(out)
}

/// Projects [`Money`] (`turbo:money`) mentions into [`Entities`]
/// (`turbo:entities`) typed [`MONEY_ENTITY`] (`"money"`), aligned to `tokens`.
pub fn money_to_entities(money: &Money, tokens: &Tokens) -> Result<Entities, String> {
    if money.spans.view() != tokens.spans.view() {
        return Err("money and tokens are in different views".into());
    }
    let kinds = TagSet::new(NER_TAGSET, &[MONEY_ENTITY]);
    let mut out = empty_entities(tokens, money.len(), kinds);
    for (_, span) in money.spans.iter() {
        if let Some(range) = span_token_range(tokens, span) {
            push_entity(&mut out, span, range, Tag(0));
        }
    }
    Ok(out)
}

/// Projects percentage mentions from [`Quantities`] (`turbo:quantities`, where
/// [`Quantities::is_percent`] holds) into [`Entities`] (`turbo:entities`)
/// typed [`PERCENT_ENTITY`] (`"percent"`), aligned to `tokens`.
///
/// Non-percentage unit quantities (such as `"80 kg"` or `"2.5 km"`) are left out.
pub fn quantities_to_entities(quantities: &Quantities, tokens: &Tokens) -> Result<Entities, String> {
    if quantities.spans.view() != tokens.spans.view() {
        return Err("quantities and tokens are in different views".into());
    }
    let kinds = TagSet::new(NER_TAGSET, &[PERCENT_ENTITY]);
    let mut out = empty_entities(tokens, quantities.len(), kinds);
    for (ix, span) in quantities.spans.iter() {
        if quantities.is_percent(ix)
            && let Some(range) = span_token_range(tokens, span)
        {
            push_entity(&mut out, span, range, Tag(0));
        }
    }
    Ok(out)
}

/// Projects any combination of [`Temporals`], [`Times`], [`Money`], and
/// [`Quantities`] into a single [`Entities`] layer over [`RULE_ENTITY_KINDS`].
///
/// Sources are resolved in specificity priority order ([`Money`], percentage
/// [`Quantities`], [`Temporals`], then [`Times`]): all mentions within a
/// single source layer are kept, while a mention from a later source is
/// retained only when neither its byte span nor its token range overlaps a
/// mention already kept from an earlier source.
pub fn rule_entities(
    tokens: &Tokens,
    temporals: Option<&Temporals>,
    times: Option<&Times>,
    money: Option<&Money>,
    quantities: Option<&Quantities>,
    time_filter: TimeEntityFilter,
) -> Result<Entities, String> {
    let view = tokens.spans.view();
    let kinds = TagSet::new(NER_TAGSET, &RULE_ENTITY_KINDS);
    let mut kept: Vec<(u8, Span, IxRange<Tokens>, Tag)> = Vec::new();

    if let Some(m) = money {
        if m.spans.view() != view {
            return Err("money and tokens are in different views".into());
        }
        let tag = kinds.tag(MONEY_ENTITY).expect("money in RULE_ENTITY_KINDS");
        for (_, span) in m.spans.iter() {
            if let Some(range) = span_token_range(tokens, span) {
                kept.push((0, span, range, tag));
            }
        }
    }
    if let Some(q) = quantities {
        if q.spans.view() != view {
            return Err("quantities and tokens are in different views".into());
        }
        let tag = kinds.tag(PERCENT_ENTITY).expect("percent in RULE_ENTITY_KINDS");
        for (ix, span) in q.spans.iter() {
            if q.is_percent(ix)
                && let Some(range) = span_token_range(tokens, span)
                && !overlaps_earlier(&kept, 1, span, range)
            {
                kept.push((1, span, range, tag));
            }
        }
    }
    if let Some(t) = temporals {
        if t.spans.view() != view {
            return Err("temporals and tokens are in different views".into());
        }
        let tag = kinds.tag(DATE_ENTITY).expect("date in RULE_ENTITY_KINDS");
        for (_, span) in t.spans.iter() {
            if let Some(range) = span_token_range(tokens, span)
                && !overlaps_earlier(&kept, 2, span, range)
            {
                kept.push((2, span, range, tag));
            }
        }
    }
    if let Some(t) = times {
        let projected = times_to_entities_with(t, tokens, time_filter)?;
        for (ix, span) in projected.spans.iter() {
            let i = ix.get();
            let range = projected.tokens[i];
            if !overlaps_earlier(&kept, 3, span, range) {
                let label = projected.kinds.label(projected.kind[i]);
                let tag = kinds.tag(label).expect("time filter label in RULE_ENTITY_KINDS");
                kept.push((3, span, range, tag));
            }
        }
    }

    kept.sort_by_key(|&(src, span, range, tag)| (span.start, Reverse(span.end), range.start.get(), src, tag.0));
    let mut out = empty_entities(tokens, kept.len(), kinds);
    for (_, span, range, tag) in kept {
        push_entity(&mut out, span, range, tag);
    }
    Ok(out)
}

fn overlaps_earlier(kept: &[(u8, Span, IxRange<Tokens>, Tag)], source: u8, span: Span, range: IxRange<Tokens>) -> bool {
    kept.iter().any(|&(s, k_span, k_range, _)| {
        s < source
            && ((span.start < k_span.end && k_span.start < span.end)
                || (range.start < k_range.end && k_range.start < range.end))
    })
}

fn empty_entities(tokens: &Tokens, capacity: usize, kinds: Arc<TagSet>) -> Entities {
    Entities {
        spans: Spans::with_capacity(tokens.spans.view(), capacity),
        tokens: Vec::with_capacity(capacity),
        kind: Vec::with_capacity(capacity),
        kinds,
        score: Vec::with_capacity(capacity),
    }
}

fn push_entity(out: &mut Entities, span: Span, range: IxRange<Tokens>, tag: Tag) {
    let _ = Ix::<Entities>::new(out.score.len() as u32);
    out.spans.push(span);
    out.tokens.push(range);
    out.kind.push(tag);
    out.score.push(RULE_SCORE);
}
