//! Typed query description, structured search filters, language spans,
//! and layer provenance evidence produced by [`crate::QueryAnalysisProfile`].

use turbo_analysis::Span;
use turbo_analysis_layers::{
    Currency, DAY_SECONDS, Date, DateTime, Granularity, Modifier, Origin, RegionCode, TemporalValue, TimeGrain,
    TimeKind,
};
use turbo_chunker::terms::{Bm25, Collection, Query as Bm25Query, Vocabulary};
use turbo_geo::GeoPoint;
use turbo_unicode::Script;

/// Provenance tying a query filter or language span to the document layer
/// and annotator step that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterEvidence {
    /// Standard layer identifier (such as `"turbo:temporal"`, `"turbo:times"`,
    /// `"turbo:money"`, `"turbo:quantities"`, `"turbo:locations"`,
    /// `"turbo:glossary-terms"`, `"opennlp:entities"`, `"turbo:language-spans"`).
    pub layer_id: &'static str,
    /// Optional layer instance name when produced by a named layer binding
    /// (for example `Some("grammar")` or `Some("es")`).
    pub instance: Option<String>,
    /// Annotator ID recorded in the document's layer provenance.
    pub annotator: String,
}

/// A temporal expression in a query resolved to an epoch-second range
/// `[start_epoch_seconds, end_epoch_seconds)` or a duration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalFilter {
    /// Inclusive UTC start timestamp in seconds since `1970-01-01T00:00:00Z`.
    pub start_epoch_seconds: Option<i64>,
    /// Exclusive UTC end timestamp in seconds since `1970-01-01T00:00:00Z`.
    pub end_epoch_seconds: Option<i64>,
    /// Duration in milliseconds when the expression denotes a duration or range.
    pub duration_ms: Option<i64>,
    /// Normalized ISO-8601 or TIMEX3 representation of the temporal value.
    pub value_text: String,
    /// Kind of temporal mention (`Date`, `Time`, `DateTime`, `Duration`, `Set`).
    pub kind: TimeKind,
    /// Finest calendar/clock unit of the mention.
    pub grain: TimeGrain,
    /// TIMEX3 modifier (`Exact`, `Approx`, `LessThan`, `MoreThan`, `Start`, `Mid`, `End`).
    pub modifier: Modifier,
    /// Whether the mention was stated as an absolute date or resolved relative to the anchor date.
    pub origin: Origin,
    /// Whether multiple candidate readings existed before picking the preferred resolution.
    pub ambiguous: bool,
}

/// A monetary amount filter with exact decimal components and ISO 4217 currency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoneyFilter {
    /// Formatted signed decimal string (for example `"1200000.0"` or `"0.50"`).
    pub decimal_text: String,
    /// Whether the amount is negative.
    pub negative: bool,
    /// Unscaled decimal digits (`BigDecimal` coefficient).
    pub coefficient: String,
    /// Decimal scale (`BigDecimal.scale()`).
    pub scale: i32,
    /// Three-letter ISO 4217 currency code.
    pub currency: Currency,
}

/// A measured quantity or percentage filter with exact decimal components and unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantityFilter {
    /// Formatted signed decimal string.
    pub decimal_text: String,
    /// Whether the value is negative.
    pub negative: bool,
    /// Unscaled decimal digits.
    pub coefficient: String,
    /// Decimal scale.
    pub scale: i32,
    /// Unit symbol or token (`"%"` for percentages, `"kg"`, `"km"`, etc.).
    pub unit: String,
    /// Whether this quantity represents a percentage (`unit == "%"`).
    pub is_percent: bool,
}

/// A geographic place filter resolved against a [`turbo_geo::Gazetteer`].
#[derive(Clone, Debug, PartialEq)]
pub struct PlaceFilter {
    /// Stable record ID within `source` (for example `"1159151259"`).
    pub record_id: String,
    /// Gazetteer dataset source identifier (for example `"naturalearth"`).
    pub source: String,
    /// Canonical place name in the gazetteer.
    pub canonical_name: String,
    /// ISO 3166-1 alpha-2 country/region code when present.
    pub country_code: Option<RegionCode>,
    /// Administrative containment hierarchy, outermost first.
    pub containment: Vec<String>,
    /// Gazetteer feature class (such as `"CITY"`, `"ADMIN"`, `"POI"`).
    pub feature_class: Option<String>,
    /// WGS84 coordinates (latitude, longitude).
    pub location: GeoPoint,
    /// Population recorded in the gazetteer (`0` when unknown).
    pub population: i64,
    /// Geocoder resolution confidence in `[0.0, 1.0]`.
    pub confidence: f64,
}

/// A domain glossary term matched in the query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlossaryFilter {
    /// Stable concept/entry identifier in the glossary.
    pub id: String,
    /// Canonical matched term from the glossary entry.
    pub term: String,
    /// Entry index in the [`turbo_glossary_layers::Glossary`].
    pub entry_index: u32,
}

/// A named or pattern entity matched in the query (such as email, URL, phone, or dictionary entity).
#[derive(Clone, Debug, PartialEq)]
pub struct EntityFilter {
    /// Entity type label (for example `"email"`, `"url"`, `"product"`, `"person"`).
    pub entity_type: String,
    /// Finder confidence score.
    pub score: f64,
}

/// Specific payload of a [`QueryFilter`].
#[derive(Clone, Debug, PartialEq)]
pub enum FilterKind {
    Temporal(TemporalFilter),
    Money(MoneyFilter),
    Quantity(QuantityFilter),
    Place(PlaceFilter),
    Glossary(GlossaryFilter),
    Entity(EntityFilter),
}

/// A structured search filter extracted from a query, tied to its exact byte
/// [`Span`] in the source query and the layer [`FilterEvidence`] that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryFilter {
    /// UTF-8 byte span in the source query (`&query[span.range()] == surface`).
    pub span: Span,
    /// Exact substring of the source query covered by `span`.
    pub surface: String,
    /// Layer and annotator provenance for this filter.
    pub evidence: FilterEvidence,
    /// Typed filter payload.
    pub kind: FilterKind,
}

/// One analyzed query token and its normalized/stemmed/indexed representations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryTerm {
    /// Zero-based token position in `opennlp:tokens`.
    pub token_index: usize,
    /// UTF-8 byte span in the source query.
    pub span: Span,
    /// Exact surface slice in the source query.
    pub surface: String,
    /// Case- and accent-folded token form from `turbo:terms`.
    pub folded: String,
    /// Stemmed token form from `turbo:terms`, when stemming is active.
    pub stem: Option<String>,
    /// Final term string indexed into `turbo:vocabulary` / `turbo:indexed-terms`
    /// (`None` for punctuation-only tokens that are not indexed).
    pub indexed_term: Option<String>,
    /// Whether `turbo:stopwords` flagged this token as a stopword.
    pub is_stopword: bool,
    /// Whether this token's source span is covered by at least one [`QueryFilter`].
    pub covered_by_filter: bool,
}

/// A contiguous single-language run within the query from `turbo:language-spans`.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageSpanEvidence {
    /// UTF-8 byte span in the source query.
    pub span: Span,
    /// Exact substring of the source query covered by `span`.
    pub surface: String,
    /// ISO 639-3 language code (or `"und"` when undetermined).
    pub code: String,
    /// Dominant Unicode script of the run's letters.
    pub script: Script,
    /// Language probability over the candidate languages of `script`.
    pub confidence: f64,
    /// Layer and annotator provenance.
    pub evidence: FilterEvidence,
}

/// Document-level and span-level language detection evidence for the query.
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageEvidence {
    /// Best overall ISO 639-3 language code from `opennlp:language`.
    pub best_code: String,
    /// Letter-share score of `best_code` in `[0.0, 1.0]`.
    pub best_score: f64,
    /// All detected languages in descending score order.
    pub rankings: Vec<(String, f64)>,
    /// Contiguous per-span language runs from `turbo:language-spans`.
    pub spans: Vec<LanguageSpanEvidence>,
    /// Layer and annotator provenance for `opennlp:language`.
    pub evidence: FilterEvidence,
}

/// Complete structured output of running [`crate::QueryAnalysisProfile`] on a query.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryAnalysis {
    /// Original query string.
    pub query: String,
    /// Detected language summary and per-clause/per-script language spans.
    pub language: Option<LanguageEvidence>,
    /// Per-token linguistic and lexical analysis aligned to `opennlp:tokens`.
    pub terms: Vec<QueryTerm>,
    /// Indexed lexical terms in token order, identical to the document profile's
    /// indexed terms for the same text.
    pub bm25_terms: Vec<String>,
    /// Indexed non-stopword lexical terms whose spans are not consumed by any [`QueryFilter`].
    pub residual_terms: Vec<String>,
    /// Source byte spans of tokens not consumed by any [`QueryFilter`].
    pub free_text_spans: Vec<Span>,
    /// Residual free-text surface tokens joined by single ASCII spaces.
    pub free_text: String,
    /// Extracted structured filters in source-span order.
    pub filters: Vec<QueryFilter>,
}

impl QueryAnalysis {
    /// Builds a [`Bm25Query`] from [`Self::bm25_terms`] for scoring document chunks or units.
    pub fn bm25_query(&self, model: Bm25, collection: &impl Collection, vocabulary: &Vocabulary) -> Bm25Query {
        let refs: Vec<&str> = self.bm25_terms.iter().map(String::as_str).collect();
        Bm25Query::new(model, collection, vocabulary, &refs)
    }

    /// Builds a [`Bm25Query`] from only the unfiltered [`Self::residual_terms`].
    pub fn residual_bm25_query(&self, model: Bm25, collection: &impl Collection, vocabulary: &Vocabulary) -> Bm25Query {
        let refs: Vec<&str> = self.residual_terms.iter().map(String::as_str).collect();
        Bm25Query::new(model, collection, vocabulary, &refs)
    }

    /// Iterates over all [`TemporalFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn temporal_filters(&self) -> impl Iterator<Item = (&QueryFilter, &TemporalFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Temporal(t) => Some((f, t)),
            _ => None,
        })
    }

    /// Iterates over all [`MoneyFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn money_filters(&self) -> impl Iterator<Item = (&QueryFilter, &MoneyFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Money(m) => Some((f, m)),
            _ => None,
        })
    }

    /// Iterates over all [`QuantityFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn quantity_filters(&self) -> impl Iterator<Item = (&QueryFilter, &QuantityFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Quantity(q) => Some((f, q)),
            _ => None,
        })
    }

    /// Iterates over all [`PlaceFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn place_filters(&self) -> impl Iterator<Item = (&QueryFilter, &PlaceFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Place(p) => Some((f, p)),
            _ => None,
        })
    }

    /// Iterates over all [`GlossaryFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn glossary_filters(&self) -> impl Iterator<Item = (&QueryFilter, &GlossaryFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Glossary(g) => Some((f, g)),
            _ => None,
        })
    }

    /// Iterates over all [`EntityFilter`]s with their source [`Span`] and [`FilterEvidence`].
    pub fn entity_filters(&self) -> impl Iterator<Item = (&QueryFilter, &EntityFilter)> {
        self.filters.iter().filter_map(|f| match &f.kind {
            FilterKind::Entity(e) => Some((f, e)),
            _ => None,
        })
    }
}

/// Resolves a calendar [`TemporalValue`] from `turbo:temporal` into an exact
/// half-open UTC epoch-second range `[start_seconds, end_seconds)` and maps
/// its [`Granularity`] to [`TimeGrain`].
pub(crate) fn resolve_temporal_value_range(val: TemporalValue) -> (Option<i64>, Option<i64>, TimeGrain) {
    let grain = match val.granularity() {
        Granularity::Day => TimeGrain::Day,
        Granularity::Week => TimeGrain::Week,
        Granularity::Month => TimeGrain::Month,
        Granularity::Quarter => TimeGrain::Quarter,
        Granularity::Year => TimeGrain::Year,
    };
    let range = match val {
        TemporalValue::Day(d) => {
            let start = DateTime::midnight(d).local_seconds();
            Some((start, start + DAY_SECONDS))
        }
        TemporalValue::Week { year, week } => iso_week_start(year, week).map(|d| {
            let start = DateTime::midnight(d).local_seconds();
            (start, start + 7 * DAY_SECONDS)
        }),
        TemporalValue::Month { year, month } => {
            let start_d = Date::new(year, month, 1);
            let (ny, nm) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
            let end_d = Date::new(ny, nm, 1);
            start_d
                .zip(end_d)
                .map(|(s, e)| (DateTime::midnight(s).local_seconds(), DateTime::midnight(e).local_seconds()))
        }
        TemporalValue::Quarter { year, quarter } => {
            let start_month = (quarter.saturating_sub(1) * 3) + 1;
            let start_d = Date::new(year, start_month, 1);
            let (ny, nm) = if start_month + 3 > 12 { (year + 1, 1) } else { (year, start_month + 3) };
            let end_d = Date::new(ny, nm, 1);
            start_d
                .zip(end_d)
                .map(|(s, e)| (DateTime::midnight(s).local_seconds(), DateTime::midnight(e).local_seconds()))
        }
        TemporalValue::Year(year) => {
            let start_d = Date::new(year, 1, 1);
            let end_d = Date::new(year + 1, 1, 1);
            start_d
                .zip(end_d)
                .map(|(s, e)| (DateTime::midnight(s).local_seconds(), DateTime::midnight(e).local_seconds()))
        }
    };
    match range {
        Some((s, e)) => (Some(s), Some(e), grain),
        None => (None, None, grain),
    }
}

/// Computes the Monday [`Date`] of ISO week `week` in ISO week-based `year`.
fn iso_week_start(year: i32, week: u8) -> Option<Date> {
    let jan4 = Date::new(year, 1, 4)?;
    let jan4_epoch = jan4.epoch_day();
    // 1970-01-01 was a Thursday (weekday 3 in Monday=0..Sunday=6).
    let jan4_monday_offset = (jan4_epoch + 3).rem_euclid(7);
    let week1_monday = jan4_epoch - jan4_monday_offset;
    Date::from_epoch_day(week1_monday + i64::from(week.saturating_sub(1)) * 7)
}

/// Converts a point timestamp with a known [`TimeGrain`] into an `[start, end)`
/// interval when the grammar resolved a date/time mention to a single instant.
pub(crate) fn point_end_from_grain(start_seconds: i64, grain: TimeGrain) -> i64 {
    let step = match grain {
        TimeGrain::Second => 1,
        TimeGrain::Minute => 60,
        TimeGrain::Hour => 3_600,
        TimeGrain::PartOfDay => 4 * 3_600,
        TimeGrain::Day => DAY_SECONDS,
        TimeGrain::Weekend => 2 * DAY_SECONDS,
        TimeGrain::Week => 7 * DAY_SECONDS,
        TimeGrain::Month => 30 * DAY_SECONDS,
        TimeGrain::Quarter | TimeGrain::Season => 91 * DAY_SECONDS,
        TimeGrain::Half => 182 * DAY_SECONDS,
        TimeGrain::Year => 365 * DAY_SECONDS,
        TimeGrain::Decade => 3_652 * DAY_SECONDS,
        TimeGrain::Century => 36_525 * DAY_SECONDS,
    };
    start_seconds.saturating_add(step)
}
