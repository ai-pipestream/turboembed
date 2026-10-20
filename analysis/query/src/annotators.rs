//! Query-tuned annotator steps:
//! - [`default_span_detector`]: an in-memory [`turbo_langspan::SpanDetector`] covering
//!   major Latin languages (`deu`, `eng`, `fra`, `ita`, `nld`, `por`, `spa`)
//!   and script-distinct languages (`ara`, `ell`, `heb`, `jpn`, `kor`, `rus`, `zho`).
//! - [`GazetteerCandidateAnnotator`]: fast multi-token gazetteer phrase
//!   matcher producing candidate location entities for [`turbo_geo::GeocodeAnnotator`].
//! - [`QueryGrammarAnnotator`] and [`QueryTimeAnnotator`]: trigger-gated
//!   wrappers around [`GrammarAnnotator`] and [`TimeAnnotator`] that skip
//!   chart parsing when no grammar/time lexical triggers are present.

use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, IxRange, Refs, Span, Spans, View};
use turbo_analysis_layers::{
    DateTime, DocumentDate, Entities, Money, Numbers, Ordinals, Quantities, Tag, TagSet, Times, Tokens,
};
use turbo_geo::{DEFAULT_LOCATION_TYPE, Gazetteer};
use turbo_grammar::{GRAMMAR, GrammarAnnotator, TimeAnnotator};
use turbo_stopword::Stopwords;

pub use crate::lang_model::default_span_detector;

/// Non-place query words that can collide with minor town names in a global gazetteer.
#[rustfmt::skip]
const NON_PLACE_QUERY_WORDS: &[&str] = &[
    "a", "about", "after", "ago", "all", "also", "am", "an", "and", "any", "april", "are", "as", "at", "august",
    "ayer", "baratos", "be", "before", "between", "billig", "billige", "both", "but", "by", "can", "cent", "cents",
    "cheap", "coming", "con", "day", "days", "de", "december", "del", "demain", "der", "des", "dollar", "dollars",
    "each", "el", "en", "euro", "euros", "every", "february", "flight", "flights", "for", "friday", "from", "het",
    "hotel", "hotels", "hour", "hours", "hoy", "in", "into", "is", "it", "january", "july", "june", "kg", "km", "la",
    "las", "last", "le", "les", "los", "low", "lunes", "mañana", "march", "martes", "may", "mes", "meses", "mi",
    "minute", "minutes", "monday", "month", "months", "morgen", "morning", "nach", "naar", "near", "next", "night",
    "no", "not", "november", "now", "october", "of", "on", "one", "only", "or", "over", "para", "past", "percent",
    "pm", "por", "pound", "pounds", "price", "quarter", "saturday", "second", "seconds", "semana", "september", "so",
    "some", "sunday", "tarde", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "three", "through", "thursday", "tickets", "time", "to", "today", "tomorrow", "tonight", "too", "tuesday", "two",
    "un", "una", "under", "until", "up", "usd", "van", "very", "vols", "vuelos", "was", "we", "wednesday", "week",
    "weeks", "what", "when", "where", "which", "while", "who", "with", "year", "years", "yesterday",
];

fn is_non_place_word(lower: &str) -> bool {
    NON_PLACE_QUERY_WORDS.contains(&lower)
}

/// Scans 1-to-4 token phrases against a [`Gazetteer`] and emits candidate
/// `"location"` entities in `opennlp:entities` for [`turbo_geo::GeocodeAnnotator`].
pub struct GazetteerCandidateAnnotator<G> {
    gazetteer: Arc<G>,
    kinds: Arc<TagSet>,
    min_lowercase_population: u64,
}

impl<G: Gazetteer> GazetteerCandidateAnnotator<G> {
    pub fn new(gazetteer: Arc<G>) -> Self {
        Self {
            gazetteer,
            kinds: TagSet::new("opennlp:entities", &[DEFAULT_LOCATION_TYPE]),
            min_lowercase_population: 50_000,
        }
    }

    pub fn bound(self, instance: &'static str) -> Bound<Self>
    where
        G: Send + Sync + 'static,
    {
        Bound::new(self).write(0, instance)
    }
}

impl<G: Gazetteer + Send + Sync + 'static> Annotator for GazetteerCandidateAnnotator<G> {
    const ID: &'static str = "turbo-query:gazetteer-candidates";
    type Reads = (View, Tokens, Option<Stopwords>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, stopwords): Refs<'_, Self>) -> Result<(Entities,), Error> {
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: Arc::clone(&self.kinds),
            score: Vec::new(),
        };

        let mut i = 0usize;
        while i < n {
            let max_w = 4.min(n - i);
            let mut matched = false;
            for w in (1..=max_w).rev() {
                let j = i + w;
                let span_start = starts[i] as usize;
                let span_end = ends[j - 1] as usize;
                let Some(surface) = text.get(span_start..span_end) else { continue };
                if !surface.chars().any(char::is_alphabetic) || surface.chars().any(|c| c.is_ascii_digit()) {
                    continue;
                }
                if w == 1 {
                    if surface.chars().count() < 2 {
                        continue;
                    }
                    if stopwords.is_some_and(|sw| sw.stop.get(i).copied().unwrap_or(false)) {
                        continue;
                    }
                    let lower = surface.to_lowercase();
                    if is_non_place_word(&lower) {
                        continue;
                    }
                } else {
                    let first_tok = &text[starts[i] as usize..ends[i] as usize];
                    let last_tok = &text[starts[j - 1] as usize..ends[j - 1] as usize];
                    if is_non_place_word(&first_tok.to_lowercase()) || is_non_place_word(&last_tok.to_lowercase()) {
                        continue;
                    }
                }

                let entries = self.gazetteer.lookup(surface);
                if let Some(best) = entries.first() {
                    let starts_upper = surface.chars().next().is_some_and(char::is_uppercase);
                    if w == 1 && !starts_upper && best.population() < self.min_lowercase_population {
                        continue;
                    }
                    out.spans.push(Span::new(starts[i], ends[j - 1]));
                    out.tokens.push(IxRange::new(i as u32, j as u32));
                    out.kind.push(Tag(0));
                    out.score.push(1.0);
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }
        Ok((out,))
    }
}

/// Trigger-gated wrapper around [`GrammarAnnotator`] that runs the chart
/// parser only when the query contains spelled-out number/unit/currency tokens.
pub struct QueryGrammarAnnotator {
    inner: GrammarAnnotator,
}

impl QueryGrammarAnnotator {
    pub fn english() -> Self {
        Self { inner: GrammarAnnotator::english() }
    }

    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, GRAMMAR).write(1, GRAMMAR).write(3, GRAMMAR)
    }
}

#[rustfmt::skip]
const GRAMMAR_TRIGGERS: &[&str] = &[
    "a", "an", "billion", "cent", "cents", "couple", "dozen", "eight", "eighteen", "eighth", "eighty", "eleven",
    "euro", "euros", "feet", "fifteen", "fifth", "fifty", "first", "five", "foot", "forty", "four", "fourteen",
    "fourth", "franc", "francs", "gallon", "gallons", "gram", "grams", "half", "hundred", "inch", "inches", "kilo",
    "kilogram", "kilograms", "kilometer", "kilometers", "kilos", "liter", "liters", "meter", "meters", "mile", "miles",
    "million", "minus", "nine", "nineteen", "ninety", "once", "one", "ounce", "ounces", "pair", "per", "percent",
    "percentage", "pint", "pints", "pound", "pounds", "quarter", "second", "seven", "seventeen", "seventy", "six",
    "sixteen", "sixty", "ten", "tenth", "third", "thirteen", "thirty", "thousand", "three", "trillion", "twelve",
    "twentieth", "twenty", "twice", "two", "yard", "yards", "yen", "zero",
];

fn has_grammar_trigger(text: &str, tokens: &Tokens) -> bool {
    let (starts, ends) = tokens.spans.columns();
    let mut hits = 0usize;
    for (&s, &e) in starts.iter().zip(ends) {
        if let Some(tok) = text.get(s as usize..e as usize) {
            let lower = tok.to_ascii_lowercase();
            if GRAMMAR_TRIGGERS.contains(&lower.as_str()) {
                if lower != "a" && lower != "an" {
                    return true;
                }
                hits += 1;
            }
        }
    }
    hits >= 2
}

impl Annotator for QueryGrammarAnnotator {
    const ID: &'static str = "turbo-query:grammar";
    type Reads = (View, Tokens);
    type Writes = (Numbers, Quantities, Ordinals, Money);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if !has_grammar_trigger(view.text(), tokens) {
            return Ok((
                Numbers::new(view.id()),
                Quantities::new(view.id()),
                Ordinals::new(view.id()),
                Money::new(view.id()),
            ));
        }
        self.inner.annotate((view,))
    }
}

/// Trigger-gated wrapper around [`TimeAnnotator`] that runs the time chart
/// parser only when the query contains time/date keywords not already covered
/// by `TemporalExtractor`, or Spanish temporal triggers.
pub struct QueryTimeAnnotator {
    inner: TimeAnnotator,
    reference: Option<DateTime>,
    triggers: &'static [&'static str],
}

#[rustfmt::skip]
const EN_TIME_TRIGGERS: &[&str] = &[
    "after", "afternoon", "am", "april", "aug", "august", "before", "between", "christmas", "coming", "dawn", "day",
    "days", "dec", "december", "dusk", "easter", "end", "eve", "evening", "every", "feb", "february", "for", "fri",
    "friday", "from", "half", "hour", "hours", "jan", "january", "jul", "july", "jun", "june", "last", "mar", "march",
    "may", "mid", "midnight", "min", "mins", "minute", "minutes", "mon", "monday", "month", "months", "morning",
    "next", "night", "noon", "nov", "november", "now", "oct", "october", "past", "pm", "quarter", "sat", "saturday",
    "sec", "second", "seconds", "sep", "sept", "september", "since", "start", "sun", "sunday", "this", "thu", "thurs",
    "thursday", "till", "today", "tomorrow", "tonight", "tue", "tues", "tuesday", "until", "wed", "wednesday", "week",
    "weekend", "weeks", "year", "years", "yesterday",
];

#[rustfmt::skip]
const ES_TIME_TRIGGERS: &[&str] = &[
    "abril", "agosto", "ayer", "diciembre", "domingo", "día", "días", "enero", "esta", "este", "febrero", "hora",
    "horas", "hoy", "jueves", "julio", "junio", "lunes", "martes", "marzo", "mayo", "mañana", "mediodía", "mes",
    "meses", "miércoles", "minuto", "minutos", "noche", "noviembre", "octubre", "pasada", "pasado", "próxima",
    "próximo", "semana", "semanas", "septiembre", "sábado", "tarde", "viernes",
];

impl QueryTimeAnnotator {
    pub fn english(reference: Option<DateTime>) -> Self {
        let mut inner = TimeAnnotator::english();
        if let Some(r) = reference {
            inner = inner.with_reference(r);
        }
        Self { inner, reference, triggers: EN_TIME_TRIGGERS }
    }

    pub fn spanish(reference: Option<DateTime>) -> Self {
        let mut inner = TimeAnnotator::spanish();
        if let Some(r) = reference {
            inner = inner.with_reference(r);
        }
        Self { inner, reference, triggers: ES_TIME_TRIGGERS }
    }
}

impl Annotator for QueryTimeAnnotator {
    const ID: &'static str = "turbo-query:times";
    type Reads = (View, Tokens, Option<DocumentDate>);
    type Writes = (Times,);

    fn annotate(&self, (view, tokens, doc_date): Refs<'_, Self>) -> Result<(Times,), Error> {
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let triggered = starts.iter().zip(ends).any(|(&s, &e)| {
            text.get(s as usize..e as usize).is_some_and(|tok| {
                let lower = tok.to_lowercase();
                self.triggers.contains(&lower.as_str())
            })
        });
        if !triggered {
            return Ok((Times::new(view.id()),));
        }
        let r = doc_date.and_then(|d| d.date).map(DateTime::midnight).or(self.reference);
        Ok((self.inner.find(text, view.id(), r),))
    }
}
