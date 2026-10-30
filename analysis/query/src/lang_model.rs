//! Model-free script evidence and query lexical helpers.

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Tokens};
use turbo_langspan::{NgramDetector, ScriptDetector};

/// Script-only evidence until the trained language head in #1525 is available.
pub fn default_span_detector() -> ScriptDetector {
    ScriptDetector
}

/// Query wrapper for script or trained n-gram language detection.
pub enum QuerySpanDetector {
    Script(ScriptDetector),
    Ngram(NgramDetector),
}

impl QuerySpanDetector {
    /// Wraps model-free script evidence for query processing.
    pub fn new(inner: ScriptDetector) -> Self {
        Self::Script(inner)
    }

    /// Wraps the trained n-gram language detector.
    pub fn with_ngram(inner: NgramDetector) -> Self {
        Self::Ngram(inner)
    }
}

impl Annotator for QuerySpanDetector {
    const ID: &'static str = ScriptDetector::ID;
    type Reads = (View, Option<Sentences>, Option<Tokens>);
    type Writes = (LanguageSpans, Language);

    fn model(&self) -> Option<ModelRef> {
        match self {
            Self::Script(inner) => inner.model(),
            Self::Ngram(inner) => inner.model(),
        }
    }

    fn config_hash(&self) -> u64 {
        match self {
            Self::Script(inner) => inner.config_hash(),
            Self::Ngram(inner) => inner.config_hash(),
        }
    }

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        match self {
            Self::Script(inner) => inner.annotate(input),
            Self::Ngram(inner) => inner.annotate(input),
        }
    }
}

pub(crate) fn with_lowercase<R>(tok: &str, f: impl FnOnce(&str) -> R) -> R {
    if tok.is_ascii() && tok.len() <= 64 {
        let mut buf = [0u8; 64];
        for (dst, src) in buf[..tok.len()].iter_mut().zip(tok.bytes()) {
            *dst = src.to_ascii_lowercase();
        }
        f(std::str::from_utf8(&buf[..tok.len()]).expect("ASCII is valid UTF-8"))
    } else {
        f(&tok.to_lowercase())
    }
}

pub(crate) use crate::SequentialBuilder;

#[rustfmt::skip]
pub(crate) fn is_non_place_word(lower: &str) -> bool {
    matches!(
        lower,
        "a" | "about" | "after" | "ago" | "all" | "also" | "am" | "an" | "and" | "any" | "april" | "are" | "as" | "at" | "august"
            | "ayer" | "baratos" | "be" | "before" | "between" | "billig" | "billige" | "both" | "but" | "by" | "can" | "cent" | "cents"
            | "cheap" | "coming" | "con" | "day" | "days" | "de" | "december" | "del" | "demain" | "der" | "des" | "dollar" | "dollars"
            | "each" | "el" | "en" | "euro" | "euros" | "every" | "february" | "flight" | "flights" | "for" | "friday" | "from" | "het"
            | "hotel" | "hotels" | "hour" | "hours" | "hoy" | "in" | "into" | "is" | "it" | "january" | "july" | "june" | "kg" | "km" | "la"
            | "las" | "last" | "le" | "les" | "los" | "low" | "lunes" | "mañana" | "march" | "martes" | "may" | "mes" | "meses" | "mi"
            | "minute" | "minutes" | "monday" | "month" | "months" | "morgen" | "morning" | "nach" | "naar" | "near" | "next" | "night"
            | "no" | "not" | "november" | "now" | "october" | "of" | "on" | "one" | "only" | "or" | "over" | "para" | "past" | "percent"
            | "pm" | "por" | "pound" | "pounds" | "price" | "quarter" | "saturday" | "second" | "seconds" | "semana" | "september" | "so"
            | "some" | "sunday" | "tarde" | "than" | "that" | "the" | "their" | "them" | "then" | "there" | "these" | "they" | "this"
            | "three" | "through" | "thursday" | "tickets" | "time" | "to" | "today" | "tomorrow" | "tonight" | "too" | "tuesday" | "two"
            | "un" | "una" | "under" | "until" | "up" | "usd" | "van" | "very" | "vols" | "vuelos" | "was" | "we" | "wednesday" | "week"
            | "weeks" | "what" | "when" | "where" | "which" | "while" | "who" | "with" | "year" | "years" | "yesterday"
    )
}

#[rustfmt::skip]
pub(crate) fn is_grammar_trigger(lower: &str) -> bool {
    matches!(
        lower,
        "a" | "an" | "billion" | "couple" | "dozen" | "eight" | "eighteen" | "eighth" | "eighty"
            | "eleven" | "fifteen" | "fifth" | "fifty" | "first" | "five" | "forty" | "four"
            | "fourteen" | "fourth" | "half" | "hundred" | "million" | "minus" | "nine"
            | "nineteen" | "ninety" | "once" | "one" | "pair" | "quarter" | "second" | "seven"
            | "seventeen" | "seventy" | "six" | "sixteen" | "sixty" | "ten" | "tenth" | "third"
            | "thirteen" | "thirty" | "thousand" | "three" | "trillion" | "twelve" | "twentieth"
            | "twenty" | "twice" | "two" | "zero"
    )
}

#[rustfmt::skip]
pub(crate) fn is_temporal_cursor_token(tok: &str) -> bool {
    if !tok.is_ascii() {
        return false;
    }
    if tok.bytes().any(|b| b.is_ascii_digit()) {
        return (tok.len() == 4 && tok.bytes().all(|b| b.is_ascii_digit()))
            || (tok.len() >= 2 && (tok.bytes().any(|b| b == b'-') || matches!(tok.as_bytes()[0], b'Q' | b'q')));
    }
    (3..=8).contains(&tok.len()) && with_lowercase(tok, |l| {
        matches!(l, "ago" | "day" | "days" | "month" | "months" | "quarter" | "quarters" | "week" | "weeks" | "year" | "years")
    })
}

#[rustfmt::skip]
pub(crate) fn is_en_time_trigger(tok: &str) -> bool {
    !tok.bytes().any(|b| b.is_ascii_digit()) && (2..=13).contains(&tok.len()) && with_lowercase(tok, |l| {
        matches!(
            l,
            "afternoon" | "afternoons" | "am" | "annual" | "annually" | "apr" | "april" | "aug" | "august" | "autumn" | "biannual"
                | "bimonthly" | "biweekly" | "breakfast" | "christmas" | "coming" | "current" | "daily" | "dawn" | "day" | "days" | "daytime"
                | "dec" | "decade" | "decades" | "december" | "dinner" | "dinnertime" | "dusk" | "easter" | "eight" | "eighteen" | "eighties"
                | "eighty" | "eleven" | "eve" | "evening" | "evenings" | "every" | "everyday" | "fall" | "feb" | "february" | "fifteen"
                | "fifties" | "fifty" | "five" | "following" | "forties" | "fortnight" | "fortnightly" | "fortnights" | "forty" | "four"
                | "fourteen" | "fri" | "friday" | "fridays" | "half" | "halloween" | "hour" | "hourly" | "hours" | "hr" | "hrs" | "hundred"
                | "jan" | "january" | "jul" | "july" | "jun" | "june" | "juneteenth" | "last" | "lunch" | "lunchtime" | "mar" | "march" | "may"
                | "mid" | "midnight" | "min" | "mins" | "minute" | "minutes" | "mon" | "monday" | "mondays" | "month" | "monthly" | "months"
                | "morning" | "mornings" | "next" | "night" | "nightly" | "nights" | "nighttime" | "nine" | "nineteen" | "nineties" | "ninety"
                | "noon" | "nov" | "november" | "now" | "oct" | "october" | "one" | "otd" | "past" | "pm" | "previous" | "prior" | "quarter"
                | "quarterly" | "quarters" | "sat" | "saturday" | "saturdays" | "sec" | "second" | "seconds" | "secs" | "semiannual"
                | "semiannually" | "sep" | "sept" | "september" | "seven" | "seventeen" | "seventies" | "seventy" | "six" | "sixteen"
                | "sixties" | "sixty" | "spring" | "summer" | "sun" | "sunday" | "sundays" | "ten" | "thanksgiving" | "thirteen" | "thirties"
                | "thirty" | "this" | "three" | "thu" | "thur" | "thurs" | "thursday" | "thursdays" | "tmr" | "tmrw" | "today" | "tomorrow"
                | "tonight" | "tonite" | "tue" | "tues" | "tuesday" | "tuesdays" | "twelve" | "twenties" | "twenty" | "two" | "upcoming"
                | "wed" | "wednesday" | "wednesdays" | "week" | "weekday" | "weekdays" | "weekend" | "weekends" | "weekly" | "weeks"
                | "winter" | "wk" | "wks" | "xmas" | "year" | "yearly" | "years" | "yesterday" | "yr" | "yrs"
        )
    })
}

#[rustfmt::skip]
pub(crate) fn is_es_time_trigger(tok: &str, sole_english: bool) -> bool {
    !tok.bytes().any(|b| b.is_ascii_digit()) && (2..=16).contains(&tok.len()) && with_lowercase(tok, |l| {
        matches!(
            l,
            "abr" | "abril" | "actual" | "agosto" | "anoche" | "anos" | "anteayer" | "anterior" | "anteriores" | "antier"
                | "anual" | "anuales" | "anualmente" | "ayer" | "año" | "años" | "bimensual" | "bimensuales" | "bimensualmente" | "calendario"
                | "cada" | "decada" | "decadas" | "decenio" | "decenios" | "dia" | "diariamente" | "dias" | "dic" | "diciembre" | "dom"
                | "domingo" | "domingos" | "día" | "días" | "ene" | "enero" | "escolar" | "esta" | "estas" | "este" | "estos"
                | "febrero" | "finde" | "findes" | "fiscal" | "habil" | "hora" | "horas" | "hoy" | "hra" | "hras" | "invierno"
                | "jue" | "jueves" | "julio" | "junio" | "lun" | "lunes" | "martes" | "marzo" | "mayo"
                | "mañana" | "mañanas" | "mediodia" | "mediodía" | "mensual" | "mensuales" | "mensualmente" | "mes" | "meses" | "mie" | "mier"
                | "miercoles" | "minuto" | "minutos" | "miércoles" | "navidad" | "noche" | "nochebuena" | "noches"
                | "nochevieja" | "noviembre" | "octubre" | "otoño" | "pasada" | "pasadas" | "pasado" | "pasados" | "pascua"
                | "primavera" | "proxima" | "proximas" | "proximo" | "proximos" | "próxima" | "próximas" | "próximo" | "próximos"
                | "quincena" | "quincenal" | "quincenales" | "quincenalmente" | "quincenas" | "sab" | "sabado" | "sabados" | "seg"
                | "segundo" | "segundos" | "segs" | "semana" | "semanal" | "semanales" | "semanalmente" | "semanas" | "semestral"
                | "semestrales" | "semestralmente" | "semestre" | "semestres" | "septiembre" | "set" | "setiembre"
                | "siguiente" | "siguientes" | "sábado" | "sábados" | "tarde" | "tardes" | "trimestral" | "trimestrales" | "trimestralmente"
                | "trimestre" | "trimestres" | "ultima" | "ultimas" | "ultimo" | "ultimos" | "verano" | "vie" | "viernes" | "última"
                | "últimas" | "último" | "últimos"
        ) || (!sole_english && matches!(l, "ago" | "feb" | "hr" | "hrs" | "jul" | "jun" | "mar" | "may" | "min" | "mins" | "nov" | "oct" | "sep" | "sept"))
    })
}

#[rustfmt::skip]
pub(crate) const COMMON_EN_TIME_PHRASES: &[&str] = &[
    "last friday", "last monday", "last month", "last night", "last quarter", "last saturday", "last sunday",
    "last thursday", "last tuesday", "last wednesday", "last week", "last weekend", "last year", "next friday",
    "next monday", "next month", "next quarter", "next saturday", "next sunday", "next thursday", "next tuesday",
    "next wednesday", "next week", "next weekend", "next year", "now", "this afternoon", "this evening", "this friday",
    "this monday", "this month", "this morning", "this night", "this quarter", "this saturday", "this sunday",
    "this thursday", "this tuesday", "this wednesday", "this week", "this weekend", "this year", "today", "tomorrow",
    "tomorrow afternoon", "tomorrow evening", "tomorrow morning", "tomorrow night", "tonight", "yesterday",
];

#[rustfmt::skip]
pub(crate) const COMMON_ES_TIME_PHRASES: &[&str] = &[
    "anoche", "ayer", "el mes pasado", "esta mañana", "esta noche", "esta semana", "esta tarde", "este año",
    "este mes", "hoy", "la semana pasada", "mañana",
];

#[rustfmt::skip]
pub(crate) fn is_place_connector(lower: &str) -> bool {
    matches!(
        lower,
        "am" | "an" | "and" | "da" | "das" | "de" | "del" | "des" | "do" | "dos" | "du"
            | "el" | "la" | "las" | "le" | "les" | "los" | "of" | "on" | "st" | "upon"
    )
}

pub(crate) fn is_loc_preposition(lower: &str) -> bool {
    matches!(lower, "a" | "at" | "en" | "from" | "in" | "naar" | "nach" | "near" | "to")
}

pub(crate) fn is_numeric_time_token(text: &str, s: usize, e: usize) -> bool {
    let tok = &text[s..e];
    if !tok.bytes().any(|b| b.is_ascii_digit()) {
        return false;
    }
    if (tok.len() == 4 && tok.bytes().all(|b| b.is_ascii_digit()))
        || tok.bytes().any(|b| matches!(b, b':' | b'/' | b'-' | b'.') || b.is_ascii_alphabetic())
    {
        return true;
    }
    let prev = text.as_bytes().get(s.wrapping_sub(1)).copied().unwrap_or(b' ');
    let next = text.as_bytes().get(e).copied().unwrap_or(b' ');
    matches!(prev, b':' | b'/' | b'-' | b'.') || matches!(next, b':' | b'/' | b'-' | b'.')
}

/// Whether a lowercased token modifies a time expression's
/// relation or position within its period, so dropping it would
/// change the meaning (`before tomorrow` is not `tomorrow`).
#[rustfmt::skip]
pub(crate) fn is_time_modifier(lower: &str) -> bool {
    matches!(
        lower,
        "after" | "ago" | "antes" | "before" | "beginning" | "between" | "by" | "dentro" | "desde"
            | "después" | "despues" | "during" | "early" | "end" | "entre" | "finales" | "fines"
            | "from" | "hace" | "hasta" | "hence" | "late" | "later" | "mediados" | "mid" | "middle"
            | "past" | "principios" | "prior" | "since" | "start" | "through" | "till" | "to" | "until"
            | "within"
    )
}

#[rustfmt::skip]
pub(crate) fn is_time_ctx(text: &str, s: usize, e: usize, trig: bool) -> bool {
    if trig {
        return true;
    }
    let Some(tok) = text.get(s..e) else { return false };
    let has_digit = tok.bytes().any(|b| b.is_ascii_digit());
    let is_currency_num = has_digit && text.as_bytes().get(s.wrapping_sub(1)) == Some(&b'$');
    if (has_digit && !is_currency_num) || tok.bytes().all(|b| matches!(b, b':' | b'/' | b'-' | b'.' | b',')) {
        return true;
    }
    with_lowercase(tok, |lower| {
        matches!(
            lower,
            "a" | "after" | "ago" | "al" | "an" | "and" | "at" | "before" | "beginning" | "between" | "by" | "con" | "cuarto" | "de" | "del"
                | "dentro" | "desde" | "during" | "early" | "el" | "en" | "end" | "entre" | "esta" | "este" | "for" | "from" | "hace" | "half"
                | "hasta" | "hence" | "in" | "la" | "las" | "late" | "later" | "los" | "media" | "menos" | "mid" | "middle" | "o" | "oclock"
                | "of" | "on" | "para" | "past" | "por" | "since" | "start" | "the" | "through" | "till" | "to" | "un" | "una" | "unas" | "unos"
                | "until" | "within" | "y"
        ) || is_grammar_trigger(lower)
    })
}
