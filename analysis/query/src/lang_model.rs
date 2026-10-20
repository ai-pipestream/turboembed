//! Compact, self-contained in-memory [`SpanDetector`] for short queries and
//! mixed-language clauses, covering Latin languages (`deu`, `eng`, `fra`,
//! `ita`, `nld`, `por`, `spa`) and script-distinct languages (`ara`, `ell`,
//! `heb`, `jpn`, `kor`, `rus`, `zho`).

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tag, Tokens};
use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_langspan::decode::SwitchCost;
use turbo_langspan::{Settings, SpanDetector};
use turbo_maxent::{ModelFile, crc32};
use turbo_unicode::Script;

/// Outcomes of the built-in query language model, in alphabetical order.
const OUTCOMES: &[&str] =
    &["ara", "deu", "ell", "eng", "fra", "heb", "ita", "jpn", "kor", "nld", "por", "rus", "spa", "zho"];

const IDX_DEU: u32 = 1;
const IDX_ENG: u32 = 3;
const IDX_FRA: u32 = 4;
const IDX_ITA: u32 = 6;
const IDX_NLD: u32 = 9;
const IDX_POR: u32 = 10;
const IDX_SPA: u32 = 12;

#[rustfmt::skip]
const PREDICATES: &[(&str, &[(u32, f64)])] = &[
    // English trigrams & short markers
    ("the", &[(IDX_ENG, 3.0)]), ("and", &[(IDX_ENG, 2.5)]), ("for", &[(IDX_ENG, 2.2)]),
    ("fli", &[(IDX_ENG, 3.0)]), ("lig", &[(IDX_ENG, 2.5)]), ("igh", &[(IDX_ENG, 3.2)]),
    ("ght", &[(IDX_ENG, 3.2)]), ("hts", &[(IDX_ENG, 2.8)]), ("che", &[(IDX_ENG, 2.4)]),
    ("hea", &[(IDX_ENG, 2.5)]), ("eap", &[(IDX_ENG, 3.0)]), ("tom", &[(IDX_ENG, 2.5)]),
    ("omo", &[(IDX_ENG, 2.5)]), ("orr", &[(IDX_ENG, 2.8)]), ("row", &[(IDX_ENG, 3.0)]),
    ("yes", &[(IDX_ENG, 2.8)]), ("day", &[(IDX_ENG, 2.8)]), ("wee", &[(IDX_ENG, 2.8)]),
    ("eek", &[(IDX_ENG, 2.8)]), ("und", &[(IDX_ENG, 2.0), (IDX_DEU, 2.5)]),
    ("der", &[(IDX_ENG, 1.8), (IDX_DEU, 2.4)]), ("wit", &[(IDX_ENG, 2.6)]),
    ("ith", &[(IDX_ENG, 2.8)]), ("fro", &[(IDX_ENG, 2.6)]), ("rom", &[(IDX_ENG, 2.4)]),
    ("nex", &[(IDX_ENG, 2.8)]), ("ext", &[(IDX_ENG, 2.6)]), (" to", &[(IDX_ENG, 2.2)]),
    ("to ", &[(IDX_ENG, 2.2)]), (" in", &[(IDX_ENG, 1.5)]), ("cor", &[(IDX_ENG, 2.5)]),
    ("dis", &[(IDX_ENG, 2.6)]), ("isc", &[(IDX_ENG, 2.6)]), ("cou", &[(IDX_ENG, 2.6)]),
    ("unt", &[(IDX_ENG, 2.5)]), ("dir", &[(IDX_ENG, 2.4)]), ("ect", &[(IDX_ENG, 2.4)]),
    ("bag", &[(IDX_ENG, 2.4)]), ("all", &[(IDX_ENG, 2.4)]), ("low", &[(IDX_ENG, 2.4)]),
    ("ass", &[(IDX_ENG, 2.2)]), ("sen", &[(IDX_ENG, 2.2)]), ("eng", &[(IDX_ENG, 2.4)]),
    // Spanish trigrams & diacritic n-grams
    ("vue", &[(IDX_SPA, 3.5)]), ("uel", &[(IDX_SPA, 3.5)]), ("elo", &[(IDX_SPA, 3.2)]),
    ("los", &[(IDX_SPA, 3.2)]), ("bar", &[(IDX_SPA, 2.5)]), ("ara", &[(IDX_SPA, 2.8)]),
    ("rat", &[(IDX_SPA, 2.5)]), ("ato", &[(IDX_SPA, 3.0)]), ("tos", &[(IDX_SPA, 3.2)]),
    ("mañ", &[(IDX_SPA, 4.0)]), ("aña", &[(IDX_SPA, 4.0)]), ("ñan", &[(IDX_SPA, 4.0)]),
    ("ana", &[(IDX_SPA, 2.5)]), ("ñ", &[(IDX_SPA, 4.0)]), ("par", &[(IDX_SPA, 2.4)]),
    ("por", &[(IDX_SPA, 2.4), (IDX_POR, 2.4)]), ("con", &[(IDX_SPA, 2.2), (IDX_ITA, 2.0)]),
    ("las", &[(IDX_SPA, 3.0)]), ("del", &[(IDX_SPA, 2.8), (IDX_ITA, 2.2)]),
    ("una", &[(IDX_SPA, 2.8), (IDX_ITA, 2.2)]), ("hoy", &[(IDX_SPA, 3.5)]),
    ("aye", &[(IDX_SPA, 3.2)]), ("yer", &[(IDX_SPA, 3.0)]), ("sem", &[(IDX_SPA, 2.8)]),
    ("ema", &[(IDX_SPA, 2.6)]), ("tar", &[(IDX_SPA, 2.4)]), ("ard", &[(IDX_SPA, 2.4)]),
    ("rde", &[(IDX_SPA, 2.8)]),
    // French trigrams
    ("les", &[(IDX_FRA, 3.2)]), ("des", &[(IDX_FRA, 3.0)]), ("pou", &[(IDX_FRA, 3.0)]),
    ("our", &[(IDX_FRA, 2.4)]), ("dem", &[(IDX_FRA, 3.0)]), ("mai", &[(IDX_FRA, 2.8)]),
    ("ain", &[(IDX_FRA, 2.8)]), ("vol", &[(IDX_FRA, 2.8)]), ("ols", &[(IDX_FRA, 2.8)]),
    ("pas", &[(IDX_FRA, 2.5)]),
    // German trigrams
    ("flü", &[(IDX_DEU, 4.0)]), ("lüg", &[(IDX_DEU, 4.0)]), ("üge", &[(IDX_DEU, 4.0)]),
    ("nac", &[(IDX_DEU, 3.2)]), ("ach", &[(IDX_DEU, 3.2)]),
    ("mor", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]), ("org", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]),
    ("rge", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]), ("gen", &[(IDX_DEU, 2.8), (IDX_NLD, 2.6)]),
    ("bil", &[(IDX_DEU, 2.6)]), ("lli", &[(IDX_DEU, 2.6)]), ("ige", &[(IDX_DEU, 2.6)]),
    // Dutch trigrams
    ("vlu", &[(IDX_NLD, 3.5)]), ("uch", &[(IDX_NLD, 3.2)]), ("cht", &[(IDX_NLD, 3.0)]),
    ("ten", &[(IDX_NLD, 2.6)]), ("naa", &[(IDX_NLD, 3.5)]), ("aar", &[(IDX_NLD, 3.2)]),
    ("van", &[(IDX_NLD, 3.0)]), ("het", &[(IDX_NLD, 3.2)]),
];

/// Builds a compact, self-contained [`SpanDetector`] tuned for short queries
/// and mixed-language clauses without requiring external model files.
pub fn default_span_detector() -> SpanDetector {
    let encoded: Vec<GisPredicate> =
        PREDICATES.iter().map(|(name, params)| (name.encode_utf16().collect(), params.to_vec())).collect();
    let classifier = build_gis_bytes(OUTCOMES, &encoded);
    let manifest = "Language=und\nfactory=opennlp.tools.langdetect.LanguageDetectorFactory\n\
Component-Name=LanguageDetectorME\nOpenNLP-Version=3.0.0-SNAPSHOT\nManifest-Version=1.0\n";
    let archive = build_zip(&[("manifest.properties", manifest.as_bytes()), ("langdetect.model", &classifier)]);
    let model = LanguageModel::open(ModelFile::open(archive).expect("valid in-memory GIS zip")).expect("valid model");
    let detector = LanguageDetector::new(model);
    let settings = Settings {
        switch: SwitchCost { space: 0.15, punctuation: 0.08, sentence: 0.04 },
        clause_letters: 18,
        rescore_margin: 0.0,
        shortlist: true,
    };
    SpanDetector::new(detector, settings)
}

type GisPredicate = (Vec<u16>, Vec<(u32, f64)>);

fn build_gis_bytes(outcomes: &[&str], predicates: &[GisPredicate]) -> Vec<u8> {
    let mut out = Vec::new();
    let write_utf = |buf: &mut Vec<u8>, units: &[u16]| {
        let mut mod_utf8 = Vec::with_capacity(units.len() * 2);
        for &u in units {
            turbo_unicode::modified_utf8::push_modified_utf8(&mut mod_utf8, u32::from(u));
        }
        buf.extend_from_slice(&(mod_utf8.len() as u16).to_be_bytes());
        buf.extend_from_slice(&mod_utf8);
    };
    let write_str = |buf: &mut Vec<u8>, s: &str| {
        let units: Vec<u16> = s.encode_utf16().collect();
        write_utf(buf, &units);
    };
    write_str(&mut out, "GIS");
    out.extend_from_slice(&1i32.to_be_bytes());
    out.extend_from_slice(&1.0f64.to_bits().to_be_bytes());
    out.extend_from_slice(&(outcomes.len() as i32).to_be_bytes());
    for o in outcomes {
        write_str(&mut out, o);
    }
    out.extend_from_slice(&(predicates.len() as i32).to_be_bytes());
    for (_, params) in predicates {
        let pat: Vec<String> =
            std::iter::once("1".to_owned()).chain(params.iter().map(|(o, _)| o.to_string())).collect();
        write_str(&mut out, &pat.join(" "));
    }
    out.extend_from_slice(&(predicates.len() as i32).to_be_bytes());
    for (name, _) in predicates {
        write_utf(&mut out, name);
    }
    for (_, params) in predicates {
        for &(_, w) in params {
            out.extend_from_slice(&w.to_bits().to_be_bytes());
        }
    }
    out
}

fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut z = Vec::new();
    let mut central = Vec::new();
    for &(name, data) in entries {
        let offset = z.len() as u32;
        let head = |sig: u32, out: &mut Vec<u8>, is_central: bool| {
            out.extend_from_slice(&sig.to_le_bytes());
            if is_central {
                out.extend_from_slice(&20u16.to_le_bytes());
            }
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc32(data).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        };
        head(0x0403_4b50, &mut z, false);
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
        head(0x0201_4b50, &mut central, true);
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let dir = z.len() as u32;
    z.extend_from_slice(&central);
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    let n = entries.len() as u16;
    z.extend_from_slice(&[0, 0, 0, 0]);
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&(central.len() as u32).to_le_bytes());
    z.extend_from_slice(&dir.to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
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

/// Query-tuned wrapper around [`SpanDetector`] that fast-paths single-sentence
/// pure-ASCII English queries and delegates mixed-language, multi-sentence, or
/// non-ASCII queries to the underlying [`SpanDetector`].
pub struct QuerySpanDetector {
    inner: SpanDetector,
    eng_tag: Option<Tag>,
}

impl QuerySpanDetector {
    pub fn new(inner: SpanDetector, fast_ascii_english: bool) -> Self {
        let eng_tag = if fast_ascii_english { inner.languages().tag("eng") } else { None };
        Self { inner, eng_tag }
    }
}

impl Annotator for QuerySpanDetector {
    const ID: &'static str = SpanDetector::ID;
    const VERSION: &'static str = SpanDetector::VERSION;
    type Reads = (View, Option<Sentences>, Option<Tokens>);
    type Writes = (LanguageSpans, Language);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<(LanguageSpans, Language), Error> {
        let text = view.text();
        if let Some(eng_tag) = self.eng_tag
            && text.is_ascii()
            && sentences.is_none_or(|s| s.spans.view() == view.id() && s.spans.len() <= 1)
            && let Some(toks) = tokens
            && toks.spans.view() == view.id()
            && toks.spans.is_ordered()
        {
            let (starts, ends) = toks.spans.columns();
            let mut first_alpha: Option<u32> = None;
            let mut last_alpha: Option<u32> = None;
            let mut has_non_en = false;
            for (&s, &e) in starts.iter().zip(ends) {
                let tok = &text[s as usize..e as usize];
                if tok.bytes().any(|b| b.is_ascii_alphabetic()) {
                    first_alpha.get_or_insert(s);
                    last_alpha = Some(e);
                    if with_lowercase(tok, |lower| NON_EN_ASCII_WORDS.binary_search(&lower).is_ok()) {
                        has_non_en = true;
                        break;
                    }
                }
            }
            if !has_non_en {
                let mut layer = LanguageSpans::new(view.id(), self.inner.languages().clone());
                let mut code = Strs::default();
                let mut score = Vec::new();
                if let Some(start) = first_alpha {
                    let end = ends.last().copied().unwrap_or_else(|| last_alpha.unwrap_or(start));
                    layer.push(Span::new(start, end), eng_tag, Script::Latin, 1.0);
                    code.push("eng");
                    score.push(1.0);
                }
                return Ok((layer, Language { code, score }));
            }
        }
        match self.inner.annotate((view, sentences, tokens)) {
            Ok(res) => Ok(res),
            Err(Error::Skipped(_)) => Ok((
                LanguageSpans::new(view.id(), self.inner.languages().clone()),
                Language { code: Strs::default(), score: Vec::new() },
            )),
            Err(e) => Err(e),
        }
    }

    fn model(&self) -> Option<ModelRef> {
        self.inner.model()
    }

    fn config_hash(&self) -> u64 {
        self.inner.config_hash()
    }
}

#[rustfmt::skip]
pub(crate) const NON_EN_ASCII_WORDS: &[&str] = &[
    "ach", "al", "ayer", "baratos", "billig", "billige", "con", "de", "del", "demain", "der", "des", "du", "een",
    "el", "en", "esta", "este", "het", "hoy", "la", "las", "le", "les", "los", "lunes", "martes", "mes", "meses",
    "morgen", "naar", "nach", "para", "por", "semana", "tarde", "un", "una", "van", "vluchten", "vols", "vuelos",
];

/// Non-place query words that can collide with minor town names in a global gazetteer.
#[rustfmt::skip]
pub(crate) const NON_PLACE_QUERY_WORDS: &[&str] = &[
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
        return tok.len() >= 2 && (tok.bytes().any(|b| b == b'-') || matches!(tok.as_bytes()[0], b'Q' | b'q'));
    }
    (3..=9).contains(&tok.len()) && with_lowercase(tok, |l| {
        matches!(
            l,
            "ago" | "apr" | "april" | "aug" | "august" | "day" | "days" | "dec" | "december"
                | "feb" | "february" | "jan" | "january" | "jul" | "july" | "jun" | "june"
                | "mar" | "march" | "may" | "month" | "months" | "nov" | "november" | "oct"
                | "october" | "sep" | "september" | "week" | "weeks" | "year" | "years"
        )
    })
}

#[rustfmt::skip]
pub(crate) const EN_TIME_TRIGGERS: &[&str] = &[
    "afternoon", "afternoons", "am", "annual", "annually", "apr", "april", "aug", "august", "autumn", "biannual",
    "bimonthly", "biweekly", "breakfast", "christmas", "coming", "current", "daily", "dawn", "day", "days", "daytime",
    "dec", "decade", "decades", "december", "dinner", "dinnertime", "dusk", "easter", "eight", "eighteen", "eighties",
    "eighty", "eleven", "eve", "evening", "evenings", "every", "everyday", "fall", "feb", "february", "fifteen",
    "fifties", "fifty", "five", "following", "forties", "fortnight", "fortnightly", "fortnights", "forty", "four",
    "fourteen", "fri", "friday", "fridays", "half", "halloween", "hour", "hourly", "hours", "hr", "hrs", "hundred",
    "jan", "january", "jul", "july", "jun", "june", "juneteenth", "last", "lunch", "lunchtime", "mar", "march", "may",
    "mid", "midnight", "min", "mins", "minute", "minutes", "mon", "monday", "mondays", "month", "monthly", "months",
    "morning", "mornings", "next", "night", "nightly", "nights", "nighttime", "nine", "nineteen", "nineties", "ninety",
    "noon", "nov", "november", "now", "oct", "october", "one", "otd", "past", "pm", "previous", "prior", "quarter",
    "quarterly", "quarters", "sat", "saturday", "saturdays", "sec", "second", "seconds", "secs", "semiannual",
    "semiannually", "sep", "sept", "september", "seven", "seventeen", "seventies", "seventy", "six", "sixteen",
    "sixties", "sixty", "spring", "summer", "sun", "sunday", "sundays", "ten", "thanksgiving", "thirteen", "thirties",
    "thirty", "this", "three", "thu", "thur", "thurs", "thursday", "thursdays", "tmr", "tmrw", "today", "tomorrow",
    "tonight", "tonite", "tue", "tues", "tuesday", "tuesdays", "twelve", "twenties", "twenty", "two", "upcoming",
    "wed", "wednesday", "wednesdays", "week", "weekday", "weekdays", "weekend", "weekends", "weekly", "weeks",
    "winter", "wk", "wks", "xmas", "year", "yearly", "years", "yesterday", "yr", "yrs",
];

#[rustfmt::skip]
pub(crate) const ES_TIME_TRIGGERS: &[&str] = &[
    "abr", "abril", "actual", "ago", "agosto", "anoche", "anos", "anteayer", "anterior", "anteriores", "antier",
    "anual", "anuales", "anualmente", "ayer", "año", "años", "bimensual", "bimensuales", "bimensualmente", "calendario",
    "cada", "decada", "decadas", "decenio", "decenios", "dia", "diariamente", "dias", "dic", "diciembre", "dom",
    "domingo", "domingos", "día", "días", "ene", "enero", "escolar", "esta", "estas", "este", "estos", "feb",
    "febrero", "finde", "findes", "fiscal", "habil", "hora", "horas", "hoy", "hr", "hra", "hras", "hrs", "invierno",
    "jue", "jueves", "jul", "julio", "jun", "junio", "lun", "lunes", "mar", "martes", "marzo", "may", "mayo",
    "mañana", "mañanas", "mediodia", "mediodía", "mensual", "mensuales", "mensualmente", "mes", "meses", "mie", "mier",
    "miercoles", "min", "mins", "minuto", "minutos", "miércoles", "navidad", "noche", "nochebuena", "noches",
    "nochevieja", "nov", "noviembre", "oct", "octubre", "otoño", "pasada", "pasadas", "pasado", "pasados", "pascua",
    "primavera", "proxima", "proximas", "proximo", "proximos", "próxima", "próximas", "próximo", "próximos",
    "quincena", "quincenal", "quincenales", "quincenalmente", "quincenas", "sab", "sabado", "sabados", "seg",
    "segundo", "segundos", "segs", "semana", "semanal", "semanales", "semanalmente", "semanas", "semestral",
    "semestrales", "semestralmente", "semestre", "semestres", "sep", "sept", "septiembre", "set", "setiembre",
    "siguiente", "siguientes", "sábado", "sábados", "tarde", "tardes", "trimestral", "trimestrales", "trimestralmente",
    "trimestre", "trimestres", "ultima", "ultimas", "ultimo", "ultimos", "verano", "vie", "viernes", "última",
    "últimas", "último", "últimos",
];

#[rustfmt::skip]
pub(crate) const TIME_CONTEXT_WORDS: &[&str] = &[
    "a", "after", "ago", "al", "an", "and", "at", "before", "beginning", "between", "by", "con", "cuarto", "de", "del",
    "dentro", "desde", "during", "early", "el", "en", "end", "entre", "esta", "este", "for", "from", "hace", "half",
    "hasta", "hence", "in", "la", "las", "late", "later", "los", "media", "menos", "mid", "middle", "o", "oclock",
    "of", "on", "para", "past", "por", "since", "start", "the", "through", "till", "to", "un", "una", "unas", "unos",
    "until", "within", "y",
];

#[rustfmt::skip]
pub(crate) const COMMON_EN_TIME_PHRASES: &[&str] = &[
    "last friday", "last monday", "last month", "last night", "last quarter", "last saturday", "last sunday",
    "last thursday", "last tuesday", "last wednesday", "last week", "last weekend", "last year", "next friday",
    "next monday", "next month", "next quarter", "next saturday", "next sunday", "next thursday", "next tuesday",
    "next wednesday", "next week", "next weekend", "next year", "this afternoon", "this evening", "this friday",
    "this monday", "this month", "this morning", "this night", "this quarter", "this saturday", "this sunday",
    "this thursday", "this tuesday", "this wednesday", "this week", "this weekend", "this year", "today", "tomorrow",
    "tomorrow afternoon", "tomorrow evening", "tomorrow morning", "tomorrow night", "tonight", "yesterday",
];

#[rustfmt::skip]
pub(crate) const COMMON_ES_TIME_PHRASES: &[&str] = &[
    "anoche", "ayer", "el mes pasado", "esta mañana", "esta noche", "esta semana", "esta tarde", "este año",
    "este mes", "hoy", "la semana pasada", "mañana",
];

pub(crate) fn is_non_place_word(lower: &str) -> bool {
    NON_PLACE_QUERY_WORDS.binary_search(&lower).is_ok()
}

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
    with_lowercase(tok, |lower| TIME_CONTEXT_WORDS.binary_search(&lower).is_ok() || is_grammar_trigger(lower))
}
