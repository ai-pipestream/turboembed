//! The bundled English inventory: four classes of abbreviations, each
//! sorted by bytes and free of duplicates, with the context rule that
//! decides when its full stop ends no sentence, and the words that are
//! never abbreviations.

use super::{Abbreviation, Bundle, ContextRule};

use ContextRule::{Numeric as N, Prepositive as P, Trailing as T};

const fn a(text: &'static str, rule: ContextRule) -> Abbreviation {
    Abbreviation { text, rule }
}

/// Everyday abbreviations: Latin connectives, months, times and streets.
pub(super) static GENERAL: &[Abbreviation] = &[
    a("Apr.", N),
    a("Aug.", N),
    a("Ave.", T),
    a("Blvd.", T),
    a("Dec.", N),
    a("Feb.", N),
    a("Jan.", N),
    a("Mar.", N),
    a("Mt.", P),
    a("Nov.", N),
    a("Oct.", N),
    a("Rd.", T),
    a("Sep.", N),
    a("Sept.", N),
    a("a.m.", T),
    a("approx.", N),
    a("ca.", N),
    a("dept.", T),
    a("e.g.", P),
    a("esp.", P),
    a("est.", N),
    a("etc.", T),
    a("fig.", N),
    a("i.e.", P),
    a("misc.", T),
    a("p.m.", T),
    a("viz.", P),
    a("vs.", P),
];

/// Titles and honorifics, which precede a name; `Jr.` and `Sr.` follow
/// one and can end a sentence.
pub(super) static TITLES: &[Abbreviation] = &[
    a("Adm.", P),
    a("Capt.", P),
    a("Cmdr.", P),
    a("Col.", P),
    a("Dr.", P),
    a("Gen.", P),
    a("Gov.", P),
    a("Hon.", P),
    a("Jr.", T),
    a("Lt.", P),
    a("Maj.", P),
    a("Messrs.", P),
    a("Mr.", P),
    a("Mrs.", P),
    a("Ms.", P),
    a("Pres.", P),
    a("Prof.", P),
    a("Rep.", P),
    a("Rev.", P),
    a("Sen.", P),
    a("Sgt.", P),
    a("Sr.", T),
    a("St.", P),
];

/// Citations, degrees and numbered parts of a work.
pub(super) static ACADEMIC: &[Abbreviation] = &[
    a("B.A.", T),
    a("M.A.", T),
    a("Ph.D.", T),
    a("cf.", P),
    a("ch.", N),
    a("ed.", T),
    a("eds.", T),
    a("eq.", N),
    a("et al.", T),
    a("et seq.", T),
    a("ib.", T),
    a("ibid.", T),
    a("loc. cit.", T),
    a("no.", N),
    a("op. cit.", T),
    a("p.", N),
    a("pp.", N),
    a("trans.", T),
    a("vol.", N),
    a("vols.", N),
];

/// Company suffixes, case names and the numbered parts of statutes.
pub(super) static LEGAL_TECHNICAL: &[Abbreviation] = &[
    a("Art.", N),
    a("Bros.", T),
    a("Cl.", N),
    a("Co.", T),
    a("Corp.", T),
    a("Ex.", N),
    a("Fed. Reg.", N),
    a("Inc.", T),
    a("Ltd.", T),
    a("Mfg.", T),
    a("No.", N),
    a("Nos.", N),
    a("Pub. L.", N),
    a("Sec.", N),
    a("Stat.", N),
    a("U.S.", T),
    a("U.S.C.", T),
    a("para.", N),
    a("v.", P),
];

/// Words that are never abbreviations, in lower case, sorted: a full stop
/// after one is a sentence end as far as the inventory knows.
pub(super) static PENULTIMATES: &[&str] = &[
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "is", "it", "me", "my", "no", "of", "oh", "ok", "on",
    "or", "so", "to", "up", "us", "we", "yes",
];

pub(super) static ENGLISH: Bundle =
    Bundle { code: "en", classes: [GENERAL, TITLES, ACADEMIC, LEGAL_TECHNICAL], penultimates: PENULTIMATES };
