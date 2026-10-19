use super::*;

/// A country word and a currency it issues: with `generic`, the currency
/// of that name (`Swiss` and `franc` give `CHF`); with none, a currency
/// a following word names already (`Japanese yen`).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Nation {
    pub word: &'static str,
    pub generic: Option<Generic>,
    pub code: &'static str,
    pub flags: u8,
}

pub(crate) const fn n(word: &'static str, generic: Generic, code: &'static str) -> Nation {
    Nation { word, generic: Some(generic), code, flags: 0 }
}

pub(crate) const fn named(word: &'static str, code: &'static str) -> Nation {
    Nation { word, generic: None, code, flags: 0 }
}

/// A short form that only goes before `$`: `C$`, `HK$`.
pub(crate) const fn dollar(word: &'static str, code: &'static str) -> Nation {
    Nation { word, generic: Some(Generic::Dollar), code, flags: ONLY_DOLLAR }
}

/// Country words, a word's rows together. A word with a space is read by
/// a rule, not the lexicon.
pub(crate) static NATIONS: &[Nation] = &[
    n("us", Generic::Dollar, "usd"),
    n("u.s.", Generic::Dollar, "usd"),
    n("united states", Generic::Dollar, "usd"),
    n("american", Generic::Dollar, "usd"),
    n("australian", Generic::Dollar, "aud"),
    n("australia", Generic::Dollar, "aud"),
    n("aussie", Generic::Dollar, "aud"),
    dollar("au", "aud"),
    n("canadian", Generic::Dollar, "cad"),
    n("canada", Generic::Dollar, "cad"),
    dollar("c", "cad"),
    dollar("can", "cad"),
    dollar("ca", "cad"),
    n("singapore", Generic::Dollar, "sgd"),
    n("singaporean", Generic::Dollar, "sgd"),
    dollar("s", "sgd"),
    dollar("sg", "sgd"),
    n("hk", Generic::Dollar, "hkd"),
    n("hong kong", Generic::Dollar, "hkd"),
    n("nz", Generic::Dollar, "nzd"),
    n("new zealand", Generic::Dollar, "nzd"),
    n("jamaican", Generic::Dollar, "jmd"),
    dollar("j", "jmd"),
    n("bahamian", Generic::Dollar, "bsd"),
    n("barbadian", Generic::Dollar, "bbd"),
    n("belize", Generic::Dollar, "bzd"),
    n("bermudian", Generic::Dollar, "bmd"),
    n("brunei", Generic::Dollar, "bnd"),
    n("fijian", Generic::Dollar, "fjd"),
    n("fiji", Generic::Dollar, "fjd"),
    n("guyanese", Generic::Dollar, "gyd"),
    n("liberian", Generic::Dollar, "lrd"),
    n("namibian", Generic::Dollar, "nad"),
    n("surinamese", Generic::Dollar, "srd"),
    n("taiwan", Generic::Dollar, "twd"),
    n("taiwanese", Generic::Dollar, "twd"),
    n("new taiwan", Generic::Dollar, "twd"),
    dollar("nt", "twd"),
    n("trinidadian", Generic::Dollar, "ttd"),
    n("trinidad and tobago", Generic::Dollar, "ttd"),
    n("cayman islands", Generic::Dollar, "kyd"),
    n("solomon islands", Generic::Dollar, "sbd"),
    n("mexican", Generic::Peso, "mxn"),
    n("mexico", Generic::Peso, "mxn"),
    dollar("mex", "mxn"),
    n("philippine", Generic::Peso, "php"),
    n("argentine", Generic::Peso, "ars"),
    n("argentinian", Generic::Peso, "ars"),
    n("chilean", Generic::Peso, "clp"),
    n("colombian", Generic::Peso, "cop"),
    n("cuban", Generic::Peso, "cup"),
    n("dominican", Generic::Peso, "dop"),
    n("uruguayan", Generic::Peso, "uyu"),
    dollar("r", "brl"),
    named("brazilian", "brl"),
    n("british", Generic::Pound, "gbp"),
    n("uk", Generic::Pound, "gbp"),
    n("egyptian", Generic::Pound, "egp"),
    n("lebanese", Generic::Pound, "lbp"),
    n("syrian", Generic::Pound, "syp"),
    n("sudanese", Generic::Pound, "sdg"),
    n("gibraltar", Generic::Pound, "gip"),
    n("swiss", Generic::Franc, "chf"),
    n("french", Generic::Franc, "frf"),
    n("congolese", Generic::Franc, "cdf"),
    n("rwandan", Generic::Franc, "rwf"),
    n("guinean", Generic::Franc, "gnf"),
    n("burundian", Generic::Franc, "bif"),
    n("djiboutian", Generic::Franc, "djf"),
    n("comorian", Generic::Franc, "kmf"),
    n("indian", Generic::Rupee, "inr"),
    n("pakistani", Generic::Rupee, "pkr"),
    n("nepalese", Generic::Rupee, "npr"),
    n("nepali", Generic::Rupee, "npr"),
    n("mauritian", Generic::Rupee, "mur"),
    n("seychellois", Generic::Rupee, "scr"),
    n("sri lankan", Generic::Rupee, "lkr"),
    n("russian", Generic::Ruble, "rub"),
    n("belarusian", Generic::Ruble, "byn"),
    n("danish", Generic::Krone, "dkk"),
    n("norwegian", Generic::Krone, "nok"),
    n("swedish", Generic::Krone, "sek"),
    n("icelandic", Generic::Krone, "isk"),
    n("kuwaiti", Generic::Dinar, "kwd"),
    n("iraqi", Generic::Dinar, "iqd"),
    n("jordanian", Generic::Dinar, "jod"),
    n("bahraini", Generic::Dinar, "bhd"),
    n("libyan", Generic::Dinar, "lyd"),
    n("tunisian", Generic::Dinar, "tnd"),
    n("algerian", Generic::Dinar, "dzd"),
    n("serbian", Generic::Dinar, "rsd"),
    n("saudi", Generic::Riyal, "sar"),
    n("qatari", Generic::Riyal, "qar"),
    n("omani", Generic::Riyal, "omr"),
    n("iranian", Generic::Riyal, "irr"),
    n("yemeni", Generic::Riyal, "yer"),
    n("kenyan", Generic::Shilling, "kes"),
    n("ugandan", Generic::Shilling, "ugx"),
    n("tanzanian", Generic::Shilling, "tzs"),
    n("somali", Generic::Shilling, "sos"),
    n("moroccan", Generic::Dirham, "mad"),
    n("emirati", Generic::Dirham, "aed"),
    n("uae", Generic::Dirham, "aed"),
    n("romanian", Generic::Leu, "ron"),
    n("moldovan", Generic::Leu, "mdl"),
    n("azerbaijani", Generic::Manat, "azn"),
    n("turkmen", Generic::Manat, "tmt"),
    n("turkish", Generic::Lira, "try"),
    n("italian", Generic::Lira, "itl"),
    n("german", Generic::Mark, "dem"),
    n("deutsche", Generic::Mark, "dem"),
    n("korean", Generic::Won, "krw"),
    n("south korean", Generic::Won, "krw"),
    n("north korean", Generic::Won, "kpw"),
    named("japanese", "jpy"),
    named("chinese", "cny"),
    named("dutch", "nlg"),
    named("finnish", "fim"),
    named("spanish", "esp"),
    named("thai", "thb"),
    named("polish", "pln"),
    named("hungarian", "huf"),
    named("israeli", "ils"),
    named("nigerian", "ngn"),
    named("indonesian", "idr"),
    named("malaysian", "myr"),
    named("ukrainian", "uah"),
    named("czech", "czk"),
    named("ghanaian", "ghs"),
    named("kazakh", "kzt"),
    named("kazakhstani", "kzt"),
    named("vietnamese", "vnd"),
    named("south african", "zar"),
];

/// The first row of `word` in [`NATIONS`].
pub(crate) fn nation(word: &str) -> Option<usize> {
    NATIONS.iter().position(|r| r.word == word)
}

/// The rows of the word whose first row is `first`.
pub(crate) fn nation_rows(first: usize) -> &'static [Nation] {
    let word = NATIONS[first].word;
    let n = NATIONS[first..].iter().take_while(|r| r.word == word).count();
    &NATIONS[first..first + n]
}
