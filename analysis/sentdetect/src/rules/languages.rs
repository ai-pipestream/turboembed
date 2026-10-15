//! The languages the rule splitter knows, each with its own rules or a
//! fallback chain to languages that have them.
//!
//! The table is written for this project. A language has its own rules
//! when there is data for it under `data/rules` (abbreviations, sentence
//! starters) or when its terminators differ from the Unicode defaults.
//! Every other language names the languages whose rules suit its text
//! best, in order: a language written in the same script and region whose
//! abbreviations appear in its texts (Russian for the Cyrillic-script
//! languages of Russia, French for the languages of francophone Africa),
//! or one it is a variety of (Swiss German, Western Armenian). A language
//! whose chain reaches no rules, or that names none, gets the root rules:
//! the UAX #29 sentence rules with the tailorings every language shares,
//! and no abbreviations.

use super::profile::{self, LanguageRules};

/// One language of the table.
#[derive(Debug)]
pub struct Language {
    /// The lower case BCP 47 tag, such as `de` or `cbk-zam`.
    pub code: &'static str,
    /// The English name.
    pub name: &'static str,
    /// The language's own rules, when it has any.
    pub rules: Option<&'static LanguageRules>,
    /// The languages to try, in order, when it has none.
    pub fallbacks: &'static [&'static str],
}

const fn own(code: &'static str, name: &'static str, rules: &'static LanguageRules) -> Language {
    Language { code, name, rules: Some(rules), fallbacks: &[] }
}

const fn falls(code: &'static str, name: &'static str, fallbacks: &'static [&'static str]) -> Language {
    Language { code, name, rules: None, fallbacks }
}

/// Every language, sorted by code.
pub static LANGUAGES: &[Language] = &[
    falls("ab", "Abkhaz", &["ru"]),
    falls("ace", "Acehnese", &[]),
    falls("acm", "Mesopotamian Arabic", &["ar"]),
    falls("ady", "Adyghe", &["ru"]),
    falls("aeb", "Tunisian Arabic", &["ar"]),
    falls("af", "Afrikaans", &[]),
    falls("ak", "Akan", &["en"]),
    falls("als", "Alemannic", &["gsw"]),
    falls("alt", "Southern Altai", &["ru"]),
    own("am", "Amharic", &profile::AM),
    falls("an", "Aragonese", &["es"]),
    falls("ang", "Old English", &["en"]),
    falls("apc", "Levantine Arabic", &["ar"]),
    own("ar", "Arabic", &profile::AR),
    falls("arc", "Aramaic", &[]),
    falls("arq", "Algerian Arabic", &["ar"]),
    falls("ary", "Moroccan Arabic", &["ar"]),
    falls("arz", "Egyptian Arabic", &["ar"]),
    falls("as", "Assamese", &[]),
    falls("ast", "Asturian", &["es"]),
    falls("av", "Avaric", &["ru"]),
    falls("ay", "Aymara", &["es"]),
    falls("az", "Azerbaijani", &[]),
    falls("azb", "South Azerbaijani", &[]),
    falls("ba", "Bashkir", &["ru"]),
    falls("ban", "Balinese", &[]),
    falls("bar", "Bavarian", &["de"]),
    falls("bcl", "Central Bikol", &["en"]),
    falls("be", "Belarusian", &["ru"]),
    own("bg", "Bulgarian", &profile::BG),
    falls("bho", "Bhojpuri", &[]),
    falls("bi", "Bislama", &["en"]),
    falls("bjn", "Banjar", &[]),
    falls("blk", "Pa'O", &["my"]),
    falls("bm", "Bambara", &["fr"]),
    falls("bn", "Bengali", &[]),
    falls("bo", "Tibetan", &[]),
    falls("bpy", "Bishnupriya", &[]),
    falls("br", "Breton", &["fr"]),
    falls("bs", "Bosnian", &[]),
    falls("bug", "Buginese", &[]),
    falls("bxr", "Buryat", &["ru"]),
    falls("ca", "Catalan", &["es", "fr"]),
    falls("cbk-zam", "Chavacano", &["es"]),
    falls("cdo", "Min Dong", &["zh"]),
    falls("ce", "Chechen", &["ru"]),
    falls("ceb", "Cebuano", &["en"]),
    falls("ch", "Chamorro", &["en"]),
    falls("chr", "Cherokee", &["en"]),
    falls("chy", "Cheyenne", &["en"]),
    falls("ckb", "Central Kurdish", &[]),
    falls("co", "Corsican", &["it", "fr"]),
    falls("cr", "Cree", &["en"]),
    falls("crh", "Crimean Tatar", &[]),
    falls("cs", "Czech", &[]),
    falls("csb", "Kashubian", &["pl"]),
    falls("cu", "Church Slavonic", &["ru"]),
    falls("cv", "Chuvash", &["ru"]),
    falls("cy", "Welsh", &["en"]),
    own("da", "Danish", &profile::DA),
    own("de", "German", &profile::DE),
    falls("din", "Dinka", &["en"]),
    falls("diq", "Zazaki", &[]),
    falls("dsb", "Lower Sorbian", &["de"]),
    falls("dty", "Doteli", &[]),
    falls("dv", "Dhivehi", &[]),
    falls("dz", "Dzongkha", &[]),
    falls("ee", "Ewe", &["en"]),
    own("el", "Greek", &profile::EL),
    falls("eml", "Emilian-Romagnol", &["it"]),
    own("en", "English", &profile::EN),
    falls("eo", "Esperanto", &[]),
    own("es", "Spanish", &profile::ES),
    falls("et", "Estonian", &[]),
    falls("eu", "Basque", &["es", "fr"]),
    falls("ext", "Extremaduran", &["es"]),
    falls("fa", "Persian", &[]),
    falls("ff", "Fula", &["fr"]),
    falls("fi", "Finnish", &[]),
    falls("fil", "Filipino", &["en"]),
    falls("fj", "Fijian", &["en"]),
    falls("fo", "Faroese", &["da"]),
    own("fr", "French", &profile::FR),
    falls("frp", "Arpitan", &["fr"]),
    falls("frr", "North Frisian", &["de"]),
    falls("fur", "Friulian", &["it"]),
    falls("fy", "West Frisian", &[]),
    falls("ga", "Irish", &["en"]),
    falls("gag", "Gagauz", &[]),
    falls("gan", "Gan Chinese", &["zh"]),
    falls("gd", "Scottish Gaelic", &["en"]),
    falls("gl", "Galician", &["pt", "es"]),
    falls("glk", "Gilaki", &[]),
    falls("gn", "Guarani", &["es"]),
    falls("gom", "Goan Konkani", &[]),
    falls("gor", "Gorontalo", &[]),
    falls("got", "Gothic", &[]),
    falls("gsw", "Swiss German", &["de"]),
    falls("gu", "Gujarati", &[]),
    falls("gv", "Manx", &["en"]),
    falls("ha", "Hausa", &["en"]),
    falls("hak", "Hakka Chinese", &["zh"]),
    falls("haw", "Hawaiian", &["en"]),
    falls("he", "Hebrew", &[]),
    falls("hi", "Hindi", &[]),
    falls("hif", "Fiji Hindi", &["en"]),
    falls("hr", "Croatian", &[]),
    falls("hsb", "Upper Sorbian", &["de"]),
    falls("ht", "Haitian Creole", &["fr"]),
    falls("hu", "Hungarian", &[]),
    own("hy", "Armenian", &profile::HY),
    falls("hyw", "Western Armenian", &["hy"]),
    falls("ia", "Interlingua", &[]),
    falls("id", "Indonesian", &[]),
    falls("ie", "Interlingue", &[]),
    falls("ig", "Igbo", &["en"]),
    falls("ik", "Inupiaq", &["en"]),
    falls("ilo", "Ilocano", &["en"]),
    falls("inh", "Ingush", &["ru"]),
    falls("io", "Ido", &[]),
    falls("is", "Icelandic", &[]),
    own("it", "Italian", &profile::IT),
    falls("iu", "Inuktitut", &["en"]),
    falls("ja", "Japanese", &[]),
    falls("jam", "Jamaican Patois", &["en"]),
    falls("jbo", "Lojban", &[]),
    falls("jv", "Javanese", &[]),
    falls("ka", "Georgian", &[]),
    falls("kaa", "Karakalpak", &["kk"]),
    falls("kab", "Kabyle", &["fr"]),
    falls("kbd", "Kabardian", &["ru"]),
    falls("kbp", "Kabiye", &["fr"]),
    falls("kg", "Kongo", &["fr"]),
    falls("ki", "Kikuyu", &["en"]),
    own("kk", "Kazakh", &profile::KK),
    falls("kl", "Greenlandic", &["da"]),
    falls("km", "Khmer", &[]),
    falls("kn", "Kannada", &[]),
    falls("ko", "Korean", &[]),
    falls("koi", "Komi-Permyak", &["ru"]),
    falls("krc", "Karachay-Balkar", &["ru"]),
    falls("ks", "Kashmiri", &[]),
    falls("ksh", "Colognian", &["de"]),
    falls("ku", "Kurdish", &[]),
    falls("kv", "Komi", &["ru"]),
    falls("kw", "Cornish", &["en"]),
    falls("ky", "Kyrgyz", &["ru"]),
    falls("la", "Latin", &[]),
    falls("lad", "Ladino", &["es"]),
    falls("lb", "Luxembourgish", &["de", "fr"]),
    falls("lbe", "Lak", &["ru"]),
    falls("lez", "Lezgian", &["ru"]),
    falls("lfn", "Lingua Franca Nova", &[]),
    falls("lg", "Ganda", &["en"]),
    falls("li", "Limburgish", &[]),
    falls("lij", "Ligurian", &["it"]),
    falls("lld", "Ladin", &["it"]),
    falls("lmo", "Lombard", &["it"]),
    falls("ln", "Lingala", &["fr"]),
    falls("lo", "Lao", &[]),
    falls("lt", "Lithuanian", &[]),
    falls("ltg", "Latgalian", &[]),
    falls("lv", "Latvian", &[]),
    falls("lzh", "Classical Chinese", &["zh"]),
    falls("mad", "Madurese", &[]),
    falls("mai", "Maithili", &[]),
    falls("mdf", "Moksha", &["ru"]),
    falls("mg", "Malagasy", &["fr"]),
    falls("mhr", "Eastern Mari", &["ru"]),
    falls("mi", "Maori", &["en"]),
    falls("min", "Minangkabau", &[]),
    falls("mk", "Macedonian", &["bg"]),
    falls("ml", "Malayalam", &[]),
    falls("mn", "Mongolian", &["ru"]),
    falls("mni", "Manipuri", &[]),
    falls("mnw", "Mon", &["my"]),
    falls("mr", "Marathi", &[]),
    falls("mrj", "Western Mari", &["ru"]),
    falls("ms", "Malay", &[]),
    falls("mt", "Maltese", &["en", "it"]),
    falls("mwl", "Mirandese", &["pt"]),
    own("my", "Burmese", &profile::MY),
    falls("myv", "Erzya", &["ru"]),
    falls("mzn", "Mazanderani", &[]),
    falls("na", "Nauruan", &["en"]),
    falls("nah", "Nahuatl", &["es"]),
    falls("nan", "Min Nan", &["zh"]),
    falls("nap", "Neapolitan", &["it"]),
    falls("nb", "Norwegian Bokmal", &["da"]),
    falls("nds", "Low German", &["de"]),
    falls("ne", "Nepali", &[]),
    falls("new", "Newar", &[]),
    falls("nia", "Nias", &[]),
    falls("nl", "Dutch", &[]),
    falls("nn", "Norwegian Nynorsk", &["da"]),
    falls("no", "Norwegian", &["nb"]),
    falls("nov", "Novial", &[]),
    falls("nqo", "N'Ko", &[]),
    falls("nrm", "Norman", &["fr"]),
    falls("nso", "Northern Sotho", &["en"]),
    falls("nv", "Navajo", &["en"]),
    falls("ny", "Chichewa", &["en"]),
    falls("oc", "Occitan", &["fr"]),
    falls("olo", "Livvi-Karelian", &["ru"]),
    falls("om", "Oromo", &["en"]),
    falls("or", "Odia", &[]),
    falls("os", "Ossetian", &["ru"]),
    falls("pa", "Punjabi", &[]),
    falls("pag", "Pangasinan", &["en"]),
    falls("pam", "Kapampangan", &["en"]),
    falls("pap", "Papiamento", &[]),
    falls("pcd", "Picard", &["fr"]),
    falls("pcm", "Nigerian Pidgin", &["en"]),
    falls("pdc", "Pennsylvania German", &["de"]),
    falls("pfl", "Palatine German", &["de"]),
    falls("pi", "Pali", &[]),
    falls("pih", "Norfuk", &["en"]),
    own("pl", "Polish", &profile::PL),
    falls("pms", "Piedmontese", &["it"]),
    falls("pnb", "Western Punjabi", &[]),
    falls("pnt", "Pontic Greek", &["el"]),
    falls("ps", "Pashto", &[]),
    own("pt", "Portuguese", &profile::PT),
    falls("qu", "Quechua", &["es"]),
    falls("rm", "Romansh", &["de", "it"]),
    falls("rmy", "Romani", &[]),
    falls("rn", "Kirundi", &["fr"]),
    falls("ro", "Romanian", &[]),
    own("ru", "Russian", &profile::RU),
    falls("rue", "Rusyn", &["uk", "ru"]),
    falls("rw", "Kinyarwanda", &["en", "fr"]),
    falls("sa", "Sanskrit", &[]),
    falls("sah", "Yakut", &["ru"]),
    falls("sat", "Santali", &[]),
    falls("sc", "Sardinian", &["it"]),
    falls("scn", "Sicilian", &["it"]),
    falls("sco", "Scots", &["en"]),
    falls("sd", "Sindhi", &[]),
    falls("se", "Northern Sami", &[]),
    falls("sg", "Sango", &["fr"]),
    falls("sh", "Serbo-Croatian", &[]),
    falls("shn", "Shan", &["my"]),
    falls("si", "Sinhala", &[]),
    falls("sk", "Slovak", &[]),
    falls("skr", "Saraiki", &[]),
    falls("sl", "Slovenian", &[]),
    falls("sm", "Samoan", &["en"]),
    falls("smn", "Inari Sami", &[]),
    falls("sn", "Shona", &["en"]),
    falls("so", "Somali", &["en"]),
    falls("sq", "Albanian", &[]),
    falls("sr", "Serbian", &[]),
    falls("srn", "Sranan Tongo", &[]),
    falls("ss", "Swati", &["en"]),
    falls("st", "Sesotho", &["en"]),
    falls("stq", "Saterland Frisian", &["de"]),
    falls("su", "Sundanese", &[]),
    falls("sv", "Swedish", &[]),
    falls("sw", "Swahili", &["en"]),
    falls("szl", "Silesian", &["pl"]),
    falls("ta", "Tamil", &[]),
    falls("tcy", "Tulu", &[]),
    falls("te", "Telugu", &[]),
    falls("tet", "Tetum", &["pt"]),
    falls("tg", "Tajik", &["ru"]),
    falls("th", "Thai", &[]),
    falls("ti", "Tigrinya", &["am"]),
    falls("tk", "Turkmen", &[]),
    falls("tl", "Tagalog", &["en"]),
    falls("tn", "Tswana", &["en"]),
    falls("to", "Tongan", &["en"]),
    falls("tpi", "Tok Pisin", &["en"]),
    falls("tr", "Turkish", &[]),
    falls("ts", "Tsonga", &["en"]),
    falls("tt", "Tatar", &["ru"]),
    falls("tum", "Tumbuka", &["en"]),
    falls("tw", "Twi", &["ak"]),
    falls("ty", "Tahitian", &["fr"]),
    falls("tyv", "Tuvan", &["ru"]),
    falls("udm", "Udmurt", &["ru"]),
    falls("ug", "Uyghur", &[]),
    own("uk", "Ukrainian", &profile::UK),
    falls("ur", "Urdu", &[]),
    falls("uz", "Uzbek", &[]),
    falls("ve", "Venda", &["en"]),
    falls("vec", "Venetian", &["it"]),
    falls("vep", "Veps", &["ru"]),
    falls("vi", "Vietnamese", &[]),
    falls("vls", "West Flemish", &[]),
    falls("vo", "Volapuk", &[]),
    falls("wa", "Walloon", &["fr"]),
    falls("war", "Waray", &["en"]),
    falls("wo", "Wolof", &["fr"]),
    falls("wuu", "Wu Chinese", &["zh"]),
    falls("xal", "Kalmyk", &["ru"]),
    falls("xh", "Xhosa", &["en"]),
    falls("xmf", "Mingrelian", &[]),
    falls("yi", "Yiddish", &[]),
    falls("yo", "Yoruba", &["en"]),
    falls("yue", "Cantonese", &["zh"]),
    falls("za", "Zhuang", &[]),
    falls("zea", "Zeelandic", &[]),
    falls("zh", "Chinese", &[]),
    falls("zu", "Zulu", &["en"]),
];

/// The language with exactly this code, which must already be lower case.
pub fn language(code: &str) -> Option<&'static Language> {
    LANGUAGES.binary_search_by(|l| l.code.cmp(code)).ok().map(|i| &LANGUAGES[i])
}

/// How a language tag was resolved to rules.
#[derive(Copy, Clone, Debug)]
pub struct Resolved {
    /// The language of the table the tag named, after subtags that name no
    /// entry were dropped (`de-AT` names `de`); `None` for a tag the table
    /// does not know.
    pub language: Option<&'static Language>,
    /// The language whose rules apply: `language` itself, or the first
    /// language on its fallback chain that has rules; `None` for the root
    /// rules.
    pub rules_of: Option<&'static Language>,
}

impl Resolved {
    /// The rules that apply.
    pub fn rules(&self) -> &'static LanguageRules {
        self.rules_of.and_then(|l| l.rules).unwrap_or(&profile::ROOT)
    }

    /// The code of the language whose rules apply, `und` for the root.
    pub fn rules_code(&self) -> &'static str {
        self.rules_of.map_or("und", |l| l.code)
    }
}

/// The longest tag a lookup reads; longer ones are cut there.
const MAX_TAG: usize = 32;

/// Resolves a BCP 47 language tag, in any case and with `-` or `_`
/// between subtags: the longest prefix of whole subtags that the table
/// lists, then that language's own rules or the first rules its fallback
/// chain reaches, depth first in the order listed.
pub fn resolve(tag: &str) -> Resolved {
    let mut buf = [0u8; MAX_TAG];
    let mut len = 0;
    for b in tag.bytes().take(MAX_TAG) {
        buf[len] = if b == b'_' { b'-' } else { b.to_ascii_lowercase() };
        len += 1;
    }
    let mut lower = &buf[..len];
    let language = loop {
        let Ok(code) = std::str::from_utf8(lower) else { break None };
        if let Some(l) = language(code) {
            break Some(l);
        }
        match lower.iter().rposition(|&b| b == b'-') {
            Some(i) => lower = &lower[..i],
            None => break None,
        }
    };
    let rules_of = language.and_then(|l| with_rules(l, 0));
    Resolved { language, rules_of }
}

/// The first language with rules on `l`'s chain, `l` included. The table
/// test checks that no chain is deeper than a few steps or loops; the
/// depth limit keeps a mistake in it from recursing without end.
fn with_rules(l: &'static Language, depth: usize) -> Option<&'static Language> {
    if l.rules.is_some() {
        return Some(l);
    }
    if depth > 8 {
        return None;
    }
    l.fallbacks.iter().filter_map(|&code| language(code)).find_map(|next| with_rules(next, depth + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted_and_unique() {
        assert!(LANGUAGES.windows(2).all(|w| w[0].code < w[1].code));
        assert!(LANGUAGES.len() >= 250, "{} languages", LANGUAGES.len());
        for l in LANGUAGES {
            assert!(l.code.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'), "{}", l.code);
            assert!(l.rules.is_none() || l.fallbacks.is_empty(), "{} has rules and fallbacks", l.code);
        }
    }

    #[test]
    fn every_fallback_is_listed_and_chains_end() {
        fn depth(l: &Language, seen: &mut Vec<&'static str>) -> usize {
            l.fallbacks
                .iter()
                .map(|&code| {
                    assert!(!seen.contains(&code), "a loop through {code}");
                    let next = language(code).unwrap_or_else(|| panic!("{} falls back to unknown {code}", l.code));
                    seen.push(code);
                    let d = 1 + depth(next, seen);
                    seen.pop();
                    d
                })
                .max()
                .unwrap_or(0)
        }
        for l in LANGUAGES {
            assert!(depth(l, &mut vec![l.code]) <= 3, "{}", l.code);
            for &f in l.fallbacks {
                assert_ne!(f, l.code);
            }
        }
    }

    #[test]
    fn tags_resolve() {
        let r = resolve("de-AT");
        assert_eq!((r.language.unwrap().code, r.rules_code()), ("de", "de"));
        assert_eq!(resolve("DE_ch").rules_code(), "de");
        assert_eq!(resolve("als").rules_code(), "de");
        assert_eq!(resolve("no").rules_code(), "da");
        assert_eq!(resolve("ca").rules_code(), "es");
        assert_eq!(resolve("tw").rules_code(), "en");
        assert_eq!(resolve("cbk-zam").language.unwrap().code, "cbk-zam");
        assert_eq!(resolve("cbk-zam-x-y").language.unwrap().code, "cbk-zam");
        let zh = resolve("zh-Hant-TW");
        assert_eq!((zh.language.unwrap().code, zh.rules_code()), ("zh", "und"));
        let unknown = resolve("qaa");
        assert!(unknown.language.is_none() && unknown.rules_of.is_none());
        assert!(std::ptr::eq(unknown.rules(), &profile::ROOT));
        assert!(resolve("").language.is_none());
        assert!(resolve("-").language.is_none());
        // A tag that is not ASCII names nothing, and a long one is cut.
        assert!(resolve("dé").language.is_none());
        assert_eq!(resolve(&format!("en-{}", "x".repeat(100))).rules_code(), "en");
    }

    #[test]
    fn languages_with_rules() {
        let with: Vec<&str> = LANGUAGES.iter().filter(|l| l.rules.is_some()).map(|l| l.code).collect();
        assert_eq!(
            with,
            ["am", "ar", "bg", "da", "de", "el", "en", "es", "fr", "hy", "it", "kk", "my", "pl", "pt", "ru", "uk"]
        );
        let reached = LANGUAGES.iter().filter(|l| resolve(l.code).rules_of.is_some()).count();
        assert_eq!(reached, 186, "languages whose chain reaches rules");
    }
}
