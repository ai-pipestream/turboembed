//! Which languages each script can be: the first filter, before any
//! n-gram is scored.
//!
//! A run of Greek letters is Greek whatever an n-gram model says about
//! its three-letter windows, and a run of Latin letters is never Russian.
//! So the detector splits the text into runs of one script and scores
//! each run only against the languages written in that script, as
//! lingua and whatlang do. A script with one such language (Greek,
//! Hebrew, Thai, Hangul, Georgian, Armenian and most Indic scripts for
//! the OpenNLP model) needs no scoring at all.
//!
//! The table below is the writing systems of the 103 languages of
//! OpenNLP's `langdetect-183.bin`, from what each language is written in,
//! not from the model's weights: the model gives small positive weights
//! to letters of scripts a language is never written in (Han and Hangul
//! letters for nearly every language), so the weights cannot say which
//! scripts a language uses.

use turbo_unicode::Script;

/// The script group a letter's script falls in: Hiragana, Katakana and
/// Bopomofo go with Han, since Japanese writes Han and kana in one run;
/// Common, Inherited and Unknown belong to no group (a combining mark
/// goes with the letter before it).
pub fn group(script: Script) -> Option<Script> {
    match script {
        Script::Common | Script::Inherited | Script::Unknown => None,
        Script::Hiragana | Script::Katakana | Script::Bopomofo => Some(Script::Han),
        other => Some(other),
    }
}

/// The script groups a language is written in, by its ISO 639-3 code,
/// for the languages of `langdetect-183.bin`; `None` for a code the table
/// does not know, which the detector then allows in every script.
///
/// Serbian is written in both Cyrillic and Latin, and has both. Chinese
/// (`cmn`) and Japanese are both in the Han group; kana tell them apart.
/// Korean is Hangul only: Han characters in Korean text are rare, and a
/// Han run inside Hangul text is scored as Chinese or Japanese.
pub fn written_in(code: &str) -> Option<&'static [Script]> {
    use Script::*;
    const LATIN: &[Script] = &[Latin];
    const CYRILLIC: &[Script] = &[Cyrillic];
    const ARABIC: &[Script] = &[Arabic];
    const DEVANAGARI: &[Script] = &[Devanagari];
    Some(match code {
        "afr" | "ast" | "aze" | "bos" | "bre" | "cat" | "ceb" | "ces" | "cym" | "dan" | "deu" | "ekk" | "eng"
        | "epo" | "est" | "eus" | "fao" | "fin" | "fra" | "fry" | "gle" | "glg" | "gsw" | "hrv" | "hun" | "ind"
        | "isl" | "ita" | "jav" | "lat" | "lav" | "lim" | "lit" | "ltz" | "lvs" | "min" | "mlt" | "mri" | "msa"
        | "nan" | "nds" | "nld" | "nno" | "nob" | "oci" | "plt" | "pol" | "por" | "ron" | "slk" | "slv" | "som"
        | "spa" | "sqi" | "sun" | "swa" | "swe" | "tgl" | "tur" | "uzb" | "vie" | "vol" | "war" | "zul" => LATIN,
        "bak" | "bel" | "bul" | "che" | "kaz" | "kir" | "mkd" | "mon" | "rus" | "tat" | "tgk" | "ukr" => CYRILLIC,
        "srp" => &[Cyrillic, Latin],
        "ara" | "fas" | "pes" | "pnb" | "pus" | "urd" => ARABIC,
        "hin" | "mar" | "nep" | "san" => DEVANAGARI,
        "ben" => &[Bengali],
        "ell" => &[Greek],
        "guj" => &[Gujarati],
        "heb" => &[Hebrew],
        "hye" => &[Armenian],
        "kan" => &[Kannada],
        "kat" => &[Georgian],
        "kor" => &[Hangul],
        "mal" => &[Malayalam],
        "pan" => &[Gurmukhi],
        "sin" => &[Sinhala],
        "tam" => &[Tamil],
        "tel" => &[Telugu],
        "tha" => &[Thai],
        "cmn" | "jpn" => &[Han],
        _ => return None,
    })
}

/// For one model, the languages (outcome indexes) each script group can
/// be. A language the table does not know is a candidate in every group.
/// A language left out by the caller (to restrict a detector to the
/// languages a collection can hold) is a candidate in none.
#[derive(Debug, Clone, Hash)]
pub struct Candidates {
    /// Each group the table names with its languages, in outcome order.
    by_group: Vec<(Script, Vec<u32>)>,
    /// The languages the table does not know.
    anywhere: Vec<u32>,
}

impl Candidates {
    pub(crate) fn config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// The candidates for a model's outcome `codes`, in outcome order,
    /// keeping the codes `keep` accepts.
    pub fn new<'a>(codes: impl Iterator<Item = &'a str>, keep: impl Fn(&str) -> bool) -> Candidates {
        let mut by_group: Vec<(Script, Vec<u32>)> = Vec::new();
        let mut anywhere = Vec::new();
        for (i, code) in codes.enumerate() {
            if !keep(code) {
                continue;
            }
            let Some(scripts) = written_in(code) else {
                anywhere.push(i as u32);
                continue;
            };
            for &s in scripts {
                match by_group.iter_mut().find(|(g, _)| *g == s) {
                    Some((_, list)) => list.push(i as u32),
                    None => by_group.push((s, vec![i as u32])),
                }
            }
        }
        for (_, list) in &mut by_group {
            list.extend_from_slice(&anywhere);
            list.sort_unstable();
        }
        Candidates { by_group, anywhere }
    }

    /// The languages a run of script group `group` can be, in outcome
    /// order: empty when the model has none written in it.
    pub fn of(&self, group: Script) -> &[u32] {
        self.by_group.iter().find(|(g, _)| *g == group).map_or(&self.anywhere, |(_, list)| list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kana_and_bopomofo_go_with_han_and_neutral_scripts_with_nothing() {
        assert_eq!(group(Script::Hiragana), Some(Script::Han));
        assert_eq!(group(Script::Katakana), Some(Script::Han));
        assert_eq!(group(Script::Bopomofo), Some(Script::Han));
        assert_eq!(group(Script::Latin), Some(Script::Latin));
        assert_eq!(group(Script::Common), None);
        assert_eq!(group(Script::Inherited), None);
        assert_eq!(group(Script::Unknown), None);
    }

    #[test]
    fn candidates_follow_the_table_and_unknown_codes_go_everywhere() {
        let c = Candidates::new(["eng", "rus", "srp", "xyz", "ell"].into_iter(), |_| true);
        assert_eq!(c.of(Script::Latin), [0, 2, 3]);
        assert_eq!(c.of(Script::Cyrillic), [1, 2, 3]);
        assert_eq!(c.of(Script::Greek), [3, 4]);
        assert_eq!(c.of(Script::Thai), [3]);
        let c = Candidates::new(["eng"].into_iter(), |_| true);
        assert!(c.of(Script::Thai).is_empty());
        let c = Candidates::new(["eng", "rus", "srp", "xyz", "ell"].into_iter(), |c| c != "srp" && c != "xyz");
        assert_eq!(c.of(Script::Latin), [0]);
        assert_eq!(c.of(Script::Cyrillic), [1]);
        assert!(c.of(Script::Thai).is_empty());
    }

    #[test]
    fn every_group_in_the_table_is_a_group() {
        for code in ["eng", "srp", "cmn", "jpn", "kor", "ara", "hin", "tha", "ell", "heb"] {
            for &s in written_in(code).unwrap() {
                assert_eq!(group(s), Some(s), "{code} {s:?}");
            }
        }
    }
}
