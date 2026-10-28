//! Static script metadata for the closed-set evaluation baseline.
//! No model bytes or language probabilities are read here. This deliberately
//! incomplete table is not a general language detector.

use turbo_unicode::Script;

/// Group kana and Bopomofo with Han; exclude neutral scripts.
pub fn group(script: Script) -> Option<Script> {
    match script {
        Script::Common | Script::Inherited | Script::Unknown => None,
        Script::Hiragana | Script::Katakana | Script::Bopomofo => Some(Script::Han),
        other => Some(other),
    }
}

/// Listed script groups by ISO 639-3 language code. Unknown codes return None.
/// This keeps the existing evaluation convention: Korean uses Hangul, while
/// Chinese and Japanese share the Han group.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_metadata_preserves_shared_scripts_and_unknowns() {
        assert_eq!(group(Script::Hiragana), Some(Script::Han));
        assert_eq!(group(Script::Katakana), Some(Script::Han));
        assert_eq!(group(Script::Inherited), None);
        assert_eq!(written_in("eng"), written_in("spa"));
        assert_eq!(written_in("cmn"), written_in("jpn"));
        assert_eq!(written_in("srp"), Some(&[Script::Cyrillic, Script::Latin][..]));
        assert_eq!(written_in("xyz"), None);
    }
}
