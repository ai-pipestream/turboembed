//! Language code resolution for the bundled lists.
//!
//! OpenNLP accepts lower-case ISO 639-1, ISO 639-2/T, and the bibliographic
//! codes dut/fre/ger only. Upper case and tags with a region are refused.

use turbo_stopword::StopwordLanguage;

#[test]
fn supported_codes_resolve() {
    for language in StopwordLanguage::ALL {
        assert_eq!(StopwordLanguage::from_code(language.iso639_1()), Some(language));
        assert_eq!(StopwordLanguage::from_code(language.iso639_3()), Some(language));
    }
    assert_eq!(StopwordLanguage::from_code("ger"), Some(StopwordLanguage::German));
    assert_eq!(StopwordLanguage::from_code("dut"), Some(StopwordLanguage::Dutch));
    assert_eq!(StopwordLanguage::from_code("fre"), Some(StopwordLanguage::French));
}

#[test]
fn refused_codes_stay_none() {
    for refused in [
        "EN",
        "Eng",
        "en-US",
        "en_US",
        "",
        "sv",
        "und",
        "x-unspecified",
        "eng ",
        " eng",
        "deu-DE",
        "GER",
        "DUT",
        "FRE",
        "xx",
        "zz",
    ] {
        assert_eq!(StopwordLanguage::from_code(refused), None, "{refused:?}");
    }
}
