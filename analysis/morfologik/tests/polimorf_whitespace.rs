use turbo_morfologik::{DictionaryCharset, MorfologikError, PoliMorfDictionaryReader};

#[test]
fn unicode_blank_lines_are_skipped_before_dictionary_lookup() {
    for blank in ["\u{85}", "\u{a0}", "\u{202f}", "\u{3000}", "", " \t\u{85}\u{a0}\u{202f}"] {
        let input = format!("{blank}\r\npsa\tpies\tsubst:sg:gen:m2\r\n{blank}\n");
        let dictionary = PoliMorfDictionaryReader::read(input.as_bytes()).unwrap();
        assert_eq!(dictionary.len(), 1);
        assert_eq!(dictionary.lemmatize(&["Psa"], &["subst:sg:gen:m2"]), ["pies"]);
    }
    // Whitespace is tested after decoding the declared charset.
    let encoded = DictionaryCharset::Iso8859_1.encode("\u{85}\npsa\tpies\tsubst:sg:gen:m2\n").unwrap();
    let dictionary = PoliMorfDictionaryReader::read_with_charset(&encoded, DictionaryCharset::Iso8859_1).unwrap();
    assert_eq!(dictionary.lemmatize(&["psa"], &["subst:sg:gen:m2"]), ["pies"]);
}

#[test]
fn legacy_separators_are_malformed_under_the_unicode_default() {
    for c in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
        let input = format!("psa\tpies\tsubst:sg:gen:m2\n{c}\n");
        let error = PoliMorfDictionaryReader::read(input.as_bytes()).unwrap_err();
        assert!(matches!(error, MorfologikError::Format(_)));
        assert_eq!(error.message(), format!("PoliMorf line 2 has fewer than 3 tab-separated fields: {c}"));
    }
}
