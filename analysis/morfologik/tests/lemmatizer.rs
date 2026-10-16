//! Integration tests for `turbo-morfologik` metadata, input conversions,
//! PoliMorf reader, lemmatizer, tag dictionary, builder, and annotators,
//! porting OpenNLP 3.x's `MorfologikInputConversionTest`,
//! `MorfologikMetadataTest`, `PoliMorfDictionaryReaderUsageExampleTest`,
//! `MorfologikLemmatizerTest`, `MorfologikTagDictionaryTest`,
//! `MorfologikLocaleTest`, and `MorfologikDictionaryBuilderTest`.

use std::sync::Arc;

use turbo_morfologik::{
    BaseFormEncoding, DictionaryCharset, MorfologikDictionary, MorfologikDictionaryBuilder, MorfologikDictionaryReader,
    MorfologikError, MorfologikLemmatizer, MorfologikMetadata, MorfologikTagDictionary, PoliMorfDictionaryReader,
};

/// Builds an FSA5 automaton accepting one non-empty byte sequence (`FsaTestData.singleSequence`).
fn single_fsa5_sequence(labels: &[u8]) -> Vec<u8> {
    assert!(!labels.is_empty());
    let header: [u8; 12] = [b'\\', b'f', b's', b'a', 0x05, b'_', b'+', 0x01, 0, 0x02, 0, 0x06];
    let mut out = Vec::with_capacity(header.len() + labels.len() * 2);
    out.extend_from_slice(&header);
    for (i, &label) in labels.iter().enumerate() {
        out.push(label);
        out.push(if i + 1 == labels.len() { 0x03 } else { 0x06 });
    }
    out
}

/// Builds a CFSA2 automaton accepting one non-empty byte sequence (`FsaTestData.singleCfsaSequence`).
fn single_cfsa2_sequence(labels: &[u8]) -> Vec<u8> {
    assert!(!labels.is_empty());
    let header: [u8; 10] = [b'\\', b'f', b's', b'a', 0xc6, 0, 7, 0, 0xc0, b'^'];
    let mut out = Vec::with_capacity(header.len() + labels.len() * 2 + 1);
    out.extend_from_slice(&header);
    for (i, &label) in labels.iter().enumerate() {
        out.push(if i + 1 == labels.len() { 0x60 } else { 0xc0 });
        out.push(label);
    }
    out.push(0);
    out
}

// ============================================================================
// MorfologikInputConversionTest & MorfologikMetadataTest
// ============================================================================

#[test]
fn input_conversions_and_metadata_validation() {
    let conversion_cases = [
        ("æ ae", "cæsar", "caesar"),
        ("æ ae", "CÆSAR", "caesar"),
        ("ĳ ij", "Ĳssel", "ijssel"),
        ("ﬃ ffi", "oﬃce", "office"),
        ("’ '", "can’t", "can't"),
        ("a aa", "a", "aa"),
        ("a aa", "aaa", "aaaaaa"),
        ("aa a", "aaa", "aa"),
        ("aa a", "aaaa", "aa"),
        ("ab b", "abab", "bb"),
        ("a b,b c", "a", "c"),
        ("b c,a b", "a", "b"),
        ("ab x,a y", "aba", "xy"),
        ("a b,b a", "a", "a"),
        ("a a", "aaa", "aaa"),
        ("🦊 fox", "🦊", "fox"),
        ("a 🦊", "a", "🦊"),
        ("c# csharp", "C#", "csharp"),
        ("x Y", "x", "y"),
        ("ος os", "ΟΣ", "os"),
        ("i\u{0307} dot", "İ", "dot"),
        ("æ ae", "plain", "plain"),
        ("a b,", "a", "b"),
        ("a b,,,", "a", "b"),
        ("a b,  ", "a", "b"),
        (" a b , c d ", "ac", "bd"),
        ("a b,\\tc d", "ac", "bd"),
        (",,,", "a", "a"),
    ];

    for (rules, token, form) in conversion_cases {
        let seq = format!("{form}+lemma#unchanged+tag").into_bytes();
        let info = format!(
            "fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=NONE\nfsa.dict.input-conversion={rules}\n"
        );
        for cfsa in [false, true] {
            let data = if cfsa { single_cfsa2_sequence(&seq) } else { single_fsa5_sequence(&seq) };
            let dict = MorfologikDictionaryReader::read(&data, info.as_bytes()).unwrap();
            assert_eq!(dict.lemmatize(&[token], &["tag"]), vec!["lemma#unchanged"]);
            assert_eq!(dict.lemmatize_all(&[token], &["tag"]), vec![vec!["lemma#unchanged".to_owned()]]);
            assert_eq!(dict.lemmatize(&[token], &["wrong-tag"]), vec!["O"]);
        }
    }

    let invalid_rules =
        ["", " ", "a", "a ", " a", "a  b", "a b c", "a\\tb", "a b,a c", "a b,a b", ",a b", "a b,,c d", "a b,\\u0000"];
    for rules in invalid_rules {
        let info = format!(
            "fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=NONE\nfsa.dict.input-conversion={rules}\n"
        );
        let err = MorfologikDictionaryReader::read(b"unread", info.as_bytes()).unwrap_err();
        assert!(matches!(err, MorfologikError::InvalidArgument(_)));
    }
}

#[test]
fn metadata_separators_charsets_and_unmappable_bytes() {
    let valid_separators = [
        ("UTF-8", "+", "+"),
        ("ISO-8859-1", "é", "é"),
        ("ISO-8859-1", "é", "\\u00e9"),
        ("windows-1252", "€", "€"),
        ("windows-1252", "€", "\\u20ac"),
        ("windows-1250", "Ł", "Ł"),
        ("windows-1251", "Я", "Я"),
        ("UTF-8", "\t", "\\t"),
        ("UTF-8", "\n", "\\n"),
        ("UTF-8", "\0", "\\u0000"),
        ("UTF-8", "\\", "\\\\"),
    ];
    for (charset_name, sep, prop) in valid_separators {
        let charset = DictionaryCharset::for_name(charset_name).unwrap();
        let seq = charset.encode(&format!("word{sep}lemma{sep}NN")).unwrap();
        let info = format!("fsa.dict.separator={prop}\nfsa.dict.encoding={charset_name}\nfsa.dict.encoder=NONE\n");
        let dict = MorfologikDictionaryReader::read(&single_fsa5_sequence(&seq), info.as_bytes()).unwrap();
        assert_eq!(dict.lemmatize(&["word"], &["NN"]), vec!["lemma"]);
    }

    let invalid_separators = [
        ("UTF-8", "\\u00e9"),
        ("UTF-8", "\\u20ac"),
        ("UTF-16", "+"),
        ("UTF-16LE", "+"),
        ("US-ASCII", "\\u20ac"),
        ("ISO-8859-1", "\\u20ac"),
        ("UTF-8", "\\uD800"),
        ("UTF-8", "\\uDC00"),
        ("UTF-8", "\\uD83D\\uDE00"),
        ("UTF-8", "++"),
        ("UTF-8", ""),
    ];
    for (charset_name, prop) in invalid_separators {
        let info = format!("fsa.dict.separator={prop}\nfsa.dict.encoding={charset_name}\nfsa.dict.encoder=NONE\n");
        let err = MorfologikDictionaryReader::read(b"unread", info.as_bytes()).unwrap_err();
        assert!(matches!(err, MorfologikError::InvalidArgument(_)));
    }

    // Encoder whitespace handling.
    for enc in ["NONE ", " NONE ", "\\tNoNe\\t", "\\fnone\\f", "\\nnone\\r", "\\u0000NONE\\u0000"] {
        let info = format!("fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder={enc}\n");
        let dict = MorfologikDictionaryReader::read(&single_fsa5_sequence(b"word+lemma+NN"), info.as_bytes()).unwrap();
        assert_eq!(dict.lemmatize(&["word"], &["NN"]), vec!["lemma"]);
    }
    for bad_enc in ["NO NE", "none\\u00a0", "\\u2003NONE", " NONE extra ", " ", "\\t\\n"] {
        let info = format!("fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder={bad_enc}\n");
        assert!(MorfologikMetadata::parse(info.as_bytes()).is_err());
    }

    // Unmappable byte 0x81 in windows-1252 is rejected by both dictionary readers.
    for field in [0, 1, 2] {
        let mut row = *b"x\ty\tz";
        row[field * 2] = 0x81;
        assert!(PoliMorfDictionaryReader::read_with_charset(&row, DictionaryCharset::Windows1252).is_err());
        row[1] = b'+';
        row[3] = b'+';
        assert!(
            MorfologikDictionaryReader::read_with_options(
                &single_cfsa2_sequence(&row),
                b'+',
                BaseFormEncoding::None,
                DictionaryCharset::Windows1252
            )
            .is_err()
        );
    }
}

// ============================================================================
// PoliMorfDictionaryReaderUsageExampleTest
// ============================================================================

#[test]
fn polimorf_dictionary_reader_matches_opennlp() {
    let rows = [
        "pies\tpies\tsubst:sg:nom:m2",
        "psa\tpies\tsubst:sg:gen:m2",
        "psy\tpies\tsubst:pl:nom:m2",
        "kota\tkot\tsubst:sg:gen:m2",
        "formy\tforma\tsubst:pl:nom:f",
        "formy\tform\tsubst:pl:nom:f",
    ];
    let text = format!("{}\n", rows.join("\n"));
    let lemmatizer = PoliMorfDictionaryReader::read(text.as_bytes()).unwrap();

    assert_eq!(lemmatizer.lemmatize(&["psa"], &["subst:sg:gen:m2"]), vec!["pies"]);
    assert_eq!(lemmatizer.lemmatize(&["Psa"], &["subst:sg:gen:m2"]), vec!["pies"]);
    assert_eq!(lemmatizer.lemmatize(&["kota"], &["subst:sg:gen:m2"]), vec!["kot"]);
    assert_eq!(lemmatizer.lemmatize(&["psa"], &["adj:sg:nom:m2:pos"]), vec!["O"]);
    assert_eq!(lemmatizer.lemmatize(&["kanapa"], &["subst:sg:nom:f"]), vec!["O"]);
    assert_eq!(
        lemmatizer.lemmatize_all(&["formy"], &["subst:pl:nom:f"]),
        vec![vec!["forma".to_owned(), "form".to_owned()]]
    );

    // Blank lines and lines consisting solely of U+00A0 / U+202F are skipped.
    let nbsp_text = "\u{00A0}\u{202F}\npsa\tpies\tsubst:sg:gen:m2\n   \n";
    let nbsp_lemmatizer = PoliMorfDictionaryReader::read(nbsp_text.as_bytes()).unwrap();
    assert_eq!(nbsp_lemmatizer.lemmatize(&["psa"], &["subst:sg:gen:m2"]), vec!["pies"]);

    // Fewer than three fields fails loudly.
    assert!(PoliMorfDictionaryReader::read(b"psa\tpies\n").is_err());

    // Explicit ISO-8859-1 charset.
    let iso_bytes = DictionaryCharset::Iso8859_1.encode("caf\u{00e9}\tcaf\u{00e9}\tsubst:sg:nom:m2\n").unwrap();
    let iso_lemmatizer = PoliMorfDictionaryReader::read_with_charset(&iso_bytes, DictionaryCharset::Iso8859_1).unwrap();
    assert_eq!(iso_lemmatizer.lemmatize(&["caf\u{00e9}"], &["subst:sg:nom:m2"]), vec!["caf\u{00e9}"]);
}

#[test]
fn utf16_charset_decode_and_polimorf_round_trip() {
    let text = "żółw\tżółw\tsubst:sg:nom:m2\n";
    for charset in [DictionaryCharset::Utf16, DictionaryCharset::Utf16Be, DictionaryCharset::Utf16Le] {
        let encoded = charset.encode(text).unwrap();
        assert_eq!(charset.decode(&encoded).unwrap(), text);
        let lemmatizer = PoliMorfDictionaryReader::read_with_charset(&encoded, charset).unwrap();
        assert_eq!(lemmatizer.lemmatize(&["żółw"], &["subst:sg:nom:m2"]), vec!["żółw"]);
    }

    // UTF-16LE BOM is respected by DictionaryCharset::Utf16.
    let mut utf16_le_with_bom = vec![0xFF, 0xFE];
    utf16_le_with_bom.extend_from_slice(&DictionaryCharset::Utf16Le.encode(text).unwrap());
    assert_eq!(DictionaryCharset::Utf16.decode(&utf16_le_with_bom).unwrap(), text);

    // Odd byte length and unpaired surrogates are rejected.
    assert!(DictionaryCharset::Utf16Be.decode(&[0x00]).is_err());
    assert!(DictionaryCharset::Utf16Be.decode(&[0xD8, 0x00]).is_err());
}

// ============================================================================
// MorfologikLemmatizerTest, MorfologikTagDictionaryTest, MorfologikLocaleTest,
// and MorfologikDictionaryBuilderTest
// ============================================================================

const DICT_WITH_LEMMA_TXT: &[u8] = b"carro,carro,NOUN\n\
casa,casa,NOUN\n\
Casa,Casa,PROP\n\
casa,casinha,NOUN\n\
casa,casona,NOUN\n\
casar,casa,V\n\
casar,casar,V-INF\n\
ir,foi,V\n\
menino,menina,NOUN\n\
menino,menininho,NOUN\n\
menino,menino,NOUN\n\
menino,menin\xc3\xa3o,NOUN\n\
ser,foi,V\n";

const DICT_WITH_LEMMA_INFO: &[u8] = b"fsa.dict.separator=,\n\
fsa.dict.encoding=UTF-8\n\
fsa.dict.encoder=prefix\n";

#[test]
fn morfologik_lemmatizer_and_tag_dictionary_and_builder() {
    assert_eq!(MorfologikDictionaryBuilder::to_dictionary_file_name("words.info"), "words.dict");
    assert_eq!(MorfologikDictionaryBuilder::to_dictionary_file_name("words.info.bak"), "words.info.bak");

    let dict_bytes = MorfologikDictionaryBuilder::compile_bytes(DICT_WITH_LEMMA_TXT, DICT_WITH_LEMMA_INFO).unwrap();
    let dictionary = Arc::new(MorfologikDictionary::from_bytes(&dict_bytes, DICT_WITH_LEMMA_INFO).unwrap());

    // MorfologikLemmatizerTest: case-insensitive lookup, returning None on miss.
    let lemmatizer = MorfologikLemmatizer::new(dictionary.clone());
    let lemmas = lemmatizer.lemmatize(&["casa", "casa", "Casa"], &["V", "NOUN", "PROP"]);
    assert_eq!(lemmas, vec![Some("casar".to_owned()), Some("casa".to_owned()), None]);

    let multi = lemmatizer.lemmatize_all(&["foi"], &["V"]);
    assert_eq!(multi[0].len(), 2);
    assert!(multi[0].contains(&"ir".to_owned()));
    assert!(multi[0].contains(&"ser".to_owned()));

    // MorfologikTagDictionaryTest: case-insensitive vs case-sensitive.
    let tagdict_insensitive = MorfologikTagDictionary::with_case_sensitivity(dictionary.clone(), false);
    assert_eq!(tagdict_insensitive.tags("carro").unwrap(), vec!["NOUN".to_owned()]);
    let casa_tags = tagdict_insensitive.tags("casa").unwrap();
    assert_eq!(casa_tags.len(), 2);
    assert!(casa_tags.contains(&"NOUN".to_owned()));
    assert!(casa_tags.contains(&"V".to_owned()));
    let cap_casa_tags = tagdict_insensitive.tags("Casa").unwrap();
    assert_eq!(cap_casa_tags.len(), 2);
    assert!(cap_casa_tags.contains(&"NOUN".to_owned()));
    assert!(cap_casa_tags.contains(&"V".to_owned()));

    let tagdict_sensitive = MorfologikTagDictionary::with_case_sensitivity(dictionary, true);
    let sens_casa = tagdict_sensitive.tags("casa").unwrap();
    assert_eq!(sens_casa.len(), 2);
    assert_eq!(tagdict_sensitive.tags("Casa").unwrap(), vec!["PROP".to_owned()]);

    // Tagless dictionary returns None for tags().
    let tagless_txt = b"carro,carro\ncasa,casa\n";
    let tagless_bytes = MorfologikDictionaryBuilder::compile_bytes(tagless_txt, DICT_WITH_LEMMA_INFO).unwrap();
    let tagless_dict = MorfologikTagDictionary::from_bytes(&tagless_bytes, DICT_WITH_LEMMA_INFO, true).unwrap();
    assert_eq!(tagless_dict.tags("carro"), None);

    // MorfologikLocaleTest: 'I' folds to 'i' (Locale.ROOT).
    let locale_txt = b"Illinois,illinois,PROP\nindex,indices,NOUN\n";
    let locale_bytes = MorfologikDictionaryBuilder::compile_bytes(locale_txt, DICT_WITH_LEMMA_INFO).unwrap();
    let locale_dict = Arc::new(MorfologikDictionary::from_bytes(&locale_bytes, DICT_WITH_LEMMA_INFO).unwrap());
    let locale_lemmatizer = MorfologikLemmatizer::new(locale_dict.clone());
    assert_eq!(
        locale_lemmatizer.lemmatize(&["Illinois", "INDICES"], &["PROP", "NOUN"]),
        vec![Some("Illinois".to_owned()), Some("index".to_owned())]
    );
    let locale_tagdict = MorfologikTagDictionary::with_case_sensitivity(locale_dict, false);
    assert_eq!(locale_tagdict.tags("Illinois").unwrap(), vec!["PROP".to_owned()]);
    assert_eq!(locale_tagdict.tags("INDICES").unwrap(), vec!["NOUN".to_owned()]);
}

#[test]
fn annotator_and_sequence_validator_and_disk_builder_round_trip() {
    use turbo_analysis::{Annotator, Document, Span, Spans, View};
    use turbo_analysis_layers::{PosTags, TagSet, Tokens};
    use turbo_sequence::SequenceValidator;

    let tmp_dir = std::env::temp_dir().join(format!("turbo_morfologik_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let txt_path = tmp_dir.join("sample.txt");
    let info_path = tmp_dir.join("sample.info");
    std::fs::write(&txt_path, DICT_WITH_LEMMA_TXT).unwrap();
    std::fs::write(&info_path, DICT_WITH_LEMMA_INFO).unwrap();

    let builder = MorfologikDictionaryBuilder;
    let dict_path = builder.build(&txt_path).unwrap();
    assert_eq!(dict_path, tmp_dir.join("sample.dict"));

    let lemmatizer = MorfologikLemmatizer::read(&dict_path).unwrap();
    let expanded = MorfologikDictionaryReader::read_path(&dict_path).unwrap();
    let dict = Arc::new(MorfologikDictionary::read(&dict_path).unwrap());
    let tagdict = MorfologikTagDictionary::new(dict);

    assert!(tagdict.valid_sequence(0, &["Casa"], &[], "PROP"));
    assert!(!tagdict.valid_sequence(0, &["Casa"], &[], "NOUN"));
    assert!(tagdict.valid_sequence(0, &["unknown"], &[], "NOUN"));

    let doc = Document::new("casa meninão xyz");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 13));
    spans.push(Span::new(14, 17));
    let tokens = Tokens { spans, sentence: None };
    let tagset = TagSet::new("ud", &["V", "NOUN"]);
    let v_tag = tagset.tag("V").unwrap();
    let noun_tag = tagset.tag("NOUN").unwrap();
    let pos = PosTags { tagset, tags: vec![v_tag, noun_tag, noun_tag], prob: None };

    let (lemmas1,) = lemmatizer.annotate((view, &tokens, &pos)).unwrap();
    let collected1: Vec<&str> = lemmas1.lemma.iter().collect();
    assert_eq!(collected1, vec!["casar", "menino", "O"]);

    let (lemmas2,) = expanded.annotate((view, &tokens, &pos)).unwrap();
    let collected2: Vec<&str> = lemmas2.lemma.iter().collect();
    assert_eq!(collected2, vec!["casar", "menino", "O"]);

    // Mismatched view ID between Tokens and View is rejected cleanly without panicking.
    let other_doc = Document::new("x");
    let other_view = other_doc.get::<View>().unwrap();
    assert!(lemmatizer.annotate((other_view, &tokens, &pos)).is_err());
    assert!(expanded.annotate((other_view, &tokens, &pos)).is_err());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn lookup_of_a_word_containing_the_separator_finds_nothing() {
    let info = b"fsa.dict.separator=,\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=none\n";
    let dict_bytes = MorfologikDictionaryBuilder::compile_bytes(b"casa,casa,NOUN\n", info).unwrap();
    let dictionary = MorfologikDictionary::from_bytes(&dict_bytes, info).unwrap();
    assert_eq!(dictionary.lookup("casa").len(), 1);
    // Without the check the walk continues into the encoded lemma and reports "NOUN" as a stem.
    assert_eq!(dictionary.lookup("casa,casa"), Vec::new());
}

#[test]
fn lookup_skips_entries_with_undecodable_tag_bytes() {
    let info = b"fsa.dict.separator=+\nfsa.dict.encoding=windows-1252\nfsa.dict.encoder=none\n";
    let bad_tag_seq = b"word+lemma+\x81";
    let fsa_bytes = single_fsa5_sequence(bad_tag_seq);
    let dictionary = MorfologikDictionary::from_bytes(&fsa_bytes, info).unwrap();
    assert!(dictionary.lookup("word").is_empty());
}

#[test]
fn metadata_follows_java_properties_continuations_and_surrogate_pairs() {
    let info = b"fsa.dict.separator=+\n\
fsa.dict.encoding=UTF-8\n\
fsa.dict.encoder=\\\n    suffix\n\
fsa.dict.input-conversion=a b,\\\n  \\uD83D\\uDE00 c\r\
# comment\n";
    let meta = MorfologikMetadata::parse(info).unwrap();
    assert_eq!(meta.encoder, BaseFormEncoding::Suffix);
    assert_eq!(meta.input_conversion.apply("a\u{1F600}"), "bc");

    let lone = b"fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=none\nx=\\uD83D\n";
    assert!(MorfologikMetadata::parse(lone).is_err());
}
