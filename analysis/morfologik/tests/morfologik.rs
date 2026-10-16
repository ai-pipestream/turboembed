//! Integration tests for `turbo-morfologik`, porting OpenNLP 3.x's
//! `FSA5ReaderTest`, `CFSA2ReaderTest`, `MorfologikDictionaryReaderTest`,
//! `MorfologikBaseEncodingTest`, `MorfologikDictionaryExpansionTest`,
//! `MorfologikInputConversionTest`, `MorfologikMetadataTest`,
//! `PoliMorfDictionaryReaderUsageExampleTest`, `MorfologikLemmatizerTest`,
//! `MorfologikTagDictionaryTest`, `MorfologikLocaleTest`, and
//! `MorfologikDictionaryBuilderTest`.

use std::sync::Arc;

use turbo_morfologik::{
    BaseFormEncoding, Cfsa2Reader, DictionaryCharset, Fsa5Reader, FsaReader, MorfologikDictionary,
    MorfologikDictionaryBuilder, MorfologikDictionaryReader, MorfologikError, MorfologikLemmatizer, MorfologikMetadata,
    MorfologikTagDictionary, PoliMorfDictionaryReader, REPLACE_FORM, decode_base_form,
};

fn decode_base64(input: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for b in input.bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => panic!("invalid base64 byte: {b}"),
        };
        buf = (buf << 6) | u32::from(val);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    out
}

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

fn sequences_utf8(reader: &FsaReader) -> Vec<String> {
    reader.sequences().into_iter().map(|b| String::from_utf8(b).unwrap()).collect()
}

// ============================================================================
// FSA5ReaderTest & CFSA2ReaderTest
// ============================================================================

const FSA5_FIXTURE_B64: &str = "XGZzYQVfKwEAAF4GY3BkBm8HZwdzA2EGdGM=";
const CFSA2_FIXTURE_B64: &str = "XGZzYcYABwgAdHNvZ2RjYUBeAwYKxePkYgDHYQg=";
const EXPECTED_WORDS: &[&str] = &["cat", "cats", "do", "dog", "dogs"];

#[test]
fn fsa5_enumerates_all_accepted_sequences() {
    let bytes = decode_base64(FSA5_FIXTURE_B64);
    let reader = Fsa5Reader::from_bytes(&bytes).unwrap();
    let words: Vec<String> = reader.sequences().into_iter().map(|b| String::from_utf8(b).unwrap()).collect();
    assert_eq!(words, EXPECTED_WORDS);
    assert_eq!(sequences_utf8(&FsaReader::from_bytes(&bytes).unwrap()), EXPECTED_WORDS);
}

#[test]
fn fsa5_rejects_truncated_and_unsupported() {
    let bytes = decode_base64(FSA5_FIXTURE_B64);
    let err = Fsa5Reader::from_bytes(&bytes[..6]).unwrap_err();
    assert!(err.message().starts_with("truncated FSA5 header"));

    let mut altered = bytes.clone();
    altered[4] = 0x99;
    assert!(Fsa5Reader::from_bytes(&altered).is_err());
    assert!(FsaReader::from_bytes(&altered).is_err());
    assert!(FsaReader::from_bytes(b"not an fsa header").is_err());
}

#[test]
fn cfsa2_enumerates_all_accepted_sequences() {
    let bytes = decode_base64(CFSA2_FIXTURE_B64);
    let reader = Cfsa2Reader::from_bytes(&bytes).unwrap();
    let words: Vec<String> = reader.sequences().into_iter().map(|b| String::from_utf8(b).unwrap()).collect();
    assert_eq!(words, EXPECTED_WORDS);
    assert_eq!(sequences_utf8(&FsaReader::from_bytes(&bytes).unwrap()), EXPECTED_WORDS);
}

#[test]
fn cfsa2_rejects_truncated_and_unsupported() {
    let bytes = decode_base64(CFSA2_FIXTURE_B64);
    assert!(Cfsa2Reader::from_bytes(b"not an fsa header").is_err());
    let err = Cfsa2Reader::from_bytes(&bytes[..6]).unwrap_err();
    assert!(err.message().starts_with("truncated CFSA2 header"));

    let mut altered = bytes;
    altered[4] = 0x05;
    assert!(Cfsa2Reader::from_bytes(&altered).is_err());
}

// ============================================================================
// MorfologikDictionaryReaderTest
// ============================================================================

const DICTIONARY_B64: &str = "XGZzYcYABxIAdXRtaWdkYVNEQkFvZWNzTitAXgMOHwYVw8TOzdHJzMHPzdHQcADMxc/RytHQ0GgAx8KRTxhLEQ==";
const FSA5_DICTIONARY_B64: &str =
    "XGZzYQVfKwIAAABeBmPIAWQwAW0GaQZjBmUGKwZEBm8GdQZzBmUGKwZOBk4DAG8GZwZzBisGQgYrBk4GTgZTAwBhBnQGKxgCc2IBQfoA";

#[test]
fn decode_base_form_matches_morfologik_encoders() {
    let cases = [
        (BaseFormEncoding::Suffix, "cats", "B", "cat"),
        (BaseFormEncoding::Suffix, "mice", "Douse", "mouse"),
        (BaseFormEncoding::Suffix, "foobar", "Gbar", "bar"),
        (BaseFormEncoding::Suffix, "cat", "A", "cat"),
        (BaseFormEncoding::Prefix, "cats", "AB", "cat"),
        (BaseFormEncoding::Prefix, "foobar", "DA", "bar"),
        (BaseFormEncoding::Prefix, "unhappy", "CA", "happy"),
        (BaseFormEncoding::Prefix, "mice", "ADouse", "mouse"),
        (BaseFormEncoding::Infix, "cats", "AAB", "cat"),
        (BaseFormEncoding::Infix, "foobar", "ADA", "bar"),
        (BaseFormEncoding::Infix, "unhappy", "ACA", "happy"),
        (BaseFormEncoding::Infix, "mice", "AADouse", "mouse"),
        (BaseFormEncoding::None, "cats", "cat", "cat"),
    ];
    for (enc, form, encoded, expected) in cases {
        let decoded = decode_base_form(form.as_bytes(), encoded.as_bytes(), enc).unwrap();
        assert_eq!(String::from_utf8(decoded).unwrap(), expected);
    }

    let invalid_cases = [
        (BaseFormEncoding::Suffix, "cat", ""),
        (BaseFormEncoding::Prefix, "cat", "A"),
        (BaseFormEncoding::Infix, "cat", "AA"),
        (BaseFormEncoding::Suffix, "cat", "Z"),
        (BaseFormEncoding::Prefix, "cat", "ZA"),
        (BaseFormEncoding::Infix, "cat", "ZAA"),
    ];
    for (enc, form, encoded) in invalid_cases {
        assert!(decode_base_form(form.as_bytes(), encoded.as_bytes(), enc).is_err());
    }
}

#[test]
fn reads_cfsa2_and_fsa5_dictionaries_end_to_end() {
    let cfsa_bytes = decode_base64(DICTIONARY_B64);
    let lemmatizer = MorfologikDictionaryReader::read_with_options(
        &cfsa_bytes,
        b'+',
        BaseFormEncoding::Suffix,
        DictionaryCharset::Iso8859_1,
    )
    .unwrap();
    assert_eq!(lemmatizer.lemmatize(&["cats"], &["NNS"]), vec!["cat"]);
    assert_eq!(lemmatizer.lemmatize(&["dogs"], &["NNS"]), vec!["dog"]);
    assert_eq!(lemmatizer.lemmatize(&["mice"], &["NN"]), vec!["mouse"]);
    assert_eq!(lemmatizer.lemmatize(&["cat"], &["NN"]), vec!["cat"]);
    assert_eq!(lemmatizer.lemmatize(&["mice"], &["NNS"]), vec!["O"]);

    let fsa5_bytes = decode_base64(FSA5_DICTIONARY_B64);
    let fsa5_lemmatizer = MorfologikDictionaryReader::read_with_options(
        &fsa5_bytes,
        b'+',
        BaseFormEncoding::Suffix,
        DictionaryCharset::Iso8859_1,
    )
    .unwrap();
    assert_eq!(fsa5_lemmatizer.lemmatize(&["mice"], &["NN"]), vec!["mouse"]);
    assert_eq!(fsa5_lemmatizer.lemmatize(&["cats"], &["NNS"]), vec!["cat"]);

    let info_lemmatizer = MorfologikDictionaryReader::read(
        &cfsa_bytes,
        b"fsa.dict.separator=+\nfsa.dict.encoding=iso-8859-1\nfsa.dict.encoder=Suffix\n",
    )
    .unwrap();
    assert_eq!(info_lemmatizer.lemmatize(&["mice"], &["NN"]), vec!["mouse"]);

    // Missing metadata key, multi-char separator, unknown encoder, and missing entry separator.
    assert!(MorfologikDictionaryReader::read(&cfsa_bytes, b"fsa.dict.separator=+\n").is_err());
    assert!(
        MorfologikDictionaryReader::read(
            &cfsa_bytes,
            b"fsa.dict.separator=++\nfsa.dict.encoding=iso-8859-1\nfsa.dict.encoder=suffix\n"
        )
        .is_err()
    );
    assert!(
        MorfologikDictionaryReader::read(
            &cfsa_bytes,
            b"fsa.dict.separator=+\nfsa.dict.encoding=iso-8859-1\nfsa.dict.encoder=circumfix\n"
        )
        .is_err()
    );
    let err = MorfologikDictionaryReader::read_with_options(
        &cfsa_bytes,
        b'~',
        BaseFormEncoding::Suffix,
        DictionaryCharset::Iso8859_1,
    )
    .unwrap_err();
    assert!(err.message().starts_with("morfologik entry has no separator"));
}

// ============================================================================
// MorfologikBaseEncodingTest
// ============================================================================

#[test]
fn base_encoding_unsigned_controls_and_replacements() {
    let form = "abc".repeat(256);
    let append = "xyz";

    let encode_controls = |controls: &[usize]| -> Vec<u8> {
        let mut out = Vec::with_capacity(controls.len() + append.len());
        for &c in controls {
            out.push((c as u8).wrapping_add(b'A'));
        }
        out.extend_from_slice(append.as_bytes());
        out
    };

    for count in [0, 1, 2, 26, 50, 127, 128, 200, 234, 254] {
        // SUFFIX
        let expected = format!("{}{append}", &form[..form.len() - count]);
        assert_eq!(
            decode_base_form(form.as_bytes(), &encode_controls(&[count]), BaseFormEncoding::Suffix).unwrap(),
            expected.as_bytes()
        );
        // PREFIX (count, 0)
        let expected_p1 = format!("{}{append}", &form[count..]);
        assert_eq!(
            decode_base_form(form.as_bytes(), &encode_controls(&[count, 0]), BaseFormEncoding::Prefix).unwrap(),
            expected_p1.as_bytes()
        );
        // INFIX (count, 1, 0)
        let expected_i1 = format!("{}{}{append}", &form[..count], &form[count + 1..]);
        assert_eq!(
            decode_base_form(form.as_bytes(), &encode_controls(&[count, 1, 0]), BaseFormEncoding::Infix).unwrap(),
            expected_i1.as_bytes()
        );
    }

    // Infix at position 255 does not request full replacement.
    let expected_pos255 = format!("{}{}{append}", &form[..255], &form[256..]);
    assert_eq!(
        decode_base_form(form.as_bytes(), &encode_controls(&[255, 1, 0]), BaseFormEncoding::Infix).unwrap(),
        expected_pos255.as_bytes()
    );

    // Full replacement (REPLACE_FORM = 255) across lengths.
    for len in [0, 1, 3, 254, 255, 300, 600] {
        let f = &form.as_bytes()[..len];
        assert_eq!(
            decode_base_form(f, &encode_controls(&[REPLACE_FORM]), BaseFormEncoding::Suffix).unwrap(),
            append.as_bytes()
        );
        assert_eq!(
            decode_base_form(f, &encode_controls(&[REPLACE_FORM, 254]), BaseFormEncoding::Prefix).unwrap(),
            append.as_bytes()
        );
        assert_eq!(
            decode_base_form(f, &encode_controls(&[254, REPLACE_FORM, 1]), BaseFormEncoding::Infix).unwrap(),
            append.as_bytes()
        );
    }

    // UTF-8 multibyte byte-level suffix removal.
    assert_eq!(
        decode_base_form("żółwiami".as_bytes(), &encode_controls(&[4]), BaseFormEncoding::Suffix).unwrap(),
        "żółwxyz".as_bytes()
    );
    assert_eq!(
        decode_base_form("a😀".as_bytes(), &encode_controls(&[4]), BaseFormEncoding::Suffix).unwrap(),
        "axyz".as_bytes()
    );

    // Overlapping removals are rejected.
    for (enc, controls) in [
        (BaseFormEncoding::Prefix, vec![4, 3]),
        (BaseFormEncoding::Prefix, vec![6, 1]),
        (BaseFormEncoding::Infix, vec![2, 3, 2]),
        (BaseFormEncoding::Infix, vec![5, 1, 1]),
        (BaseFormEncoding::Infix, vec![6, 0, 1]),
    ] {
        assert!(decode_base_form(b"abcdef", &encode_controls(&controls), enc).is_err());
    }

    // Separator byte ('+' = 43 = (234 + 65) & 0xff) inside control bytes is not treated as a field separator.
    let form_300 = "a".repeat(300);
    let expected_lemma = format!("{}{append}", "a".repeat(66));
    let mut seq = Vec::new();
    seq.extend_from_slice(form_300.as_bytes());
    seq.push(b'+');
    seq.extend_from_slice(&encode_controls(&[234]));
    seq.push(b'+');
    seq.extend_from_slice(b"NN");
    let fsa_bytes = single_fsa5_sequence(&seq);
    let dict = MorfologikDictionaryReader::read_with_options(
        &fsa_bytes,
        b'+',
        BaseFormEncoding::Suffix,
        DictionaryCharset::Utf8,
    )
    .unwrap();
    assert_eq!(dict.lemmatize(&[&form_300], &["NN"]), vec![expected_lemma]);
}

#[test]
fn generated_morfologik_2_2_0_dictionaries_all_encoders_and_serializers() {
    let fixtures = [
        (
            BaseFormEncoding::Suffix,
            "XGZzYQVfKwIAAABeBmOIAWQwAW0GaQZjBmUGQgZEBm8GdQZzBmUGQgZOBk4DAG8GZwZzBkIGQvoAYQZ0UgE=",
        ),
        (BaseFormEncoding::Suffix, "XGZzYcYABw8AdXRtaWdkYURzb2VjTkJAXgMMGwYVw8TMy87IysHJy87NbQDKxcnOThHHQhc="),
        (
            BaseFormEncoding::Prefix,
            "XGZzYQVfKwIAAABeBmOoAWRAAW0GaQZjBmUGQgZBBkQGbwZ1BnMGZQZCBk4GTgMAbwZnBnMGQgZBBkIKAWEGdGIB",
        ),
        (BaseFormEncoding::Prefix, "XGZzYcYABxAAdXRtaWdkYURzb2VjTkFCQF4DDB0GFsPEzMvPzsjKwcnLz81tAMrFyc/OTxLHQhg="),
        (
            BaseFormEncoding::Infix,
            "XGZzYQVfKwIAAABeBmPIAWRQAW0GaQZjBmUGQgZBBkEGRAZvBnUGcwZlBkIGTgZOAwBvBmcGcwZCBkEGQQZCGgFhBnRyAQ==",
        ),
        (BaseFormEncoding::Infix, "XGZzYcYABxAAdXRtaWdkYURzb2VjTkJBQF4DDB8GF8PEzMvOz8/IysHJy87NbQDKxcnOz89OE8dCGQ=="),
        (
            BaseFormEncoding::None,
            "XGZzYQVfKwIAAABeBmOoAWQwAW0GaQZjBmUGQgZtBm8GdQZzBmUGQgZOBk4DAG8GZwZzBkIGZAZvBmf6AGEGdAZzBkIGYwZhBnT6AA==",
        ),
        (BaseFormEncoding::None, "XGZzYcYABw4AdWl0bWdlZGFOc29jQkBeAwwdBxXEwszGzcTLwcrGzclpAMvFys3Hy0URyMPKzczIQxE="),
    ];

    for (enc, b64) in fixtures {
        let bytes = decode_base64(b64);
        let dict = MorfologikDictionaryReader::read_with_options(&bytes, b'B', enc, DictionaryCharset::Utf8).unwrap();
        assert_eq!(dict.lemmatize(&["cats", "dogs", "mice"], &["NN", "NN", "NN"]), vec!["cat", "dog", "mouse"]);
    }
}

// ============================================================================
// MorfologikDictionaryExpansionTest (49,152 sequences with shared suffixes)
// ============================================================================

#[test]
fn expanded_dictionary_shared_suffixes_and_node_numbers() {
    const FORM_BITS: usize = 14;
    const FORM_COUNT: usize = 1 << FORM_BITS;

    let fixtures = [
        "XGZzYQVfKwIAAABeBnhQAHkGeHgAeQZ4oAB5BnjIAHkGePAAeQZ4GAF5BnhAAXkGeGgBeQZ4kAF5Bni4AXkGeOABeQZ4CAJ5BngwAnkGeFgCeQZzBisGQgYrGAMtBmEGbAZ0BisGTgZOAwBOAANWBkIDAA==",
        "XGZzYcYABwwAdHNsYVYtQk4reXhAXgOLyovKi8qLyovKi8qLyovKi8qLyovKi8qLyovKwsnHCSzGxMPBychoAAgqxWcA",
        "XGZzYQVfKyIAAAAAAAAAXgYAwHiAAHkGAGB4uAB5BgAwePAAeQYAGHgoAXkGAAx4YAF5BgAGeJgBeQYAA3jQAXkGgAF4CAJ5BsAAeEACeQZgAHh4AnkGMAB4sAJ5BhgAeOgCeQYMAHggA3kGBgB4WAN5BgMAcwYDACsGAwBCBgMAK7gELQYBAGEGAQBsBgEAdAYBACsGAQBOBgEATgMAAgBOkARWBgEAQgMA",
        "XGZzYcYBBwwAdHNsYVYtQk4reXgAQF4EgIADi8qAwAGLyoBgi8qAMIvKgBiLyoAMi8qABovKgAOLysABi8pgi8owi8oYi8oMi8oGi8oDwgPJA8cDCVDGAcQBwwHBAckByAFoAAIITcUBZwA=",
    ];

    let stem = |index: usize| -> String {
        let mut s = String::with_capacity(FORM_BITS);
        for bit in (0..FORM_BITS).rev() {
            s.push(if (index & (1 << bit)) == 0 { 'x' } else { 'y' });
        }
        s
    };

    for b64 in fixtures {
        let data = decode_base64(b64);
        let props =
            b"fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=SUFFIX\nfsa.dict.input-conversion=z x\n";
        let dict = MorfologikDictionaryReader::read(&data, props).unwrap();
        assert_eq!(dict.len(), FORM_COUNT * 2);

        for index in [0, 1, 42, 1024, FORM_COUNT - 1] {
            let base = stem(index);
            let surface = format!("{base}s").replace('x', "z");
            assert_eq!(dict.lemmatize_all(&[&surface], &["NN"]), vec![vec![base.clone(), format!("{base}-alt")]]);
            assert_eq!(dict.lemmatize_all(&[&surface], &["VB"]), vec![vec![base.clone()]]);
            assert_eq!(dict.lemmatize(&[&surface], &["NN"]), vec![base]);
        }
    }
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
fn rejects_cyclic_fsa5_and_cfsa2_automata() {
    // FSA5 where root arc at offset 4 points back to offset 4 (target = 4, encoded = (4 << 3) | 0x02 = 34).
    let cyclic_fsa5: [u8; 14] = [
        b'\\', b'f', b's', b'a', 0x05, b'_', b'+', 0x01, // header (gotoLength = 1)
        0, 0x02, // node 0 (terminal): 2 bytes
        b'^', 0x06, // node 2 (epsilon): BIT_TARGET_NEXT | BIT_LAST_ARC -> root is node 4
        b'a', 34, // node 4 (root): target = 34 >> 3 = 4 (self-cycle!)
    ];
    let err = Fsa5Reader::from_bytes(&cyclic_fsa5).unwrap_err();
    assert!(err.message().contains("cycle"), "unexpected error: {err}");

    // CFSA2 where root node at offset 2 points back to offset 2.
    let cyclic_cfsa2: [u8; 13] = [
        b'\\', b'f', b's', b'a', 0xc6, 0, 0, 0, // header (0 label mapping bytes)
        0xc0, b'^', // node 0: BIT_TARGET_NEXT | BIT_LAST_ARC -> root is node 2
        0x40, b'a', 2, // node 2 (root): BIT_LAST_ARC, inline label 'a', vint target = 2 (self-cycle!)
    ];
    let err_cfsa = Cfsa2Reader::from_bytes(&cyclic_cfsa2).unwrap_err();
    assert!(err_cfsa.message().contains("cycle"), "unexpected error: {err_cfsa}");
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

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn builder_widens_addresses_for_large_dictionaries() {
    // An unminimized trie of 60,000 random 16-byte words needs more than 2^21 bytes of arcs,
    // which no longer fits a 3-byte FSA5 address with its 3 flag bits.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut sequences: Vec<Vec<u8>> = (0..60_000)
        .map(|_| {
            (0..16)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b'a' + (state % 26) as u8
                })
                .collect()
        })
        .collect();
    sequences.sort();
    sequences.dedup();
    let bytes = turbo_morfologik::serialize_fsa5(&sequences, b'+');
    assert!(bytes.len() > 1 << 21, "fixture too small to cross the 3-byte address limit");
    let reader = Fsa5Reader::from_bytes(&bytes).unwrap();
    assert!(reader.sequences() == sequences, "large FSA5 round trip lost or changed sequences");
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
