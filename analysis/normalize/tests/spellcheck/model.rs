//! The frequency dictionary loader (`FrequencyDictionaryLoaderTest`) and
//! the binary model.

use super::*;

// ---------------------------------------------------------------------
// FrequencyDictionaryLoaderTest.
// ---------------------------------------------------------------------

fn unigrams(text: &str) -> Result<Vec<(String, u64)>, turbo_normalize::spellcheck::MalformedLine> {
    let mut out = Vec::new();
    FrequencyDictionary::read_unigrams(text, |w, c| out.push((w.to_owned(), c)))?;
    Ok(out)
}

fn bigrams(text: &str) -> Result<Vec<(String, u64)>, turbo_normalize::spellcheck::MalformedLine> {
    let mut out = Vec::new();
    FrequencyDictionary::read_bigrams(text, |a, b, c| out.push((format!("{a} {b}"), c)))?;
    Ok(out)
}

fn pairs(rows: &[(&str, u64)]) -> Vec<(String, u64)> {
    rows.iter().map(|&(w, c)| (w.to_owned(), c)).collect()
}

#[test]
fn the_loader_skips_blank_and_comment_lines() {
    let mut n = 0;
    let read = FrequencyDictionary::read_unigrams("the\t100\n\n# a comment\n   \nworld\t50\n", |_, _| n += 1).unwrap();
    assert_eq!((read, n), (2, 2));
    assert_eq!(unigrams("the\t100\n\n# a comment\n   \nworld\t50\n").unwrap(), pairs(&[("the", 100), ("world", 50)]));
    let e = unigrams("the\tnotanumber\n").unwrap_err();
    assert_eq!(e.line_number, 1);
}

#[test]
fn a_unigram_word_ends_at_a_tab_or_space_only() {
    for (line, word) in [
        ("a 5", "a"),
        ("a\t5", "a"),
        ("a \t  \t 5", "a"),
        (" a 5", "a"),
        ("\t\ta 5", "a"),
        ("a 5 \t", "a"),
        (" a  5 ", "a"),
        ("a\u{B}b 5", "a\u{B}b"),
        ("a\u{C}b 5", "a\u{C}b"),
        ("a\u{A0}b 5", "a\u{A0}b"),
        ("a\u{3000}b 5", "a\u{3000}b"),
        ("\u{1F600} 5", "\u{1F600}"),
        ("中文 5", "中文"),
        ("cafe\u{301} 5", "cafe\u{301}"),
    ] {
        assert_eq!(unigrams(&format!("{line}\n")).unwrap(), pairs(&[(word, 5)]), "{line:?}");
    }
}

#[test]
fn bigram_columns_split_on_tab_and_space_runs() {
    assert_eq!(bigrams("the  world\t3\nhello \t there   4\n").unwrap(), pairs(&[("the world", 3), ("hello there", 4)]));
    assert_eq!(bigrams("a b\u{A0}c 5\n").unwrap(), pairs(&[("a b\u{A0}c", 5)]));
}

#[test]
fn lines_of_java_whitespace_are_skipped() {
    for blank in ["\t\t", " \t \t ", "\u{B}\u{C}", "\u{3000}", "\u{2028}"] {
        let text = format!("the\t100\n{blank}\nworld 5\n");
        assert_eq!(unigrams(&text).unwrap(), pairs(&[("the", 100), ("world", 5)]), "{blank:?}");
    }
}

#[test]
fn skipped_lines_count_toward_the_line_number() {
    let e = unigrams("the\t100\n\t\t\n# note\n\nworld\n").unwrap_err();
    assert_eq!(e.line_number, 5);
}

#[test]
fn counts_accepted() {
    for (count, expected) in [("007", 7), ("0", 0), ("9223372036854775807", 9223372036854775807)] {
        assert_eq!(unigrams(&format!("the\t{count}\n")).unwrap(), pairs(&[("the", expected)]));
    }
}

#[test]
fn malformed_unigram_lines_name_the_reason() {
    for (line, reason) in [
        ("the", "expected 'word<sep>count'"),
        ("the\u{A0}100", "expected 'word<sep>count'"),
        ("\u{A0}", "expected 'word<sep>count'"),
        ("\u{A0}\u{A0}", "expected 'word<sep>count'"),
        ("the\t-5", "count must not be negative"),
        ("the\t-99999999999999999999", "count must not be negative"),
        ("the\t9223372036854775808", "count is not an integer"),
        ("the\t5\u{A0}", "count is not an integer"),
        ("the\t5\u{1}", "count is not an integer"),
        ("the\t\u{1}5", "count is not an integer"),
        ("the\t5.0", "count is not an integer"),
        ("the\t1e3", "count is not an integer"),
        ("the\t99999999999999999999", "count is not an integer"),
        ("the\t-", "count is not an integer"),
        ("the\t+", "count is not an integer"),
        ("the\t+5", "count is not an integer"),
        ("the\t+-5", "count is not an integer"),
        ("the\t-0", "count must not be negative"),
    ] {
        let e = unigrams(&format!("{line}\n")).unwrap_err();
        assert_eq!(e.line_number, 1, "{line:?}");
        assert!(e.to_string().contains(&format!("({reason})")), "{line:?}: {e}");
    }
}

#[test]
fn counts_in_any_decimal_digits() {
    for (count, value) in [
        ("\u{665}", 5),
        ("\u{FF15}", 5),
        ("5\u{665}", 55),
        ("\u{661}\u{662}\u{663}", 123),
        ("\u{FF19}\u{FF12}\u{FF12}\u{FF13}", 9223),
        ("\u{1D7CE}", 0),
        ("0", 0),
        ("007", 7),
        ("\u{660}\u{665}", 5),
    ] {
        assert_eq!(unigrams(&format!("caf\u{E9}\t{count}\n")).unwrap(), pairs(&[("caf\u{E9}", value)]));
        assert_eq!(bigrams(&format!("\u{4E2D}\u{6587} caf\u{E9}\t{count}\n")).unwrap(), pairs(&[("中文 café", value)]));
    }
}

#[test]
fn counts_that_are_not_digits_are_refused() {
    for (count, reason) in [
        ("+\u{FF15}", MalformedReason::CountNotInteger),
        ("-\u{660}", MalformedReason::CountNegative),
        ("-\u{665}", MalformedReason::CountNegative),
        ("+5", MalformedReason::CountNotInteger),
        ("-", MalformedReason::CountNotInteger),
        ("\u{1F600}", MalformedReason::CountNotInteger),
        ("\u{665}\u{301}", MalformedReason::CountNotInteger),
        ("\u{B2}", MalformedReason::CountNotInteger),
    ] {
        assert_eq!(unigrams(&format!("caf\u{E9}\t{count}\n")).unwrap_err().reason, reason, "{count:?}");
        assert_eq!(bigrams(&format!("中文 café\t{count}\n")).unwrap_err().reason, reason, "{count:?}");
    }
}

#[test]
fn other_loader_cases() {
    let e = bigrams("the\t5\n").unwrap_err();
    assert_eq!(e.line_number, 1);
    assert!(e.to_string().contains("(expected 'w1<sep>w2<sep>count')"));
    assert_eq!(unigrams("the 100 extra\n").unwrap(), pairs(&[("the", 100)]));
    assert_eq!(unigrams("\u{FEFF}# note\nthe 5\n").unwrap(), pairs(&[("the", 5)]));
    // The message is OpenNLP's, with the tab of the line written as \t.
    let e = unigrams("the\tx\n").unwrap_err();
    assert_eq!(e.to_string(), "malformed dictionary line 1 (count is not an integer): \"the\\tx\"");
}

#[test]
fn loaders_feed_an_engine() {
    let mut e = SymSpell::new(SymSpellConfig::default());
    let u =
        FrequencyDictionary::load_unigrams(&mut e, &fs::read_to_string(data("frequency_dictionary_tiny.txt")).unwrap())
            .unwrap();
    let b = FrequencyDictionary::load_bigrams(
        &mut e,
        &fs::read_to_string(data("frequency_bigramdictionary_tiny.txt")).unwrap(),
    )
    .unwrap();
    assert_eq!(u as usize, e.word_count());
    assert_eq!(b as usize, e.bigram_count());
    assert_eq!(top(&e, "teh", 2)[0].term, "the");
}

// ---------------------------------------------------------------------
// The binary model (SymSpellModelSerializationTest,
// SymSpellModelSerializerLimitsTest) and bytes written by OpenNLP.
// ---------------------------------------------------------------------

pub(super) fn java_model(name: &str) -> (Vec<u8>, SymSpellModel) {
    let bytes = fs::read(data(name)).unwrap();
    let model = SymSpellModel::read(&bytes).unwrap();
    (bytes, model)
}

#[test]
fn models_written_by_opennlp_read_and_write_back_byte_for_byte() {
    let (bytes, model) = java_model("model-tiny.bin");
    assert_eq!((model.language(), model.name(), model.version()), ("en", "spellcheck-tiny", "1.0"));
    assert_eq!(*model.config(), SymSpellConfig::default());
    assert_eq!(model.unigrams().len(), 63 + 4);
    assert_eq!(model.unigrams()[0], ("the".to_owned(), 23135851162));
    assert!(model.unigrams().contains(&("a\u{0}b".to_owned(), 7)));
    assert!(model.unigrams().contains(&("\u{10428}word".to_owned(), 1000)));
    assert!(model.unigrams().contains(&("maximum".to_owned(), MAX_COUNT)));
    assert_eq!(model.bigrams().len(), 20);
    assert_eq!(model.write().unwrap(), bytes);

    let (bytes, model) = java_model("model-levenshtein.bin");
    let c = model.config();
    assert_eq!(
        (c.max_dictionary_edit_distance(), c.prefix_length(), c.count_threshold(), c.metric(), c.corpus_word_count()),
        (1, 4, 3, Metric::Levenshtein, 1_000_000_000)
    );
    assert_eq!((model.language(), model.name(), model.version()), ("de", "spellcheck-levenshtein", "2.5"));
    assert_eq!(model.write().unwrap(), bytes);
    let engine = model.build();
    // Below the count threshold of 3: kept, not indexed.
    assert_eq!(engine.word_count(), 8);
    assert!(!engine.contains("bonk") && !engine.contains("bandana") && engine.contains("bandanas"));
    assert_eq!(engine.bigram("river", "bank"), Some(30));
}

#[test]
fn round_trips_keep_the_model_and_its_suggestions() {
    let (_, model) = java_model("model-tiny.bin");
    let again = SymSpellModel::read(&model.write().unwrap()).unwrap();
    assert_eq!(again, model);
    let (a, b) = (model.build(), again.build());
    for (typo, word) in [
        ("teh", "the"),
        ("recieve", "receive"),
        ("wrold", "world"),
        ("quikc", "quick"),
        ("becuse", "because"),
        ("calender", "calendar"),
    ] {
        let x = a.lookup(typo, Verbosity::All, 2).unwrap();
        assert_eq!(x, b.lookup(typo, Verbosity::All, 2).unwrap());
        assert_eq!(x[0].term, word);
    }
    assert_eq!(b.lookup_compound("helloworld", 2).unwrap().term, "hello world");
    // A fixed corpus size survives; the derived one stays 0.
    let fixed = SymSpellConfig::builder().corpus_word_count(987_654_321).build().unwrap();
    let m = SymSpellModel::new("en", "n", "1", fixed, model.unigrams().to_vec(), model.bigrams().to_vec()).unwrap();
    assert_eq!(SymSpellModel::read(&m.write().unwrap()).unwrap().config().corpus_word_count(), 987_654_321);
    assert_eq!(again.config().corpus_word_count(), 0);
}

#[test]
fn repeated_rows_keep_the_first_position_and_the_last_count() {
    let m = SymSpellModel::new(
        "en",
        "n",
        "1",
        SymSpellConfig::default(),
        [("a".to_owned(), 1), ("b".to_owned(), 2), ("a".to_owned(), 3)],
        [("x y".to_owned(), 1)],
    )
    .unwrap();
    assert_eq!(m.unigrams(), [("a".to_owned(), 3), ("b".to_owned(), 2)]);
    for bad in ["xy", " y", "x "] {
        let e = SymSpellModel::new("en", "n", "1", SymSpellConfig::default(), [], [(bad.to_owned(), 1)]).unwrap_err();
        assert_eq!(e, ModelError::InvalidBigramKey(bad.to_owned()));
    }
    let e = SymSpellModel::new(" \t", "n", "1", SymSpellConfig::default(), [], []).unwrap_err();
    assert_eq!(e, ModelError::Blank("language"));
}

/// A header as OpenNLP's limits test writes it, up to the unigram count
/// and optionally a bigram count.
fn header(unigram_count: i32, bigram_count: Option<i32>) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x5359_4D53u32.to_be_bytes());
    b.extend_from_slice(&1u32.to_be_bytes());
    for s in ["en", "test", "1.0"] {
        b.extend_from_slice(&(s.len() as u16).to_be_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    b.extend_from_slice(&2i32.to_be_bytes());
    b.extend_from_slice(&7i32.to_be_bytes());
    b.extend_from_slice(&1i64.to_be_bytes());
    b.extend_from_slice(&11u16.to_be_bytes());
    b.extend_from_slice(b"damerau-osa");
    b.extend_from_slice(&0i64.to_be_bytes());
    b.extend_from_slice(&unigram_count.to_be_bytes());
    if let Some(n) = bigram_count {
        b.extend_from_slice(&n.to_be_bytes());
    }
    b
}

#[test]
fn row_counts_outside_the_limit_are_refused() {
    let over = MAX_ENTRIES as i32 + 1;
    for (bytes, field) in [
        (header(i32::MAX, None), "unigram count"),
        (header(over, None), "unigram count"),
        (header(-1, None), "unigram count"),
        (header(0, Some(i32::MAX)), "bigram count"),
        (header(0, Some(over)), "bigram count"),
        (header(0, Some(-1)), "bigram count"),
    ] {
        let e = SymSpellModel::read(&bytes).unwrap_err();
        assert!(matches!(e, ModelError::EntryCount { field: f, .. } if f == field), "{e:?}");
        assert!(e.to_string().contains(field));
    }
    assert!(SymSpellModel::read(&header(0, Some(0))).is_ok());
    assert!(matches!(SymSpellModel::read_with_limit(&header(5, None), 4), Err(ModelError::EntryCount { .. })));
}

#[test]
fn corrupt_streams_are_refused() {
    let good = header(0, Some(0));
    let mut bad_magic = good.clone();
    bad_magic[0] = 0;
    assert_eq!(SymSpellModel::read(&bad_magic), Err(ModelError::BadMagic(0x0059_4D53)));
    assert!(SymSpellModel::read(&bad_magic).unwrap_err().to_string().contains("magic was 0x00594D53"));
    let mut bad_version = good.clone();
    bad_version[7] = 2;
    assert_eq!(SymSpellModel::read(&bad_version), Err(ModelError::UnsupportedVersion(2)));
    for cut in 0..good.len() {
        assert_eq!(SymSpellModel::read(&good[..cut]), Err(ModelError::Truncated), "cut at {cut}");
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert_eq!(SymSpellModel::read(&trailing), Err(ModelError::TrailingBytes(1)));
    // The prefix length (7) set to the maximum edit distance (2).
    let mut config = good.clone();
    let at = 8 + 4 + 6 + 5 + 4 + 3;
    config[at] = 2;
    assert!(matches!(SymSpellModel::read(&config), Err(ModelError::CorruptConfig(_))));
    let mut metric = good.clone();
    let id = metric.windows(11).position(|w| w == b"damerau-osa").unwrap();
    metric[id] = b'x';
    assert!(SymSpellModel::read(&metric).unwrap_err().to_string().contains("unknown edit-distance id: xamerau-osa"));
}

#[test]
fn modified_utf8_and_vlong_rows() {
    let config = SymSpellConfig::default();
    let words = ["\u{0}", "é", "€", "\u{10428}", "plain"];
    let rows: Vec<(String, u64)> =
        words.iter().zip([0, 127, 128, 1 << 35, MAX_COUNT]).map(|(w, c)| ((*w).to_owned(), c)).collect();
    let m = SymSpellModel::new("en", "n", "1", config, rows.clone(), []).unwrap();
    let bytes = m.write().unwrap();
    // U+0000 is two bytes, a supplementary character two three-byte
    // surrogates.
    let find = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
    assert!(find(&[0x00, 0x02, 0xC0, 0x80, 0x00]));
    assert!(find(&[0x00, 0x06, 0xED, 0xA0, 0x81, 0xED, 0xB0, 0xA8]));
    // 127 is one byte, 128 two.
    assert!(find(&[0x00, 0x02, 0xC3, 0xA9, 0x7F]));
    assert!(find(&[0x00, 0x03, 0xE2, 0x82, 0xAC, 0x80, 0x01]));
    assert_eq!(SymSpellModel::read(&bytes).unwrap().unigrams(), rows.as_slice());

    // A lone surrogate cannot be a Rust string.
    let mut lone = header(1, None);
    lone.extend_from_slice(&[0x00, 0x03, 0xED, 0xA0, 0x81, 0x01]);
    lone.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&lone), Err(ModelError::MalformedUtf));
    // A continuation byte where a character starts.
    let mut bad = header(1, None);
    bad.extend_from_slice(&[0x00, 0x01, 0x80, 0x01]);
    bad.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&bad), Err(ModelError::MalformedUtf));
    // Eleven bytes of continuation is too long for a vlong.
    let mut long = header(1, None);
    long.extend_from_slice(&[0x00, 0x01, b'a']);
    long.extend_from_slice(&[0x80; 10]);
    long.push(0);
    assert_eq!(SymSpellModel::read(&long), Err(ModelError::VLongTooLong));
    // Ten bytes that set the top bit: Long.MIN_VALUE in Java.
    let mut negative = header(1, None);
    negative.extend_from_slice(&[0x00, 0x01, b'a']);
    negative.extend_from_slice(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    negative.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&negative), Err(ModelError::NegativeCount));
    // A string longer than a UTF field holds.
    let m = SymSpellModel::new("en", "n", "1", config, [("x".repeat(65536), 1)], []).unwrap();
    assert_eq!(m.write(), Err(ModelError::StringTooLong));
}
