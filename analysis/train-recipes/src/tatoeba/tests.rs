use super::*;

fn records(text: &str) -> Vec<Record<'_>> {
    Records::new(DETAILED_FILE, text).map(|r| r.unwrap()).collect()
}

fn values<'a>(r: &'a Record<'_>) -> Vec<Option<&'a str>> {
    r.fields.iter().map(|f| f.as_deref()).collect()
}

#[test]
fn fields_end_at_tabs_and_records_at_line_feeds() {
    let r = records("1\teng\tHi.\n2\tfra\tSalut.");
    assert_eq!(r.len(), 2);
    assert_eq!(values(&r[0]), [Some("1"), Some("eng"), Some("Hi.")]);
    assert_eq!(values(&r[1]), [Some("2"), Some("fra"), Some("Salut.")]);
    assert_eq!((r[0].line, r[1].line), (1, 2));
    assert!(matches!(r[0].fields[2], Some(Cow::Borrowed(_))));
    // An empty field, and an empty last field.
    let r = records("3\t\tx\t\n");
    assert_eq!(values(&r[0]), [Some("3"), Some(""), Some("x"), Some("")]);
}

#[test]
fn escapes_are_mysql_s() {
    // A backslash, a tab and a line feed after a backslash are
    // themselves, \0 is U+0000, and \N alone is NULL.
    let r = records("1\t\\N\tC:\\\\dir\\\tend\\\nnext\\0\t\\N\n2\tdeu\t\\\\N\n");
    assert_eq!(r.len(), 2, "the escaped line feed continues the record");
    assert_eq!(values(&r[0]), [Some("1"), None, Some("C:\\dir\tend\nnext\0"), None]);
    // An escaped backslash before N is text.
    assert_eq!(values(&r[1]), [Some("2"), Some("deu"), Some("\\N")]);
    assert_eq!(r[1].line, 3);
    assert_eq!(
        Records::new(DETAILED_FILE, "1\t\\N\ta\\\nb\n2\tx\\N\n").map(|r| r.map(|r| r.line)).collect::<Vec<_>>(),
        [Ok(1), Err(TatoebaError::Escape { file: DETAILED_FILE, line: 3, found: 'N' })]
    );
    let bad: Vec<_> = Records::new(CC0_FILE, "1\ta\\t").collect();
    assert_eq!(bad, [Err(TatoebaError::Escape { file: CC0_FILE, line: 1, found: 't' })]);
    let end: Vec<_> = Records::new(CC0_FILE, "1\ta\\").collect();
    assert_eq!(end, [Err(TatoebaError::UnterminatedEscape { file: CC0_FILE, line: 1 })]);
    // Characters after an escape are kept whole.
    let r = records("1\tx\\\\é😀\n");
    assert_eq!(values(&r[0]), [Some("1"), Some("x\\é😀")]);
}

fn export(detailed: &str, cc0: &str) -> Export {
    Export::from_bytes(detailed.as_bytes().to_vec(), cc0.as_bytes().to_vec()).unwrap()
}

const D: &str = "1\teng\tHi.\tann\t\\N\t\\N\n\
                 2\tfra\tSalut.\tbob\t2010-01-01 00:00:00\t2010-01-01 00:00:00\n\
                 3\t\\N\tNo language.\tann\t\\N\t\\N\n\
                 4\teng\tPublic domain.\tcy\t\\N\t\\N\n";

#[test]
fn a_sentence_the_cc0_export_lists_is_cc0_and_the_others_cc_by_2_0_fr() {
    let e = export(D, "4\teng\tPublic domain.\t\\N\n");
    let s = e.sentences(|l| l == "eng").unwrap();
    assert_eq!((s.records, s.no_language, s.cc0_records), (4, 1, 1));
    let kept: Vec<(u32, &str, SentenceLicence)> = s.kept.iter().map(|s| (s.id, &*s.language, s.licence)).collect();
    assert_eq!(kept, [(1, "eng", SentenceLicence::CcBy20Fr), (4, "eng", SentenceLicence::Cc0)]);
    assert_eq!(SentenceLicence::CcBy20Fr.licence(), Licence::CcBy20Fr);
    assert_eq!(SentenceLicence::Cc0.to_string(), "CC0 1.0");
    // The digest is that of the sha256sum manifest.
    let manifest = format!("{}  sentences_CC0.csv\n{}  sentences_detailed.csv\n", e.cc0.sha256, e.detailed.sha256);
    assert_eq!(e.digest(), Sha256::of(manifest.as_bytes()).to_string());
    assert_eq!(e.detailed.size, D.len() as u64);
}

#[test]
fn a_sentence_is_classified_by_its_id_and_the_cc0_rows_that_differ_are_counted() {
    // The CC0 file is written after the detailed one: sentence 4 was
    // edited in between, 2 had its language changed and was made CC0,
    // and 7 and 9 were added.
    let cc0 = "4\teng\tPublic domain!\t\\N\n2\tdeu\tSalut.\t\\N\n9\teng\tx\t\\N\n7\teng\ty\t\\N\n";
    let e = export(D, cc0);
    let s = e.sentences(|_| true).unwrap();
    let kept: Vec<(u32, SentenceLicence)> = s.kept.iter().map(|s| (s.id, s.licence)).collect();
    assert_eq!(
        kept,
        [(1, SentenceLicence::CcBy20Fr), (2, SentenceLicence::Cc0), (4, SentenceLicence::Cc0)],
        "4 and 2 are CC0 by their ids, and 7 and 9 are not sentences of the detailed export"
    );
    assert_eq!((s.records, s.no_language, s.cc0_records), (4, 1, 4));
    assert_eq!((s.cc0_differing, s.cc0_not_in_detailed), (2, 2));
    // The same language and text in both files differ in nothing, even
    // with another `modified`.
    let counts = |cc0: &str| {
        let e = export(D, cc0);
        let s = e.sentences(|_| true).unwrap();
        (s.cc0_records, s.cc0_differing, s.cc0_not_in_detailed, s.no_language)
    };
    assert_eq!(counts("4\teng\tPublic domain.\t2020-01-01 00:00:00\n"), (1, 0, 0, 1));
    // A NULL language in one file and a language in the other differ.
    assert_eq!(counts("3\teng\tNo language.\t\\N\n"), (1, 1, 0, 1));
}

#[test]
fn bad_records_are_refused() {
    let s = |d: &str, c: &str| export(d, c).sentences(|_| true).map(|s| s.kept.len());
    assert_eq!(
        s("1\teng\tHi.\n", ""),
        Err(TatoebaError::Fields { file: DETAILED_FILE, line: 1, found: 3, expected: 6 })
    );
    assert_eq!(
        s("\teng\tHi.\ta\tb\tc\n", ""),
        Err(TatoebaError::Id { file: DETAILED_FILE, line: 1, value: String::new() })
    );
    assert_eq!(
        s("0\teng\tHi.\ta\tb\tc\n", ""),
        Err(TatoebaError::Id { file: DETAILED_FILE, line: 1, value: "0".into() })
    );
    assert_eq!(
        s("01\teng\tHi.\ta\tb\tc\n", ""),
        Err(TatoebaError::Id { file: DETAILED_FILE, line: 1, value: "01".into() })
    );
    assert_eq!(
        s("x\teng\tHi.\ta\tb\tc\n", ""),
        Err(TatoebaError::Id { file: DETAILED_FILE, line: 1, value: "x".into() })
    );
    assert_eq!(
        s("+1\teng\tHi.\ta\tb\tc\n", ""),
        Err(TatoebaError::Id { file: DETAILED_FILE, line: 1, value: "+1".into() })
    );
    assert_eq!(
        s("1\teng\t\\N\ta\tb\tc\n", ""),
        Err(TatoebaError::Null { file: DETAILED_FILE, line: 1, field: "text" })
    );
    assert_eq!(
        s("1\teng\ta\tb\tc\td\n1\teng\ta\tb\tc\td\n", ""),
        Err(TatoebaError::DuplicateId { file: DETAILED_FILE, id: 1 })
    );
    assert_eq!(
        s(D, "4\teng\tPublic domain.\t\\N\n4\teng\tPublic domain.\t\\N\n"),
        Err(TatoebaError::DuplicateId { file: CC0_FILE, id: 4 })
    );
    assert_eq!(
        Export::from_bytes(vec![b'1', 0xFF], Vec::new()).unwrap_err(),
        TatoebaError::Utf8 { file: DETAILED_FILE, offset: 1 }
    );
}
