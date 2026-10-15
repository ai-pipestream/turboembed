//! Malformed and hostile dictionaries and words: every load either succeeds
//! or fails with an error, never panics, and stemming stays within bounds of
//! time and output on input built to make the search explode. Also the
//! flags and options whose behaviour is easy to get backwards, each on a
//! dictionary small enough to read.

use std::time::{Duration, Instant};

use turbo_hunspell::{Dictionary, LoadError, LoadMode, Stemmer};

fn load(affix: &str, words: &str) -> Dictionary {
    Dictionary::from_bytes(affix.as_bytes(), words.as_bytes(), LoadMode::Strict).unwrap_or_else(|e| panic!("{e}"))
}

fn error(affix: &str, words: &str) -> LoadError {
    match Dictionary::from_bytes(affix.as_bytes(), words.as_bytes(), LoadMode::Strict) {
        Ok(_) => panic!("{affix:?} loaded"),
        Err(e) => e,
    }
}

fn stems(d: &Dictionary, word: &str) -> Vec<String> {
    Stemmer::new(d).stems(word)
}

/// Runs `f` and fails when it takes longer than `limit`. The limits are
/// loose enough for a debug build on a busy machine; what they catch is a
/// search that grows exponentially, which takes minutes.
fn within<T>(limit: Duration, what: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    assert!(t.elapsed() < limit, "{what} took {:?}", t.elapsed());
    out
}

fn fixtures() -> Vec<(String, Vec<u8>, Vec<u8>)> {
    let dir = turbo_goldens::data_dir().join("fixtures/hunspell");
    let mut out = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    for p in names {
        if p.extension().is_some_and(|e| e == "aff") {
            let words = std::fs::read(p.with_extension("dic")).unwrap();
            out.push((p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap(), words));
        }
    }
    out
}

const PROBES: [&str; 8] = ["walks", "football", "ház", "Straße", "1st", "", "-", "x-y"];

/// Every prefix of every fixture's affix file, and of its word list, loads
/// or fails cleanly, and what loads stems the probes.
#[test]
fn truncated_files_never_panic() {
    for (name, affix, words) in fixtures() {
        for mode in [LoadMode::Strict, LoadMode::AllowPartial] {
            for cut in 0..=affix.len() {
                if let Ok(d) = Dictionary::from_bytes(&affix[..cut], &words, mode) {
                    for p in PROBES {
                        Stemmer::new(&d).analyze(p);
                    }
                }
            }
            for cut in 0..=words.len() {
                if let Ok(d) = Dictionary::from_bytes(&affix, &words[..cut], mode) {
                    for p in PROBES {
                        Stemmer::new(&d).stems(p);
                    }
                }
            }
        }
        let _ = name;
    }
}

/// Each byte of each fixture replaced by a few others that change its
/// meaning: a digit, a space, a line break, a slash, a bracket, a byte
/// that is not UTF-8.
#[test]
fn altered_bytes_never_panic() {
    for (_, affix, words) in fixtures() {
        for (which, file) in [(0, &affix), (1, &words)] {
            for at in 0..file.len() {
                for b in [b'0', b'9', b' ', b'\n', b'/', b'[', b']', b'(', b'*', 0xFF, b'-'] {
                    let mut changed = file.clone();
                    changed[at] = b;
                    let (a, w) = if which == 0 { (&changed, &words) } else { (&affix, &changed) };
                    if let Ok(d) = Dictionary::from_bytes(a, w, LoadMode::AllowPartial) {
                        let s = Stemmer::new(&d);
                        for p in PROBES {
                            s.stems(p);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn counts_larger_than_the_file_are_errors_not_allocations() {
    for affix in [
        "SFX A Y 2000000000\nSFX A 0 s .\n",
        "AF 2000000000\nAF A\n",
        "AM 2000000000\nAM po:x\n",
        "CHECKCOMPOUNDREP\nREP 2000000000\nREP a b\n",
        "ICONV 2000000000\nICONV a b\n",
        "BREAK 2000000000\nBREAK -\n",
        "COMPOUNDRULE 2000000000\nCOMPOUNDRULE ab\n",
        "CHECKCOMPOUNDPATTERN 2000000000\nCHECKCOMPOUNDPATTERN a b\n",
    ] {
        error(affix, "1\nx\n");
    }
    // REP is a suggestion table, read only under CHECKCOMPOUNDREP.
    load("REP 2000000000\nREP a b\n", "1\nx\n");
    // The count line of the word list is only a hint.
    let d = load("", "99999999999999999999\nwalk\n");
    assert_eq!(stems(&d, "walk"), ["walk"]);
}

#[test]
fn flag_formats_reject_what_they_cannot_read() {
    for (affix, words) in [
        ("FLAG num\n", "1\nx/0\n"),
        ("FLAG num\n", "1\nx/65536\n"),
        ("FLAG num\n", "1\nx/-1\n"),
        ("FLAG num\n", "1\nx/1,,2\n"),
        ("FLAG num\n", "1\nx/99999999999999999999\n"),
        ("FLAG long\n", "1\nx/abc\n"),
        ("FLAG long\nSFX abc Y 1\n", "1\nx\n"),
        ("FLAG num\nSFX 0 Y 1\nSFX 0 0 s .\n", "1\nx\n"),
        ("AF 1\nAF A\n", "1\nx/0\n"),
        ("AF 1\nAF A\n", "1\nx/99999999999\n"),
        ("FLAG chars\n", "1\nx\n"),
        ("FLAG\n", "1\nx\n"),
        ("FLAG long\nFLAG num\n", "1\nx\n"),
    ] {
        error(affix, words);
    }
    // With an AF table only a run of digits is an alias; other text is
    // flags.
    let d = load("AF 1\nAF A\nSFX - Y 1\nSFX - 0 s .\n", "1\nx/-5\n");
    assert_eq!(stems(&d, "xs"), ["x"]);
    // The largest numeric flag and a supplementary UTF-8 flag both work.
    let d = load("FLAG num\nSFX 65535 Y 1\nSFX 65535 0 s .\n", "1\ncat/65535\n");
    assert_eq!(stems(&d, "cats"), ["cat"]);
    let d = load("FLAG UTF-8\nSFX \u{1F600} Y 1\nSFX \u{1F600} 0 s .\n", "1\ncat/\u{1F600}\n");
    assert_eq!(stems(&d, "cats"), ["cat"]);
    let d = load("FLAG long\nSFX zZ Y 1\nSFX zZ 0 s .\n", "1\ncat/zZ\n");
    assert_eq!(stems(&d, "cats"), ["cat"]);
    assert!(stems(&d, "cat").contains(&"cat".to_string()));
}

#[test]
fn circumfix_needs_both_sides() {
    let d = load("CIRCUMFIX C\nPFX P Y 1\nPFX P 0 ge/C .\nSFX T Y 1\nSFX T 0 t/C .\n", "1\nlach/PT\n");
    assert_eq!(stems(&d, "gelacht"), ["lach"]);
    assert!(stems(&d, "gelach").is_empty());
    assert!(stems(&d, "lacht").is_empty());
    assert_eq!(stems(&d, "lach"), ["lach"]);
}

#[test]
fn needaffix_and_onlyincompound_entries_do_not_stand_alone() {
    let d = load(
        "NEEDAFFIX X\nONLYINCOMPOUND O\nCOMPOUNDFLAG Y\nSFX S Y 1\nSFX S 0 s .\n",
        "3\nvirt/XS\nlink/OY\nfoot/Y\n",
    );
    assert!(stems(&d, "virt").is_empty());
    assert_eq!(stems(&d, "virts"), ["virt"]);
    assert!(stems(&d, "link").is_empty());
    assert!(!stems(&d, "footlink").is_empty());
}

#[test]
fn iconv_applies_before_and_oconv_after() {
    let d = load(
        "SET UTF-8\nICONV 1\nICONV \u{2019} '\nOCONV 1\nOCONV ' \u{2019}\nSFX M Y 1\nSFX M 0 's .\n",
        "1\nwalk/M\n",
    );
    assert_eq!(stems(&d, "walk\u{2019}s"), ["walk"]);
    assert_eq!(stems(&d, "walk's"), ["walk"]);
    let d = load("SET UTF-8\nOCONV 1\nOCONV a \u{e4}\n", "1\nhaus\n");
    assert_eq!(stems(&d, "haus"), ["h\u{e4}us"]);
    // Conversion patterns are matched once, longest first, and never
    // rescanned, so a pattern that maps to itself cannot loop.
    let d = load("ICONV 2\nICONV a aa\nICONV aa a\n", "1\naa\n");
    within(Duration::from_secs(5), "self-feeding ICONV", || stems(&d, &"a".repeat(10_000)));
}

/// A word made of one short entry repeated, which every compound mode can
/// split at every character.
#[test]
fn deep_compounds_stay_bounded() {
    let word = "ab".repeat(5_000);
    for affix in [
        "COMPOUNDFLAG Y\nCOMPOUNDMIN 1\n",
        "COMPOUNDFLAG Y\nCOMPOUNDMIN 1\nCHECKCOMPOUNDDUP\nCHECKCOMPOUNDTRIPLE\nSIMPLIFIEDTRIPLE\n",
        "COMPOUNDBEGIN Y\nCOMPOUNDMIDDLE Y\nCOMPOUNDEND Y\nCOMPOUNDMIN 1\n",
        "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE Y*\n",
        "COMPOUNDMIN 1\nCOMPOUNDRULE 2\nCOMPOUNDRULE Y*Y*Y*Y*Z\nCOMPOUNDRULE Y?Y?Y?Y*\n",
        "COMPOUNDFLAG Y\nCOMPOUNDMIN 1\nSFX S Y 1\nSFX S 0 b .\nPFX P Y 1\nPFX P 0 a .\n",
        "LANG hu_HU\nCOMPOUNDFLAG Y\nCOMPOUNDMIN 1\nCOMPOUNDWORDMAX 2\nCOMPOUNDSYLLABLE 1000 a\n",
    ] {
        let d = load(affix, "4\na/YSP\nb/YSP\nab/YSP\nba/YSP\n");
        let s = Stemmer::new(&d);
        for w in [&word[..], &word[..64], &word[..65], &"a".repeat(3_000)] {
            let found = within(Duration::from_secs(20), affix, || s.analyze(w));
            assert!(found.len() <= 2048, "{affix}: {} analyses", found.len());
        }
    }
}

#[test]
fn many_word_breaks_stay_bounded() {
    let d = load("BREAK 3\nBREAK -\nBREAK ^-\nBREAK -$\n", "1\na\n");
    let s = Stemmer::new(&d);
    for w in ["a-".repeat(5_000), "-".repeat(10_000), "a--".repeat(3_000)] {
        within(Duration::from_secs(20), "breaks", || s.stems(&w));
    }
}

#[test]
fn many_case_variants_stay_bounded() {
    // Every ß and SS doubles the sharp s variants of a CHECKSHARPS
    // dictionary, and dotted and dotless i do the same in Turkic.
    let d = load("CHECKSHARPS\nSET UTF-8\n", "1\nstra\u{df}e\n");
    let s = Stemmer::new(&d);
    within(Duration::from_secs(20), "sharp s", || s.stems(&"SS".repeat(5_000)));
    within(Duration::from_secs(20), "sharp s", || s.stems(&"\u{df}".repeat(5_000)));
    let d = load("LANG tr_TR\nSET UTF-8\n", "1\nistanbul\n");
    let s = Stemmer::new(&d);
    within(Duration::from_secs(20), "turkic", || s.stems(&"I\u{130}".repeat(5_000)));
}

#[test]
fn many_affixes_on_one_boundary_stay_bounded() {
    // A thousand suffixes that all strip nothing and add one letter, and a
    // word that every one of them could end.
    let mut affix = String::from("SFX A Y 1000\n");
    for i in 0..1000 {
        affix.push_str(&format!("SFX A 0 a/A {}\n", if i % 2 == 0 { "." } else { "a" }));
    }
    let d = load(&affix, "1\nb/A\n");
    let s = Stemmer::new(&d);
    let found = within(Duration::from_secs(20), "continuation chain", || s.stems(&format!("b{}", "a".repeat(200))));
    assert!(found.len() <= 2048);
}

#[test]
fn huge_words_and_lines_are_handled() {
    let d = load("SFX S Y 1\nSFX S 0 s .\n", "1\ncat/S\n");
    let s = Stemmer::new(&d);
    within(Duration::from_secs(20), "a megabyte word", || s.stems(&"x".repeat(1 << 20)));
    within(Duration::from_secs(20), "a megabyte of periods", || s.stems(&".".repeat(1 << 20)));
    let long = format!("1\n{}/S\n", "y".repeat(1 << 20));
    let d = load("SFX S Y 1\nSFX S 0 s .\n", &long);
    assert_eq!(Stemmer::new(&d).stems(&format!("{}s", "y".repeat(1 << 20))).len(), 1);
}

#[test]
fn conditions_that_cannot_be_read_fail() {
    for condition in ["[ab", "[", "[^", "a[b"] {
        error(&format!("SFX S Y 1\nSFX S 0 s {condition}\n"), "1\nx\n");
    }
    // A closing bracket alone is a literal character.
    let d = load("SFX S Y 1\nSFX S 0 s ]\n", "1\nx]/S\n");
    assert_eq!(stems(&d, "x]s"), ["x]"]);
    // An empty class matches nothing and is not an error in Java.
    let d = load("SFX S Y 1\nSFX S 0 s []\n", "1\nx/S\n");
    let _ = stems(&d, "xs");
}

#[test]
fn files_over_the_limit_are_refused() {
    let big = vec![b'#'; turbo_hunspell::MAX_FILE_BYTES + 1];
    assert!(matches!(Dictionary::from_bytes(&big, b"1\nx\n", LoadMode::Strict), Err(LoadError::TooLarge { .. })));
    assert!(matches!(Dictionary::from_bytes(b"", &big, LoadMode::Strict), Err(LoadError::TooLarge { .. })));
}
