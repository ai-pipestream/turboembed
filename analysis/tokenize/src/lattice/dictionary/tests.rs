use super::*;
use crate::lattice::LatticeTokenizer;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../goldens/data/fixtures/mecab-fixture-ja");
const MATRIX: &[u8] = include_bytes!("../../../../goldens/data/fixtures/mecab-fixture-ja/matrix.def");
const NOUNS: &[u8] = include_bytes!("../../../../goldens/data/fixtures/mecab-fixture-ja/A-nouns.csv");
const OTHERS: &[u8] = include_bytes!("../../../../goldens/data/fixtures/mecab-fixture-ja/b-others.csv");
const CHAR_DEF: &[u8] = include_bytes!("../../../../goldens/data/fixtures/mecab-fixture-ja/char.def");
const UNK_DEF: &[u8] = include_bytes!("../../../../goldens/data/fixtures/mecab-fixture-ja/unk.def");

fn from_files(
    matrix: &[u8],
    lexicons: &[(&str, &[u8])],
    char_def: &[u8],
    unk_def: &[u8],
) -> Result<MecabDictionary, DictionaryError> {
    let files = DictionaryFiles { matrix, lexicons, char_def, unk_def };
    MecabDictionary::from_files(files, WhitespaceMode::Unicode, DictionaryLimits::default())
}

fn fixture() -> MecabDictionary {
    from_files(MATRIX, &[("A-nouns.csv", NOUNS), ("b-others.csv", OTHERS)], CHAR_DEF, UNK_DEF).unwrap()
}

#[test]
fn a_directory_and_its_bytes_load_the_same_dictionary() {
    let dir = MecabDictionary::load(FIXTURE, WhitespaceMode::Unicode).unwrap();
    let mem = fixture();
    assert_eq!(dir.sha256(), mem.sha256());
    assert_eq!(dir.lexicon_len(), mem.lexicon_len());
    assert_eq!(dir.surfaces(), mem.surfaces());
    assert_eq!(dir.pos_tags().len(), mem.pos_tags().len());
    assert_eq!(dir.tag(EntryId(0)), mem.tag(EntryId(0)));
}

#[test]
fn missing_files_are_named_in_load_order() {
    let tmp = std::env::temp_dir().join(format!("turbo-lattice-missing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let load = || MecabDictionary::load(&tmp, WhitespaceMode::Unicode).unwrap_err();
    assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::Matrix)));
    std::fs::write(tmp.join("matrix.def"), MATRIX).unwrap();
    assert!(matches!(load(), DictionaryError::NoLexiconEntries));
    std::fs::write(tmp.join("A-nouns.csv"), NOUNS).unwrap();
    assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::CharDef)));
    std::fs::write(tmp.join("char.def"), CHAR_DEF).unwrap();
    assert!(matches!(load(), DictionaryError::Missing(DictionaryFile::UnkDef)));
    // A path that exists but is not a file is an I/O error.
    std::fs::create_dir(tmp.join("unk.def")).unwrap();
    assert!(matches!(load(), DictionaryError::Io { file: DictionaryFile::UnkDef, .. }));
    std::fs::remove_dir(tmp.join("unk.def")).unwrap();
    std::fs::write(tmp.join("unk.def"), UNK_DEF).unwrap();
    // A file named like a lexicon but not ending in .csv is not read.
    std::fs::write(tmp.join("z.csv.bak"), b"not, a lexicon").unwrap();
    assert!(MecabDictionary::load(&tmp, WhitespaceMode::Unicode).is_ok());
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn bytes_that_are_not_utf8_are_refused_with_their_offset() {
    let err = from_files(b"1 1\n0 0 \xff\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
    assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Matrix, offset: 8 }));
    let err = from_files(MATRIX, &[("a.csv", b"\xe6\x9d")], CHAR_DEF, UNK_DEF).unwrap_err();
    assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Lexicon(ref n), offset: 0 } if n == "a.csv"));
    let err = from_files(MATRIX, &[("a.csv", NOUNS)], b"\xc0", UNK_DEF).unwrap_err();
    assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::CharDef, .. }));
    let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, b"\x80").unwrap_err();
    assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::UnkDef, .. }));
}

/// Deviation: where unk.def names several undefined categories, the
/// Java reports the first in the iteration order of a `HashMap`, and
/// the port the first in file order.
#[test]
fn unk_def_names_only_defined_categories() {
    let unk = b"DEFAULT,0,0,0,x\nNOSUCH,0,0,0,x\n";
    let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, unk).unwrap_err();
    assert!(matches!(err, DictionaryError::UnknownCategory(ref n) if n == "NOSUCH"));
    let unk = b"ZZZ,0,0,0,x\nDEFAULT,0,0,0,x\nAAA,0,0,0,x\n";
    let err = from_files(MATRIX, &[("a.csv", NOUNS)], CHAR_DEF, unk).unwrap_err();
    assert!(matches!(err, DictionaryError::UnknownCategory(ref n) if n == "ZZZ"));
}

/// Deviation: the port checks that a whole file is UTF-8 before it
/// reads a line of it; the Java decodes as it reads, so a malformed
/// line before the bad bytes is the Java's error and the bad bytes are
/// the port's.
#[test]
fn a_file_is_checked_for_utf8_before_its_lines_are_read() {
    let err = from_files(b"1 1\nx\n\xff\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
    assert!(matches!(err, DictionaryError::Utf8 { file: DictionaryFile::Matrix, offset: 6 }));
    let err = from_files(b"1 1\nx\n", &[("a.csv", NOUNS)], CHAR_DEF, UNK_DEF).unwrap_err();
    assert!(matches!(err, DictionaryError::MatrixLine { line: 2 }));
}

/// Deviation: OpenNLP keeps any number of distinct part-of-speech
/// columns; a [`Tag`] numbers at most 65536 of them, so more are
/// refused rather than wrapped.
#[test]
fn more_tags_than_a_tag_can_number_are_refused() {
    let mut lexicon = String::new();
    for i in 0..=65536u32 {
        lexicon.push_str(&format!("w{i},0,0,0,t{i}\n"));
    }
    let char_def = b"DEFAULT 0 1 0\n";
    let unk = b"DEFAULT,0,0,0,t0\n";
    let err = from_files(b"1 1\n0 0 0\n", &[("a.csv", lexicon.as_bytes())], char_def, unk).unwrap_err();
    assert!(matches!(err, DictionaryError::TooManyTags));
    let fits: String = lexicon.lines().take(65535).map(|l| format!("{l}\n")).collect();
    let d = from_files(b"1 1\n0 0 0\n", &[("a.csv", fits.as_bytes())], char_def, unk).unwrap();
    assert_eq!(d.pos_tags().len(), 65535);
}

#[test]
fn the_entry_limit_counts_the_lexicon_and_unk_def_apart() {
    // The fixture's A-nouns.csv has 20 entries and its unk.def 13.
    let files = DictionaryFiles { matrix: MATRIX, lexicons: &[("a.csv", NOUNS)], char_def: CHAR_DEF, unk_def: UNK_DEF };
    let limits = |max_entries| DictionaryLimits { max_entries, ..DictionaryLimits::default() };
    let err = MecabDictionary::from_files(files, WhitespaceMode::Unicode, limits(19)).unwrap_err();
    assert!(matches!(err, DictionaryError::TooManyEntries { limit: 19 }));
    assert!(MecabDictionary::from_files(files, WhitespaceMode::Unicode, limits(20)).is_ok());
}

/// Mutated fixture files load or fail with an error; whatever loads
/// tokenizes text without a panic.
#[test]
fn mutated_files_never_panic() {
    let originals: [&[u8]; 5] = [MATRIX, NOUNS, OTHERS, CHAR_DEF, UNK_DEF];
    let texts = ["東京都に行く", "ｶﾀｶﾅ 123 abc", "𠀋😀αβγ", "  ", "一二三。", "\u{30FC}\u{30FC}x"];
    let limits = DictionaryLimits { max_entries: 10_000, max_matrix_cells: 10_000 };
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut loaded, mut refused) = (0, 0);
    for _ in 0..3000 {
        let mut files: Vec<Vec<u8>> = originals.iter().map(|f| f.to_vec()).collect();
        for _ in 0..1 + next() % 4 {
            let f = &mut files[(next() % 5) as usize];
            let at = if f.is_empty() { 0 } else { (next() as usize) % f.len() };
            match next() % 6 {
                0 if !f.is_empty() => f[at] = next() as u8,
                1 => f.insert(at, b"0123456789-,. \n\t"[(next() % 16) as usize]),
                2 if !f.is_empty() => {
                    f.remove(at);
                }
                3 => f.truncate(at),
                4 => f.splice(at..at, *b"-2147483648").for_each(drop),
                _ => f.splice(at..at, "\u{3000}東".bytes()).for_each(drop),
            }
        }
        let lexicons = [("A-nouns.csv", &files[1][..]), ("b-others.csv", &files[2][..])];
        let d = DictionaryFiles { matrix: &files[0], lexicons: &lexicons, char_def: &files[3], unk_def: &files[4] };
        match MecabDictionary::from_files(d, WhitespaceMode::Unicode, limits) {
            Ok(d) => {
                loaded += 1;
                let t = LatticeTokenizer::new(d, WhitespaceMode::Unicode);
                let mut out = Vec::new();
                for text in texts {
                    let _ = t.analyze(text, &mut out);
                }
            }
            Err(e) => {
                refused += 1;
                let _ = e.to_string();
            }
        }
    }
    assert!(loaded > 0 && refused > 0, "loaded {loaded}, refused {refused}");
}
