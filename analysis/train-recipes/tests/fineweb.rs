//! The web data of the language detector recipe on Parquet files written
//! for the test, one per file the recipe reads: the ranges fetched round
//! by round as `fineweb-ranges` fetches them, recorded, checked, and the
//! windows chosen from them; and the web language detector recipe end to
//! end on those files and the Tatoeba export of `tests/data/tatoeba`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_eval::model_gate::{Counts, ModelGateRecord, ModelId, PairedUnit, Task};
use turbo_train_recipes::Error;
use turbo_train_recipes::card::{GATE_FILE, GateCard};
use turbo_train_recipes::fineweb::{
    self, Avoid, FileChoice, Overlap, Side, Span, Store, WEB, WebError, WebFile, WebRecipe, WebSplit, check_ranges,
    needed, plan_web, record,
};
use turbo_train_recipes::fineweb::{Source, WebLanguage};
use turbo_train_recipes::langdetect::{self, CARD_FILE, LanguagePair, LanguageRecipe, SENTENCES_FILE, Split};
use turbo_train_recipes::langdetect_web::{
    self, COMPARE_TATOEBA_FILE, COMPARE_WEB_FILE, DOCUMENTS_FILE, LANGDETECT_WEB, MODEL_ID, WEB_GATE_FILE, WebInput,
    WebLanguageCard, WebLanguageRecipe, write_web_language_dir,
};
use turbo_train_recipes::release::{self, FINEWEB_MODELS};
use turbo_train_recipes::tatoeba::Export;
use turbo_train_recipes::{RecordedReference, Reference};
use turbo_zoo::Sha256;

/// Thrift's compact protocol, as far as a footer and page headers need it.
#[derive(Default)]
struct Thrift {
    out: Vec<u8>,
    last: Vec<i16>,
}

impl Thrift {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.out.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.out.push(v as u8);
    }

    fn field(&mut self, id: i16, ty: u8) {
        let last = self.last.last_mut().unwrap();
        self.out.push(((id - *last) as u8) << 4 | ty);
        *last = id;
    }

    fn int(&mut self, id: i16, ty: u8, v: i64) {
        self.field(id, ty);
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }

    fn bytes(&mut self, id: i16, b: &[u8]) {
        self.field(id, 8);
        self.varint(b.len() as u64);
        self.out.extend(b);
    }

    fn begin(&mut self, field: Option<i16>) {
        if let Some(id) = field {
            self.field(id, 12);
        }
        self.last.push(0);
    }

    fn end(&mut self) {
        self.out.push(0);
        self.last.pop();
    }

    fn list(&mut self, id: i16, ty: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.out.push((n as u8) << 4 | ty);
        } else {
            self.out.push(0xf0 | ty);
            self.varint(n as u64);
        }
    }
}

const COLUMNS: [&str; 3] = ["text", "id", "url"];

/// A Parquet file of required string columns `text`, `id` and `url`, in
/// row groups of `group` rows, each column one uncompressed PLAIN page.
fn parquet(rows: &[[String; 3]], group: usize) -> Vec<u8> {
    let mut file = b"PAR1".to_vec();
    let mut groups = Vec::new();
    for g in rows.chunks(group) {
        let mut chunks = Vec::new();
        for c in 0..COLUMNS.len() {
            let mut values = Vec::new();
            for r in g {
                values.extend((r[c].len() as u32).to_le_bytes());
                values.extend(r[c].as_bytes());
            }
            let mut h = Thrift::default();
            h.begin(None);
            h.int(1, 5, 0);
            h.int(2, 5, values.len() as i64);
            h.int(3, 5, values.len() as i64);
            h.begin(Some(5));
            h.int(1, 5, g.len() as i64);
            h.int(2, 5, 0);
            h.int(3, 5, 3);
            h.int(4, 5, 3);
            h.end();
            h.end();
            let offset = file.len();
            file.extend(&h.out);
            file.extend(&values);
            chunks.push((offset, file.len() - offset));
        }
        groups.push((g.len(), chunks));
    }
    let mut m = Thrift::default();
    m.begin(None);
    m.int(1, 5, 1);
    m.list(2, 12, COLUMNS.len() + 1);
    m.begin(None);
    m.bytes(4, b"schema");
    m.int(5, 5, COLUMNS.len() as i64);
    m.end();
    for name in COLUMNS {
        m.begin(None);
        m.int(1, 5, 6);
        m.int(3, 5, 0);
        m.bytes(4, name.as_bytes());
        m.end();
    }
    m.int(3, 6, rows.len() as i64);
    m.list(4, 12, groups.len());
    for (n, chunks) in &groups {
        m.begin(None);
        m.list(1, 12, chunks.len());
        for (name, &(offset, len)) in COLUMNS.iter().zip(chunks) {
            m.begin(None);
            m.int(2, 6, offset as i64);
            m.begin(Some(3));
            m.int(1, 5, 6);
            m.list(2, 5, 1);
            m.varint(0);
            m.list(3, 8, 1);
            m.varint(name.len() as u64);
            m.out.extend(name.as_bytes());
            m.int(4, 5, 0);
            m.int(5, 6, *n as i64);
            m.int(6, 6, len as i64);
            m.int(7, 6, len as i64);
            m.int(9, 6, offset as i64);
            m.end();
            m.end();
        }
        m.int(2, 6, chunks.iter().map(|c| c.1 as i64).sum());
        m.int(3, 6, *n as i64);
        m.end();
    }
    m.end();
    file.extend(&m.out);
    file.extend((m.out.len() as u32).to_le_bytes());
    file.extend(b"PAR1");
    file
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The recipe's languages with the test's split: 6 training and 3
/// held-out windows per language, of at most 12 characters, from
/// documents of at least 5; at first at most one in three of a file's
/// documents from one host; Tatoeba sentences of 16 characters or more
/// looked for in the windows.
const SPLIT: WebRecipe = WebRecipe { split: SMALL, ..WEB };

const SMALL: WebSplit = WebSplit {
    training: 6,
    held_out: 3,
    window: 12,
    min_chars: 5,
    host_share: 3,
    rows_per_document: 2,
    footer_prefix: 600,
    overlap_min_chars: 16,
};

/// The rows of one file: `n` documents, in row groups of 4. Row 1 is too
/// short, row 2 repeats row 0's window, rows 3 to 7 share one host, and
/// every held-out file's row 0 is the training file's row 0 of its
/// language.
fn rows(language: &str, path: &str, n: usize) -> Vec<[String; 3]> {
    (0..n)
        .map(|i| {
            let text = match i {
                0 => format!("{language} first document, long enough"),
                1 => "tiny".to_owned(),
                2 => format!("{language} first document, with another end"),
                _ => format!("{i:03} {language} {path} document and more"),
            };
            let host = if (3..8).contains(&i) { "same.example".to_owned() } else { format!("host{i}.example") };
            [text, format!("<urn:{path}:{i}>"), format!("https://www.{host}/{i}")]
        })
        .collect()
}

/// A file per want of `split`, served from memory, and its list entry.
fn served(split: &WebRecipe) -> (Vec<WebFile>, HashMap<String, Vec<u8>>) {
    served_with(split, |tatoeba, side| {
        let n = if tatoeba == "min" { 6 } else { 30 };
        // The held-out files' first rows are their training files'.
        rows(tatoeba, if side == Side::HeldOut { "test" } else { "train" }, n)
    })
}

/// A file per want of `split` holding the rows `rows` gives for its
/// language and side, served from memory, and its list entry.
fn served_with(
    split: &WebRecipe,
    rows: impl Fn(&str, Side) -> Vec<[String; 3]>,
) -> (Vec<WebFile>, HashMap<String, Vec<u8>>) {
    let mut files = Vec::new();
    let mut bytes = HashMap::new();
    for w in split.wants() {
        let path = w.path();
        let b = parquet(&rows(split.languages[w.language].tatoeba, w.side), 4);
        files.push(WebFile {
            repo: w.repo().to_owned(),
            revision: w.source.dataset().revision.to_owned(),
            path: path.clone(),
            size: b.len() as u64,
            sha256: Sha256::of(&b).to_string(),
        });
        bytes.insert(format!("{} {path}", w.repo()), b);
    }
    (files, bytes)
}

/// Fetches what `needed` lists, round by round, into `dir`.
fn fetch_all(dir: &Path, files: &[WebFile], split: &WebRecipe, bytes: &HashMap<String, Vec<u8>>) -> usize {
    let store = Store::new(dir, None);
    let mut rounds = 0;
    loop {
        let need = needed(files, split, &store).unwrap();
        if need.is_empty() {
            return rounds;
        }
        rounds += 1;
        for (f, span) in need {
            let b = &bytes[&format!("{} {}", f.repo, f.path)];
            let at = store.file(f, span);
            fs::create_dir_all(at.parent().unwrap()).unwrap();
            fs::write(at, &b[span.start as usize..span.end as usize]).unwrap();
        }
    }
}

/// Nothing to avoid, for the recipe's languages.
fn none() -> Avoid<'static> {
    Avoid::none(WEB.languages.len())
}

fn choice<'a>(plan: &'a [FileChoice], tatoeba: &str, side: Side) -> Vec<&'a FileChoice> {
    plan.iter().filter(|c| c.tatoeba == tatoeba && c.side == side).collect()
}

#[test]
fn ranges_are_fetched_recorded_checked_and_read() {
    let dir = scratch("fineweb-ranges");
    let (files, bytes) = served(&SPLIT);
    assert_eq!(files.len(), 2 * 102);
    // The last 8 bytes, the footer's first bytes, then the head.
    assert_eq!(fetch_all(&dir, &files, &SPLIT, &bytes), 3);
    let made = record(&files, &SPLIT, &Store::new(&dir, None)).unwrap();
    assert_eq!(made.len(), 3 * files.len());
    for (f, r) in files.iter().zip(made.chunks(3)) {
        let b = &bytes[&format!("{} {}", f.repo, f.path)];
        assert_eq!(r[0].span, Span { start: f.size - 8, end: f.size });
        assert_eq!(r[1].span.end, r[0].span.start.min(r[1].span.start + SMALL.footer_prefix));
        assert_eq!(r[2].span.start, 0);
        for x in r {
            assert_eq!(x.sha256, Sha256::of(&b[x.span.start as usize..x.span.end as usize]).to_string());
        }
    }
    let listed = fineweb::parse_ranges("r", &fineweb::write_ranges(&made)).unwrap();
    check_ranges(&listed, &made).unwrap();

    let plan = plan_web(&files, &SPLIT, &Store::new(&dir, Some(&listed)), &none()).unwrap();
    assert!(plan.files.iter().all(|c| c.excluded_host == 0 && c.tatoeba_overlap == 0));
    // A language of one source: 6 training documents from 12 rows, three
    // row groups; row 1 too short, row 2 a repeat; at most 2 of rows 3 to
    // 7 at first, so rows 0, 3, 4, 8, 9 and 10.
    let afr = choice(&plan.files, "afr", Side::Training);
    assert_eq!(afr.len(), 1);
    let a = afr[0];
    assert_eq!((a.wanted, a.file_rows, a.file_row_groups, a.row_groups_read, a.rows_read), (6, 30, 8, 3, 12));
    assert_eq!((a.too_short, a.duplicates, a.host_deferred, a.host_taken_later, a.chosen), (1, 1, 3, 0, 6));
    let afr_rows: Vec<u64> = plan.training.iter().filter(|s| s.language == 0).map(|s| s.row).collect();
    assert_eq!(afr_rows, [0, 3, 4, 8, 9, 10]);
    let first = plan.training.iter().find(|s| s.language == 0).unwrap();
    assert_eq!(first.window, "afr first do");
    assert_eq!(first.id, "<urn:train:0>");
    // Held out: row 0 is a training window and row 2 repeats it, so both
    // are passed over.
    let h = choice(&plan.files, "afr", Side::HeldOut)[0];
    assert_eq!((h.wanted, h.rows_read, h.too_short, h.duplicates, h.chosen), (3, 8, 1, 2, 3));
    let held: Vec<u64> = plan.held_out.iter().filter(|s| s.language == 0).map(|s| s.row).collect();
    assert_eq!(held, [3, 4, 5]);
    // Two sources share a language's documents.
    let srp = choice(&plan.files, "srp", Side::Training);
    assert_eq!(
        srp.iter().map(|c| (c.path.as_str(), c.wanted)).collect::<Vec<_>>(),
        [("data/srp_Cyrl/train/000_00000.parquet", 3), ("data/srp_Latn/train/000_00000.parquet", 3),]
    );
    // The second source's windows are the first's, so its first rows are
    // repeats.
    assert_eq!(srp[1].duplicates, 4);
    // A file with fewer rows than wanted is read whole and gives what it
    // has, the host's share filled from the documents passed over.
    let min = choice(&plan.files, "min", Side::Training)[0];
    assert_eq!((min.file_rows, min.row_groups_read, min.rows_read, min.chosen), (6, 2, 6, 4));
    assert_eq!((min.host_deferred, min.host_taken_later), (1, 1));
    assert_eq!(
        plan.training.len(),
        plan.files.iter().filter(|c| c.side == Side::Training).map(|c| c.chosen).sum::<usize>()
    );
    let doc_list = String::from_utf8(fineweb::documents_file(&plan, &files)).unwrap();
    assert_eq!(doc_list.lines().count(), plan.training.len());
    assert!(doc_list.contains("HuggingFaceFW/fineweb-2\tdata/afr_Latn/train/000_00000.parquet\t0\t<urn:train:0>\n"));
}

#[test]
fn a_range_that_differs_or_is_missing_is_refused() {
    let dir = scratch("fineweb-refused");
    let (files, bytes) = served(&SPLIT);
    fetch_all(&dir, &files, &SPLIT, &bytes);
    let made = record(&files, &SPLIT, &Store::new(&dir, None)).unwrap();
    let store = Store::new(&dir, Some(&made));
    // One byte of a head changed.
    let head = &made[2];
    let head_file = files.iter().find(|f| f.repo == head.repo && f.path == head.path).unwrap();
    let at = store.file(head_file, head.span);
    assert!(at.to_string_lossy().contains(&head_file.revision), "{}", at.display());
    let mut b = fs::read(&at).unwrap();
    b[100] ^= 1;
    fs::write(&at, &b).unwrap();
    match plan_web(&files, &SPLIT, &store, &none()) {
        Err(WebError::Digest { path, expected, .. }) => assert_eq!((path, expected), (at.clone(), head.sha256.clone())),
        other => panic!("{other:?}"),
    }
    // Removed.
    fs::remove_file(&at).unwrap();
    assert_eq!(plan_web(&files, &SPLIT, &store, &none()).unwrap_err(), WebError::Missing { path: at });
    // A range the list does not have.
    let fewer: Vec<_> = made.iter().skip(1).cloned().collect();
    let e = plan_web(&files, &SPLIT, &Store::new(&dir, Some(&fewer)), &none()).unwrap_err();
    assert!(e.to_string().contains("which the recipe reads, is not listed"), "{e}");
    // A file the recipe does not read, or one it reads missing.
    let mut extra = files.clone();
    extra.push(WebFile { path: "data/xyz_Latn/train/000_00000.parquet".to_owned(), ..files[0].clone() });
    let e = record(&extra, &SPLIT, &Store::new(&dir, None)).unwrap_err();
    assert!(e.to_string().ends_with("is listed, and the recipe does not read it"), "{e}");
    let e = record(&files[1..], &SPLIT, &Store::new(&dir, None)).unwrap_err();
    assert!(e.to_string().ends_with("which the recipe reads, is not listed"), "{e}");
}

#[test]
fn a_footer_prefix_too_short_for_the_rows_is_refused() {
    let dir = scratch("fineweb-prefix");
    let (files, bytes) = served(&SPLIT);
    // The schema and one row group fit in 280 bytes of these footers, and
    // three are needed.
    let split = WebRecipe { split: WebSplit { footer_prefix: 280, ..SMALL }, ..WEB };
    let store = Store::new(&dir, None);
    let e = (0..3).find_map(|_| match needed(&files, &split, &store) {
        Err(e) => Some(e),
        Ok(need) => {
            for (f, span) in need {
                let b = &bytes[&format!("{} {}", f.repo, f.path)];
                let at = store.file(f, span);
                fs::create_dir_all(at.parent().unwrap()).unwrap();
                fs::write(at, &b[span.start as usize..span.end as usize]).unwrap();
            }
            None
        }
    });
    let e = e.expect("refused").to_string();
    assert!(e.contains("row groups") && e.contains("rows, and 12 are wanted"), "{e}");
}

/// `analysis/goldens/data/fixtures/langdetect-fixture-rust.bin`, as
/// tests/langdetect.rs records it: the reference and the compared model
/// of the test recipe, for its nine languages.
const FIXTURE: RecordedReference = RecordedReference {
    id: "langdetect-fixture-rust",
    file: "langdetect-fixture-rust.bin",
    url: "analysis/goldens/data/fixtures/langdetect-fixture-rust.bin",
    size: 7503,
    sha256: "bfde716d402a696348045c2225428bbeb53f9e40f6807a5716396852a2a03ccb",
};

const fn same(code: &'static [&'static str; 1]) -> LanguagePair {
    LanguagePair { tatoeba: code[0], reference: code }
}

/// The fixture's languages, as tests/langdetect.rs has them.
const TEST_LANGUAGES: [LanguagePair; 9] = [
    same(&["deu"]),
    same(&["ell"]),
    same(&["eng"]),
    same(&["fra"]),
    same(&["ita"]),
    same(&["nld"]),
    same(&["rus"]),
    same(&["spa"]),
    same(&["tur"]),
];

const fn web(tatoeba: &'static str, sources: &'static [Source]) -> WebLanguage {
    WebLanguage { tatoeba, sources }
}

/// The same languages' web sources.
const TEST_WEB_LANGUAGES: [WebLanguage; 9] = [
    web("deu", &[Source::FineWeb2("deu_Latn")]),
    web("ell", &[Source::FineWeb2("ell_Grek")]),
    web("eng", &[Source::FineWeb]),
    web("fra", &[Source::FineWeb2("fra_Latn")]),
    web("ita", &[Source::FineWeb2("ita_Latn")]),
    web("nld", &[Source::FineWeb2("nld_Latn")]),
    web("rus", &[Source::FineWeb2("rus_Cyrl")]),
    web("spa", &[Source::FineWeb2("spa_Latn")]),
    web("tur", &[Source::FineWeb2("tur_Latn")]),
];

/// The web recipe at the test's scale: the Tatoeba side as
/// tests/langdetect.rs runs it (four sentences held out and five trained
/// on per language, which leaves out Turkish's three), and six web
/// windows of at most 40 characters trained on per language, four held
/// out, enough units for the gate.
const TEST_RECIPE: WebLanguageRecipe = WebLanguageRecipe {
    model_id: "turbo-mul-test-web-langdetect",
    tatoeba: LanguageRecipe {
        model_id: "turbo-mul-test-web-langdetect",
        reference: Reference::Recorded(FIXTURE),
        languages: &TEST_LANGUAGES,
        split: Split { seed: 429, held_out: 4, max_training: 5, min_sentences: 8 },
        iterations: 30,
        cutoff: 1,
    },
    web: WebRecipe {
        languages: &TEST_WEB_LANGUAGES,
        split: WebSplit {
            training: 6,
            held_out: 4,
            window: 40,
            min_chars: 10,
            host_share: 3,
            rows_per_document: 2,
            footer_prefix: 2_000,
            overlap_min_chars: 16,
        },
    },
    compared: Reference::Recorded(FIXTURE),
};

/// Common words of each test language, which the documents are made of.
fn words(tatoeba: &str) -> &'static [&'static str] {
    match tatoeba {
        "deu" => &["der", "und", "nicht", "ist", "das", "mit", "auch", "sich"],
        "ell" => &["και", "του", "είναι", "στο", "δεν", "για", "με", "την"],
        "eng" => &["the", "and", "with", "this", "that", "from", "have", "were"],
        "fra" => &["les", "est", "dans", "pour", "pas", "avec", "une", "sont"],
        "ita" => &["della", "che", "non", "sono", "anche", "per", "una", "gli"],
        "nld" => &["het", "een", "niet", "zijn", "voor", "ook", "maar", "wordt"],
        "rus" => &["что", "это", "как", "его", "она", "так", "все", "был"],
        "spa" => &["que", "los", "por", "una", "para", "con", "las", "está"],
        "tur" => &["bir", "için", "değil", "çok", "daha", "olarak", "gibi", "ancak"],
        other => panic!("{other}"),
    }
}

/// Twelve documents of `tatoeba`'s words, each with its own order of
/// them; the held-out ones start further on.
fn word_rows(tatoeba: &str, side: Side) -> Vec<[String; 3]> {
    let w = words(tatoeba);
    let first = if side == Side::HeldOut { 50 } else { 0 };
    (first..first + 12)
        .map(|i: usize| {
            let text: Vec<&str> = (0..12).map(|k| w[(i * 7 + k * (i % 5 + 1)) % w.len()]).collect();
            [format!("{i:02} {}", text.join(" ")), format!("<urn:{i}>"), format!("https://site{}.example/{i}", i % 4)]
        })
        .collect()
}

fn fixture() -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures").join(FIXTURE.file);
    fs::read(path).unwrap()
}

#[test]
fn the_web_recipe_trains_gates_and_compares_end_to_end() {
    let dir = scratch("web-recipe");
    let recipe = TEST_RECIPE;
    let (files, bytes) = served_with(&recipe.web, word_rows);
    fetch_all(&dir, &files, &recipe.web, &bytes);
    let ranges = record(&files, &recipe.web, &Store::new(&dir, None)).unwrap();
    let store = Store::new(&dir, Some(&ranges));
    let web = WebInput::checked(recipe.web, &files, &ranges, false, &store).unwrap();
    let export = Export::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tatoeba")).unwrap();
    let fixture = fixture();
    let outcome = langdetect_web::run(&recipe, &export, false, &web, &fixture, &fixture).unwrap();
    let again = langdetect_web::run(&recipe, &export, false, &web, &fixture, &fixture).unwrap();
    assert_eq!(outcome.trained.bytes, again.trained.bytes);
    assert_eq!(outcome.trained.documents_file, again.trained.documents_file);

    let card = &outcome.card;
    // Turkish has too few sentences, and is trained on its windows alone.
    assert_eq!(card.languages, ["deu", "ell", "eng", "fra", "ita", "nld", "rus", "spa", "tur"]);
    assert_eq!(card.excluded_languages.len(), 1);
    assert_eq!((card.training.sentences, card.training.windows, card.training.parameters.events), (40, 54, 94));
    assert_eq!((card.held_out_sentences, card.held_out_windows), (32, 36));
    assert_eq!(card.training_documents.documents, 54);
    assert_eq!(String::from_utf8(outcome.trained.documents_file.clone()).unwrap().lines().count(), 54);
    assert!(!card.web.recorded && !card.export.recorded);
    assert_eq!(card.web.files.len(), 18);
    assert_eq!(card.web.lists_sha256, fineweb::lists_digest(&files, &ranges));

    let s = &outcome.scored;
    for r in [&s.tatoeba_gate, &s.web_gate, &s.compare_tatoeba, &s.compare_web] {
        r.check().unwrap();
        assert_eq!(r.candidate.id, recipe.model_id);
        assert_eq!(r.candidate.sha256, card.file.sha256);
    }
    assert_eq!((s.tatoeba_gate.units.len(), s.web_gate.units.len()), (32, 36));
    assert_eq!((s.compare_tatoeba.units.len(), s.compare_web.units.len()), (32, 36));
    assert_eq!(s.compare_web.reference.id, FIXTURE.id);
    // The Tatoeba gate's set is the one the Tatoeba recipe gates on.
    let plain =
        langdetect::run(&LanguageRecipe { model_id: "plain", ..recipe.tatoeba }, &export, false, &fixture).unwrap();
    assert_eq!(s.tatoeba_gate.set, plain.gate.set);
    assert_eq!(s.tatoeba_gate.set.name, recipe.tatoeba.held_out_name(&export.digest()));
    // The compared model's records score the gates' units with its own
    // answers: the trained model's side is the same.
    let candidate = |r: &turbo_eval::model_gate::ModelGateRecord| -> Vec<u64> {
        r.units.iter().map(|u| u.candidate.true_positives).collect()
    };
    assert_eq!(candidate(&s.tatoeba_gate), candidate(&s.compare_tatoeba));
    assert_eq!(candidate(&s.web_gate), candidate(&s.compare_web));
    let right: u64 = candidate(&s.web_gate).iter().sum();
    assert_eq!(right as usize, card.web_per_language.iter().map(|l| l.candidate_correct).sum::<usize>());
    let compared: u64 = s.compare_web.units.iter().map(|u| u.reference.true_positives).sum();
    assert_eq!(compared as usize, card.web_per_language.iter().map(|l| l.compared_correct).sum::<usize>());
    // The fixture is both the reference and the compared model, so their
    // answers on the windows are the same.
    let reference: u64 = s.web_gate.units.iter().map(|u| u.reference.true_positives).sum();
    assert_eq!(reference, compared);

    let out = dir.join("models");
    write_web_language_dir(&out, &outcome).unwrap();
    let model_dir = out.join(recipe.model_id);
    for f in
        [SENTENCES_FILE, DOCUMENTS_FILE, CARD_FILE, GATE_FILE, WEB_GATE_FILE, COMPARE_TATOEBA_FILE, COMPARE_WEB_FILE]
    {
        assert!(model_dir.join(f).is_file(), "{f}");
    }
    assert_eq!(fs::read(model_dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
    assert_eq!(WebLanguageCard::read(&model_dir.join(CARD_FILE)).unwrap(), *card);
    assert!(write_web_language_dir(&out, &outcome).is_err(), "the directory exists");

    // No release: no recipe trains it, and under the recipe's id it was
    // gated against another reference.
    let url = "https://example.org/m.bin";
    assert!(matches!(card.artifact(url, url, url), Err(Error::NotShippable { .. })));
    let renamed = WebLanguageCard { id: MODEL_ID.to_owned(), ..card.clone() };
    match renamed.artifact(url, url, url) {
        Err(Error::NotShippable { reason, .. }) => {
            assert_eq!(reason, "it was not gated against its recipe's reference")
        }
        other => panic!("{other:?}"),
    }
    // Under the recipe's id and reference, the refusals that follow, then
    // an entry.
    let reference = LANGDETECT_WEB.tatoeba.reference;
    let mut card = renamed;
    for g in [&mut card.gate, &mut card.web_gate] {
        g.reference = reference.id().to_owned();
        g.reference_sha256 = reference.sha256().unwrap();
        g.pass = true;
    }
    card.web_gate.pass = false;
    let refused = |c: &WebLanguageCard| match c.artifact(url, url, url) {
        Err(Error::NotShippable { reason, .. }) => reason,
        other => panic!("{other:?}"),
    };
    assert_eq!(refused(&card), "it failed a gate");
    card.web_gate.pass = true;
    assert_eq!(refused(&card), "it was trained on a Tatoeba export that is not the recorded one");
    card.export.recorded = true;
    assert_eq!(refused(&card), "it was trained on web ranges that are not the recorded ones");
    card.web.recorded = true;
    let entry = card.artifact(url, url, url).unwrap();
    let names: Vec<&str> = entry.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(names, [card.file.name.as_str(), SENTENCES_FILE, DOCUMENTS_FILE]);
}

/// The release's entry is made from the model's directory with its three
/// files, each checked against the card, and with its two gate records
/// and two comparison records checked against the card. The test model
/// stands in for the release's: its card and records are rewritten to
/// name the release's model and reference, with passing gates, and to
/// come from the recorded export and web ranges.
#[test]
fn the_release_entry_checks_the_model_its_records_and_its_training_files() {
    let dir = scratch("web-release");
    let recipe = TEST_RECIPE;
    let (files, bytes) = served_with(&recipe.web, word_rows);
    fetch_all(&dir, &files, &recipe.web, &bytes);
    let ranges = record(&files, &recipe.web, &Store::new(&dir, None)).unwrap();
    let store = Store::new(&dir, Some(&ranges));
    let web = WebInput::checked(recipe.web, &files, &ranges, false, &store).unwrap();
    let export = Export::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tatoeba")).unwrap();
    let fixture = fixture();
    let mut outcome = langdetect_web::run(&recipe, &export, false, &web, &fixture, &fixture).unwrap();

    let card = &mut outcome.card;
    card.id = MODEL_ID.to_owned();
    card.file.name = format!("{MODEL_ID}.bin");
    card.export.recorded = true;
    card.web.recorded = true;
    let candidate = ModelId { id: MODEL_ID.to_owned(), sha256: card.file.sha256.clone() };
    let reference = LANGDETECT_WEB.tatoeba.reference;
    let reference = ModelId { id: reference.id().to_owned(), sha256: reference.sha256().unwrap() };
    let right = PairedUnit { candidate: Counts::accuracy(1, 1), reference: Counts::accuracy(1, 1) };
    let passing = |r: &ModelGateRecord, g: &mut GateCard| {
        let units = vec![right; r.units.len()];
        let r =
            ModelGateRecord::new(Task::LanguageDetector, r.set.clone(), candidate.clone(), reference.clone(), units)
                .unwrap();
        assert!(r.result.pass);
        g.reference = r.reference.id.clone();
        g.reference_sha256 = r.reference.sha256.clone();
        g.candidate_score = r.result.candidate.score;
        g.reference_score = r.result.reference.score;
        g.difference_low = r.result.difference.low;
        g.difference_high = r.result.difference.high;
        g.pass = true;
        r
    };
    let s = &mut outcome.scored;
    s.tatoeba_gate = passing(&s.tatoeba_gate, &mut card.gate);
    s.web_gate = passing(&s.web_gate, &mut card.web_gate);
    // The comparisons keep their units, so their scores and the card's
    // stay the same.
    let renamed = |r: &ModelGateRecord| {
        ModelGateRecord::new(
            Task::LanguageDetector,
            r.set.clone(),
            candidate.clone(),
            r.reference.clone(),
            r.units.clone(),
        )
        .unwrap()
    };
    s.compare_tatoeba = renamed(&s.compare_tatoeba);
    s.compare_web = renamed(&s.compare_web);

    let out = dir.join("models");
    write_web_language_dir(&out, &outcome).unwrap();
    let card = &outcome.card;
    let entries = release::entries(&out, &FINEWEB_MODELS).unwrap();
    let urls = [&card.file.name, SENTENCES_FILE, DOCUMENTS_FILE].map(|f| FINEWEB_MODELS.url(f));
    assert_eq!(entries, [card.artifact(&urls[0], &urls[1], &urls[2]).unwrap()]);
    let listed: Vec<(&str, &str)> = entries[0].files.iter().map(|f| (f.path.as_str(), f.url.as_str())).collect();
    assert_eq!(
        listed,
        [
            (card.file.name.as_str(), urls[0].as_str()),
            (SENTENCES_FILE, urls[1].as_str()),
            (DOCUMENTS_FILE, urls[2].as_str())
        ]
    );
    assert!(urls[2].starts_with(
        "https://github.com/ai-pipestream/turboembed/releases/download/models-tatoeba-fineweb-2026-10-10-1/"
    ));

    // A record that is not the card's makes no entry: the web gate's with
    // another outcome, and a comparison of another model or with other
    // scores.
    let model_dir = out.join(MODEL_ID);
    let refused = |field: &str| match release::entries(&out, &FINEWEB_MODELS) {
        Err(Error::InvalidCard { field: f, message }) => {
            assert_eq!(f, field, "{message}");
            message
        }
        other => panic!("{other:?}"),
    };
    // A record is written once, so the one replaced is removed first.
    let replace = |file: &str, units: Vec<PairedUnit>, of: &ModelGateRecord, candidate: ModelId| {
        let path = model_dir.join(file);
        fs::remove_file(&path).unwrap();
        let record =
            ModelGateRecord::new(Task::LanguageDetector, of.set.clone(), candidate, of.reference.clone(), units);
        record.unwrap().write(&path).unwrap();
    };
    let kept = |file: &str| fs::read(model_dir.join(file)).unwrap();
    let s = &outcome.scored;
    let wrong = PairedUnit { candidate: Counts::accuracy(0, 1), reference: Counts::accuracy(1, 1) };

    let web_gate = kept(WEB_GATE_FILE);
    replace(WEB_GATE_FILE, vec![wrong; s.web_gate.units.len()], &s.web_gate, candidate.clone());
    assert_eq!(refused("gate"), format!("{WEB_GATE_FILE} of {MODEL_ID} has another outcome than its card"));
    fs::write(model_dir.join(WEB_GATE_FILE), &web_gate).unwrap();

    let compare_web = kept(COMPARE_WEB_FILE);
    let r = &s.compare_web;
    let other = ModelId { id: "other".to_owned(), ..candidate.clone() };
    replace(COMPARE_WEB_FILE, r.units.clone(), r, other);
    assert!(refused("compared").starts_with(&format!("{COMPARE_WEB_FILE} scores other (SHA-256 ")));
    let mut units = r.units.clone();
    units[0] = if units[0].candidate.true_positives == 1 { wrong } else { right };
    replace(COMPARE_WEB_FILE, units, r, candidate.clone());
    assert_eq!(refused("compared"), format!("{COMPARE_WEB_FILE} of {MODEL_ID} has other scores than its card"));
    fs::write(model_dir.join(COMPARE_WEB_FILE), &compare_web).unwrap();
    assert!(release::entries(&out, &FINEWEB_MODELS).is_ok());

    // A changed or missing documents file makes no entry.
    let documents = model_dir.join(DOCUMENTS_FILE);
    let mut changed = outcome.trained.documents_file.clone();
    changed[0] ^= 1;
    fs::write(&documents, &changed).unwrap();
    match release::entries(&out, &FINEWEB_MODELS) {
        Err(Error::Zoo(turbo_zoo::ZooError::Integrity { id, .. })) => assert_eq!(id.as_str(), MODEL_ID),
        other => panic!("{other:?}"),
    }
    fs::remove_file(&documents).unwrap();
    assert!(matches!(release::entries(&out, &FINEWEB_MODELS), Err(Error::Io { .. })));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn windows_from_tatoeba_or_holding_a_sentence_of_the_other_side_are_passed_over() {
    let dir = scratch("fineweb-overlap");
    let split = WebRecipe { split: WebSplit { window: 40, ..SMALL }, ..WEB };
    let afr = 0;
    let deu = WEB.languages.iter().position(|l| l.tatoeba == "deu").unwrap();
    let (files, bytes) = served_with(&split, |tatoeba, side| {
        (0..30)
            .map(|i| {
                let text = match (tatoeba, side, i) {
                    ("afr", Side::Training, 0) => "A page of Tatoeba's own, in Afrikaans.".to_owned(),
                    ("afr", Side::Training, 1) => "Ek het die boek gelees, en meer.".to_owned(),
                    ("afr", Side::Training, 2) => "Sy het gesê: Ich habe das Buch gelesen.".to_owned(),
                    ("afr", Side::HeldOut, 0) => "Toe: Hy loop elke dag skool toe.".to_owned(),
                    ("afr", Side::HeldOut, 1) => "Kort sin. En nog iets daarby.".to_owned(),
                    _ => format!("{i:03} {tatoeba} {side:?} document"),
                };
                let host = if (tatoeba, side, i) == ("afr", Side::Training, 0) {
                    "en.tatoeba.org".to_owned()
                } else {
                    format!("h{i}.example")
                };
                [text, format!("<urn:{i}>"), format!("https://{host}/{i}")]
            })
            .collect()
    });
    fetch_all(&dir, &files, &split, &bytes);
    let ranges = record(&files, &split, &Store::new(&dir, None)).unwrap();
    let n = WEB.languages.len();
    // A held-out sentence of Afrikaans and one of German, a training
    // sentence of Afrikaans, and one too short to be looked for.
    let avoid = Avoid {
        training: Overlap::new(n, 16, [(afr, "Ek het die boek gelees"), (deu, "Ich habe das Buch gelesen.")]),
        held_out: Overlap::new(n, 16, [(afr, "Hy loop elke dag skool toe."), (afr, "Kort sin.")]),
    };
    let plan = plan_web(&files, &split, &Store::new(&dir, Some(&ranges)), &avoid).unwrap();
    let t = choice(&plan.files, "afr", Side::Training)[0];
    assert_eq!((t.excluded_host, t.tatoeba_overlap, t.chosen), (1, 1, 6));
    let rows: Vec<u64> = plan.training.iter().filter(|s| s.language == afr).map(|s| s.row).collect();
    // Another language's held-out sentence is kept in an Afrikaans window.
    assert_eq!(rows[0], 2);
    let h = choice(&plan.files, "afr", Side::HeldOut)[0];
    assert_eq!((h.tatoeba_overlap, h.chosen), (1, 3));
    let held: Vec<u64> = plan.held_out.iter().filter(|s| s.language == afr).map(|s| s.row).collect();
    assert_eq!(held[0], 1);
    // Every other file is untouched.
    let others = plan.files.iter().filter(|c| c.tatoeba != "afr");
    assert!(others.clone().all(|c| c.excluded_host == 0 && c.tatoeba_overlap == 0));
    assert!(plan.files.iter().all(|c| c.chosen == c.wanted));
}

#[test]
fn the_binary_says_what_to_fetch_and_needs_its_inputs() {
    let exe = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let paths = Command::new(exe).args(["fineweb-ranges", "--paths"]).output().unwrap();
    assert!(paths.status.success(), "{paths:?}");
    let text = String::from_utf8(paths.stdout).unwrap();
    assert_eq!(text.lines().count(), 204);
    let first =
        format!("{}\t{}\tdata/afr_Latn/train/000_00000.parquet", fineweb::FINEWEB_2.repo, fineweb::FINEWEB_2.revision);
    assert_eq!(text.lines().next(), Some(first.as_str()));
    assert!(text.contains(&format!(
        "{}\t{}\tsample/10BT/001_00000.parquet\n",
        fineweb::FINEWEB.repo,
        fineweb::FINEWEB.revision
    )));

    // With nothing fetched, each file's last 8 bytes are needed first.
    let dir = scratch("fineweb-binary");
    let (files, _) = served(&WEB);
    let list = dir.join("files.txt");
    fs::write(&list, fineweb::write_files(&files)).unwrap();
    let web = dir.join("web");
    let need = Command::new(exe)
        .args(["fineweb-ranges", "--need", "--files"])
        .arg(&list)
        .arg("--web")
        .arg(&web)
        .output()
        .unwrap();
    assert!(need.status.success(), "{need:?}");
    let need = String::from_utf8(need.stdout).unwrap();
    assert_eq!(need.lines().count(), files.len());
    let f = &files[0];
    let line = format!(
        "{}\t{}\t{}\t{}",
        fineweb::FINEWEB_2.url(&f.path),
        Store::new(&web, None).file(f, Span { start: f.size - 8, end: f.size }).display(),
        f.size - 8,
        f.size
    );
    assert_eq!(need.lines().next(), Some(line.as_str()));
    let record = Command::new(exe)
        .args(["fineweb-ranges", "--record", "--files"])
        .arg(&list)
        .arg("--web")
        .arg(&web)
        .output()
        .unwrap();
    assert_eq!(record.status.code(), Some(2), "{record:?}");
    assert!(String::from_utf8_lossy(&record.stderr).contains("not fetched"), "{record:?}");

    // A flag a command does not read is refused.
    let paths = Command::new(exe).args(["fineweb-ranges", "--paths", "--web", "x"]).output().unwrap();
    assert_eq!(paths.status.code(), Some(2), "{paths:?}");
    assert!(String::from_utf8_lossy(&paths.stderr).starts_with("fineweb-ranges --paths does not read --web"));

    // The recipe needs both models, the web directory, --files and
    // --ranges together and, while the committed lists are empty,
    // --unrecorded-web; every problem is named before anything is read.
    let references = dir.join("references");
    fs::create_dir(&references).unwrap();
    let run = Command::new(exe)
        .arg("tatoeba-fineweb")
        .arg("--export")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tatoeba"))
        .arg("--web")
        .arg(&web)
        .arg("--references")
        .arg(&references)
        .arg("--out")
        .arg(dir.join("models"))
        .arg("--files")
        .arg(&list)
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(2), "{run:?}");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("langdetect-183.bin"), "{stderr}");
    assert!(stderr.contains("turbo-mul-tatoeba-langdetect-1.bin"), "{stderr}");
    assert!(stderr.contains("--files and --ranges are given together"), "{stderr}");
    assert!(stderr.contains(&format!("{}: no such directory", web.display())), "{stderr}");
    assert!(stderr.contains("--unrecorded-web"), "{stderr}");
    assert!(!dir.join("models").exists());
}
