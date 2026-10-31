//! The web data of the language detector recipe on Parquet files written
//! for the test, one per file the recipe reads: the ranges fetched round
//! by round as `fineweb-ranges` fetches them, recorded, checked, and the
//! windows chosen from them; and the web language detector recipe end to
//! end on those files and the Tatoeba export of `tests/data/tatoeba`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_train_recipes::fineweb::{
    self, Avoid, FileChoice, Overlap, Side, Span, Store, WEB, WebError, WebFile, WebRecipe, WebSplit, check_ranges,
    needed, plan_web, record,
};
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

#[path = "fineweb/cli.rs"]
mod cli;
