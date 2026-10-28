//! Reviewed temporal snapshots and span/value invariants.
//!
//! `tests/snapshots/temporal/<mode>/<id>.{temporal,document-date}.tsv`.
//! Regenerate with `TURBO_BLESS=1 cargo test -p turbo-numeric --test temporal_goldens`.
//! Fixed mode reference date: 2026-03-15.
//!
//! Snapshots freeze the default (non-OpenNLP-parity) configuration. Relative to
//! the retired JVM/parity goldens that is a deliberate change: default mode
//! keeps year-range edges such as July 0999 and July 3000 in
//! `fixed/years-range`, and the 100-day relative expressions in
//! `fixed/relative-count-digits`, which parity mode omitted. See
//! `temporal_fixes.rs` for the targeted assertions.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Layer};
use turbo_numeric::{Date, DocumentDate, DocumentDateSelector, Origin, TemporalExtractor, Temporals};

const REFERENCE: &str = "2026-03-15";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/temporal"))
        .expect("tests/inputs/temporal")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs/temporal").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(mode: &str, id: &str, suffix: &str) -> PathBuf {
    root().join("snapshots/temporal").join(mode).join(format!("{id}.{suffix}.tsv"))
}

fn render_temporals(t: &Temporals) -> String {
    let mut out = String::new();
    for (i, span) in t.spans.iter() {
        let m = t.get(i);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            span.start,
            span.end,
            m.value,
            m.value.granularity().name(),
            m.origin.name(),
        ));
    }
    out
}

fn render_document_date(d: &DocumentDate) -> String {
    match (d.date, d.mention) {
        (None, None) => String::new(),
        (Some(date), None) => format!("{date}\n"),
        (None, Some(ix)) => format!("\t{}\n", ix.get()),
        (Some(date), Some(ix)) => format!("{}\t{}\n", date, ix.get()),
    }
}

fn check_invariants(text: &str, t: &Temporals) {
    t.check().unwrap_or_else(|e| panic!("Layer::check: {e}"));
    let len = text.len() as u32;
    let mut prev = 0u32;
    for (_, span) in t.spans.iter() {
        assert!(span.start <= span.end && span.end <= len);
        assert!(span.start >= prev);
        prev = span.end;
    }
}

struct Compared {
    mentions: usize,
    relative: usize,
    dated: usize,
}

fn check(mode: &str, extractor: TemporalExtractor) -> Compared {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    assert!(!ids.is_empty());
    let analyzer = Analyzer::builder().add(extractor).add(DocumentDateSelector).build().expect("analyzer");
    let mut compared = Compared { mentions: 0, relative: 0, dated: 0 };
    for id in &ids {
        let text = text_of(id);
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let empty_t = Temporals::new(turbo_analysis::ViewId::SOURCE);
        let empty_d = DocumentDate::default();
        let t = docs[0].get::<Temporals>().unwrap_or(&empty_t);
        let d = docs[0].get::<DocumentDate>().unwrap_or(&empty_d);
        compared.mentions += t.spans.len();
        compared.relative += t.origin.iter().filter(|&&o| o == Origin::Relative).count();
        compared.dated += usize::from(d.date.is_some());
        check_invariants(&text, t);
        d.check().unwrap_or_else(|e| panic!("document-date check: {e}"));
        let t_body = render_temporals(t);
        let d_body = render_document_date(d);
        let t_path = snap_path(mode, id, "temporal");
        let d_path = snap_path(mode, id, "document-date");
        if bless {
            std::fs::create_dir_all(t_path.parent().unwrap()).unwrap();
            std::fs::write(&t_path, &t_body).unwrap();
            std::fs::write(&d_path, &d_body).unwrap();
            continue;
        }
        let tw = std::fs::read_to_string(&t_path)
            .unwrap_or_else(|e| panic!("missing {} (TURBO_BLESS=1): {e}", t_path.display()));
        let dw = std::fs::read_to_string(&d_path)
            .unwrap_or_else(|e| panic!("missing {} (TURBO_BLESS=1): {e}", d_path.display()));
        assert_eq!(t_body, tw, "{mode}/{id} temporal");
        assert_eq!(d_body, dw, "{mode}/{id} document-date");
    }
    compared
}

#[test]
fn dateline_snapshots() {
    let c = check("dateline", TemporalExtractor::new());
    assert!(c.mentions > 75, "only {} mentions", c.mentions);
    assert!(c.relative >= 5, "only {} relative", c.relative);
    assert!(c.dated > 25, "only {} dated", c.dated);
}

#[test]
fn fixed_snapshots() {
    let reference = Date::parse(REFERENCE).expect("reference");
    let c = check("fixed", TemporalExtractor::with_reference(reference));
    assert!(c.mentions > 110, "only {} mentions", c.mentions);
    assert!(c.relative > 40, "only {} relative", c.relative);
    assert!(c.dated > 25, "only {} dated", c.dated);
}
