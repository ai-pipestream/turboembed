//! Reviewed number-scan snapshots and span/value invariants.
//!
//! `tests/snapshots/numbers/<notation>/<id>.tsv` holds the `turbo:numbers`
//! rows for each input in `tests/inputs/num/`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-numeric --test goldens`.
//! Snapshots freeze the default (non-OpenNLP-parity) configuration the pipeline
//! runs. Each case checks Layer::check, spans in bounds, and that `plain` is
//! digits/sign/point only and matches `DecimalRef::to_plain_string`.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Layer, Provenance, ViewId};
use turbo_analysis_layers::{NumberNotation, Numbers};
use turbo_numeric::NumberScanner;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/num"))
        .expect("tests/inputs/num")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs/num").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(notation: &str, id: &str) -> PathBuf {
    root().join("snapshots/numbers").join(notation).join(format!("{id}.tsv"))
}

fn render(_text: &str, layer: &Numbers) -> String {
    let mut out = String::new();
    for (i, span) in layer.spans.iter() {
        let k = i.get();
        let plain = layer.value(i).to_plain_string();
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            span.start,
            span.end,
            layer.numeral_end[k],
            u32::from(layer.negative[k]),
            layer.coefficient.get(k),
            layer.scale[k],
            plain,
        ));
    }
    out
}

fn check_invariants(text: &str, layer: &Numbers) {
    layer.check().unwrap_or_else(|e| panic!("Layer::check: {e}"));
    let len = text.len() as u32;
    let mut prev_end = 0u32;
    for (i, span) in layer.spans.iter() {
        assert!(span.start <= span.end && span.end <= len, "span out of bounds: {span:?}");
        assert!(span.start >= prev_end, "overlapping/out-of-order span {span:?}");
        prev_end = span.end;
        let k = i.get();
        assert!(layer.numeral_end[k] >= span.start && layer.numeral_end[k] <= span.end);
        let plain = layer.value(i).to_plain_string();
        assert!(!plain.is_empty(), "empty plain for {span:?}");
        assert!(
            plain.chars().all(|c| c.is_ascii_digit() || c == '-' || c == '.'),
            "plain {plain:?} is not digits/sign/point only"
        );
    }
}

fn check(dir: &str, notation: NumberNotation) -> usize {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let scanner = NumberScanner::new(notation);
    let ids = input_ids();
    assert!(!ids.is_empty());
    let mut numbers = 0;
    for id in &ids {
        let text = text_of(id);
        let mut layer = Numbers::new(ViewId::SOURCE);
        scanner.scan(&text, &mut layer);
        numbers += layer.spans.len();
        check_invariants(&text, &layer);
        let body = render(&text, &layer);
        let path = snap_path(dir, id);
        if bless {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &body).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
        assert_eq!(body, want, "{dir}/{id}");
        let mut doc = Document::new(text.as_str());
        doc.add(&Key::<Numbers>::DEFAULT, layer, Provenance::by("snapshot")).unwrap();
        let got = doc.get_key(&Key::<Numbers>::DEFAULT).expect("Numbers on document");
        assert_eq!(got.spans.len(), body.lines().filter(|l| !l.is_empty()).count());
    }
    numbers
}

#[test]
fn latin_us_snapshots() {
    let n = check("latin-us", NumberNotation::LatinUs);
    assert!(n > 100, "only {n} numbers compared");
}

#[test]
fn latin_eu_snapshots() {
    let n = check("latin-eu", NumberNotation::LatinEu);
    assert!(n > 100, "only {n} numbers compared");
}
