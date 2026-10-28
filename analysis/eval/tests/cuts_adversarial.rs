use std::{
    fs,
    path::{Path, PathBuf},
};
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences};
use turbo_eval::cuts::{Dump, Header, Writer, compare_dirs, compare_dumps, cuts_path};

fn dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cuts-review-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn header() -> Header {
    Header {
        set: "s".into(),
        sha256: "ab".repeat(32),
        backend: "cpu".into(),
        device: "cpu".into(),
        precision: "exact".into(),
        max: 100,
        target: 50,
        size_weight: 1000,
        overhead: 0,
        chunker: vec![],
    }
}
fn doc(end: u32) -> Document {
    let mut d = Document::new("cat. dog.");
    add_layers(&mut d, ViewId::SOURCE, end);
    d
}
fn add_layers(d: &mut Document, view: ViewId, end: u32) {
    let mut sentences = Spans::new(view);
    sentences.push(Span::new(0, 4));
    sentences.push(Span::new(5, 9));
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentences }, Provenance::by("review:fixture")).unwrap();
    d.add(
        &Key::<Boundaries>::DEFAULT,
        Boundaries { cost: vec![1000, 0], reason: vec![Cut::Sentence, Cut::Sentence] },
        Provenance::by("review:fixture"),
    )
    .unwrap();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, end));
    d.add(
        &Key::<Chunks>::DEFAULT,
        Chunks {
            spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, if end == 4 { 1 } else { 2 })],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![2],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("review:fixture"),
    )
    .unwrap();
}
fn dump(path: &Path) -> PathBuf {
    let mut w = Writer::create(path, &header()).unwrap();
    w.document("one", &doc(9)).unwrap();
    w.document("two", &doc(9)).unwrap();
    w.finish().unwrap().0
}

#[test]
fn failed_second_writer_cannot_modify_completed_dump() {
    let root = dir("writers");
    let mut first = Writer::create(&root, &header()).unwrap();
    let mut second = Writer::create(&root, &header()).unwrap();
    first.document("first", &doc(9)).unwrap();
    second.document("other", &doc(9)).unwrap();
    let path = first.finish().unwrap().0;
    let published = fs::read(&path).unwrap();
    assert_eq!(Dump::read(&path).unwrap().docs[0].0, "first");
    assert!(second.finish().is_err());
    assert!(
        fs::read(&path).unwrap() == published,
        "failed writer changed completed dump; document is now {}",
        Dump::read(&path).unwrap().docs[0].0
    );
}

#[test]
fn dropping_second_writer_cannot_modify_completed_dump() {
    let root = dir("drop-writer");
    let mut first = Writer::create(&root, &header()).unwrap();
    let mut second = Writer::create(&root, &header()).unwrap();
    first.document("first", &doc(9)).unwrap();
    second.document("other", &doc(9)).unwrap();
    let path = first.finish().unwrap().0;
    let published = fs::read(&path).unwrap();
    drop(second);
    assert!(fs::read(&path).unwrap() == published, "dropping another writer changed completed file");
}

#[test]
fn record_boundary_truncation_must_be_detected() {
    let root = dir("truncated");
    let path = dump(&root);
    let bytes = fs::read(&path).unwrap();
    let magic_len = b"turbo-eval cuts 1\n".len();
    let n = u32::from_le_bytes(bytes[magic_len..magic_len + 4].try_into().unwrap()) as usize;
    let mut p = magic_len + 4 + n;
    for _ in 0..2 {
        let n = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4 + n;
    }
    fs::write(&path, &bytes[..p]).unwrap();
    let shortened = Dump::read(&path);
    assert!(
        shortened.is_err(),
        "truncated two-document dump was accepted with {} document(s)",
        shortened.unwrap().docs.len()
    );
}

#[test]
fn header_only_truncations_must_not_report_identical() {
    let a = dir("header-a");
    let b = dir("header-b");
    for root in [&a, &b] {
        let path = dump(root);
        let bytes = fs::read(&path).unwrap();
        let m = b"turbo-eval cuts 1\n".len();
        let n = u32::from_le_bytes(bytes[m..m + 4].try_into().unwrap()) as usize;
        fs::write(&path, &bytes[..m + 4 + n]).unwrap();
    }
    let compared = compare_dirs(&a, &b);
    let cli =
        std::process::Command::new(env!("CARGO_BIN_EXE_turbo-eval")).arg("cuts").arg(&a).arg(&b).output().unwrap();
    assert!(
        compared.is_err() && !cli.status.success(),
        "corrupt artifacts produced successful comparison: {:?}; CLI status={}, stdout={}",
        compared.unwrap(),
        cli.status,
        String::from_utf8_lossy(&cli.stdout)
    );
}

#[test]
fn child_view_identity_is_not_just_its_numeric_id() {
    let make = |replacement: &str| {
        let mut d = Document::new("cat. dog.");
        let source = d.get::<View>().unwrap().clone();
        let mut vb = ViewBuilder::new(source.text());
        vb.replace(3, replacement);
        let (text, align) = vb.finish();
        d.add(&Key::<View>::named("edited"), View::derived(&source, text, align), Provenance::by("review:view"))
            .unwrap();
        let id = d.get_key(&Key::<View>::named("edited")).unwrap().id();
        add_layers(&mut d, id, 9);
        Dump { header: header(), docs: vec![("one".into(), d)] }
    };
    let a = make("rat");
    let b = make("bat");
    assert!(compare_dumps(&a, &b).is_err(), "same source and numeric ViewId conceal different compared text");
}

#[test]
fn ordinary_roundtrip_preserves_unicode_and_rejects_midrecord_truncation() {
    let mut d = Document::new("😀cat dog");
    let mut ss = Spans::new(ViewId::SOURCE);
    ss.push(Span::new(0, 11));
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: ss }, Provenance::by("review")).unwrap();
    d.add(
        &Key::<Boundaries>::DEFAULT,
        Boundaries { cost: vec![0], reason: vec![Cut::Sentence] },
        Provenance::by("review"),
    )
    .unwrap();
    let mut cs = Spans::new(ViewId::SOURCE);
    cs.push(Span::new(0, 11));
    d.add(
        &Key::<Chunks>::DEFAULT,
        Chunks {
            spans: cs,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![3],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("review"),
    )
    .unwrap();
    let root = dir("unicode");
    let mut w = Writer::create(&root, &header()).unwrap();
    w.document("unicode", &d).unwrap();
    w.finish().unwrap();
    let path = cuts_path(&root, "s");
    let got = Dump::read(&path).unwrap();
    assert_eq!(got.docs[0].1.source(), d.source());
    assert_eq!(
        got.docs[0].1.view_span_to_utf16(ViewId::SOURCE, Span::new(0, 11)).unwrap(),
        d.view_span_to_utf16(ViewId::SOURCE, Span::new(0, 11)).unwrap()
    );
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(Dump::read(&path).is_err());
}
