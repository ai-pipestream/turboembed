//! Authored MASC XML, never upstream or model-generated gold.
use std::fs;
use std::path::Path;
use turbo_formats::masc::{MascError, read_document};
use turbo_trainer_recipes::masc_ne::{self, ReadError};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("masc-ne-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        for n in 0..31 {
            write(&dir, n);
        }
        Self(dir)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn write(root: &Path, n: usize) {
    let stem = format!("doc{n:02}");
    let text = format!("😀 Ada Lovelace Paris Acme May {stem}");
    let mut seg = "<graph>".to_owned();
    let mut penn = "<graph>".to_owned();
    let mut offset = 0;
    for (i, word) in text.split(' ').enumerate() {
        let end = offset + word.encode_utf16().count();
        seg.push_str(&format!(r#"<region xml:id="seg-r{i}" anchors="{offset} {end}"/>"#));
        penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
        offset = end + 1;
    }
    seg.push_str("</graph>");
    penn.push_str("</graph>");
    let mut ne = "<graph>".to_owned();
    for (id, kind, tokens) in
        [(7, "person", vec![1, 2]), (8, "location", vec![3]), (9, "org", vec![4]), (10, "date", vec![5])]
    {
        ne.push_str(&format!(r#"<a ref="ne-n{id}" label="{kind}"/>"#));
        for t in tokens {
            ne.push_str(&format!(r#"<edge from="ne-n{id}" to="penn-n{t}"/>"#));
        }
    }
    ne.push_str("</graph>");
    let sentences = format!(r#"<graph><region anchors="0 {}"/></graph>"#, offset - 1);
    let mut header = format!(r#"<documentHeader><primaryData f.id="f.text" loc="{stem}.txt"/><annotations>"#);
    for (id, suffix, data) in
        [("f.seg", "seg", seg), ("f.penn", "penn", penn), ("f.ne", "ne", ne), ("f.s", "s", sentences)]
    {
        header.push_str(&format!(r#"<annotation f.id="{id}" loc="{stem}-{suffix}.xml"/>"#));
        fs::write(root.join(format!("{stem}-{suffix}.xml")), data).unwrap();
    }
    header.push_str("</annotations>");
    if n == 30 {
        header.push_str(r#"<textClass catRef="WR TW"/>"#);
    }
    header.push_str("</documentHeader>");
    fs::write(root.join(format!("{stem}.hdr")), header).unwrap();
    fs::write(root.join(format!("{stem}.txt")), text).unwrap();
}

#[test]
fn document_splits_schema_and_typed_refusal() {
    let fixture = Fixture::new();
    assert!(masc_ne::read(&fixture.0, false).unwrap_err().to_string().contains("corpus digest"));
    let corpus = masc_ne::read(&fixture.0, true).unwrap();
    assert!(!corpus.record.recorded);
    assert_eq!(corpus.labels().len(), 9);
    let head = turbo_trainer_recipes::run::spec(
        &corpus,
        &turbo_trainer_recipes::run::HeadOptions { crf: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(head.decode, turbo::manifest::Decode::Crf);
    assert_eq!(head.license_class, Some(turbo::manifest::LicenseClass::Permissive));
    assert!(!head.labels.iter().any(|l| l.contains("corporation")));
    assert_eq!(corpus.record.spdx, "CC-BY-3.0-US");
    let ids = |samples: &[turbo_trainer_recipes::corpus::Sentence]| {
        samples.iter().map(|s| s.words.last().unwrap().clone()).collect::<Vec<_>>()
    };
    let train = ids(&corpus.splits.train);
    let dev = ids(&corpus.splits.dev);
    let test = ids(&corpus.splits.test);
    assert!(!train.is_empty() && !dev.is_empty() && !test.is_empty());
    assert!(train.iter().all(|id| !dev.contains(id) && !test.contains(id)));
    assert!(dev.iter().all(|id| !test.contains(id)));
    assert!(!train.iter().chain(&dev).chain(&test).any(|id| id == "doc30"));
    let split = turbo_train_recipes::masc::SPLIT;
    let expected: Vec<_> =
        (0..30).filter(|n| n % split.every() == split.held_out_at()).map(|n| format!("doc{n:02}")).collect();
    assert_eq!(test, expected);
    let expected_dev: Vec<_> = (0..30)
        .filter(|n| n % split.every() != split.held_out_at())
        .enumerate()
        .filter(|(i, _)| i % 10 == 9)
        .map(|(_, n)| format!("doc{n:02}"))
        .collect();
    assert_eq!(dev, expected_dev);
    assert_eq!(
        corpus.splits.train[0].labels,
        ["O", "B-person", "I-person", "B-location", "B-organization", "B-date", "O"]
    );

    let header = fixture.0.join("doc00.hdr");
    let text = fs::read_to_string(fixture.0.join("doc00.txt")).unwrap();
    fs::write(
        fixture.0.join("doc00-s.xml"),
        format!(r#"<graph><region anchors="0 6"/><region anchors="7 {}"/></graph>"#, text.encode_utf16().count()),
    )
    .unwrap();
    let doc = read_document(&header).unwrap();
    assert!(matches!(masc_ne::sentences(&doc), Err(ReadError::Annotation(MascError::CutEntity {
        path, entity: 7, start: 3, end: 6
    })) if path == header));
    let error = masc_ne::read(&fixture.0, true).unwrap_err().to_string();
    assert!(error.contains("doc00.hdr") && error.contains("ne-n7") && error.contains("3..6"));
}
