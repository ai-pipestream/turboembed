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
    let recovered = masc_ne::read_with_documents(&fixture.0, true).unwrap();
    assert_eq!(recovered.corpus.splits.train[0].labels, corpus.splits.train[0].labels);
    for (docs, samples) in [
        (&recovered.train_documents, &recovered.corpus.splits.train),
        (&recovered.dev_documents, &recovered.corpus.splits.dev),
        (&recovered.test_documents, &recovered.corpus.splits.test),
    ] {
        assert_eq!(docs.len(), samples.len());
        for (i, d) in docs.iter().enumerate() {
            assert_eq!((d.first, d.end), (i, i + 1));
            assert_eq!(d.token_spans[0], [0, 4]);
            assert_eq!(d.source_sha256.len(), 64);
            assert!(!std::path::Path::new(&d.path).is_absolute());
        }
    }
    let gold = &recovered.corpus.splits.test;
    let exact: Vec<_> = gold.iter().map(|s| s.labels.clone()).collect();
    let outside: Vec<_> = gold.iter().map(|s| vec!["O".to_owned(); s.words.len()]).collect();
    let groups: Vec<_> = recovered.test_documents.iter().map(|d| d.first..d.end).collect();
    let types: Vec<_> = masc_ne::TYPES.iter().map(|s| (*s).to_owned()).collect();
    let comparison = turbo_trainer_recipes::ner_documents::compare(turbo_trainer_recipes::ner_documents::Request {
        gold,
        candidate: &exact,
        peer: &outside,
        documents: &groups,
        types: &types,
        resamples: 100,
        seed: 1,
    })
    .unwrap();
    assert_eq!(comparison.statistical_unit, "document");
    assert_eq!(comparison.candidate.all.f1, 1.0);
    assert_eq!(comparison.peer.all.f1, 0.0);
    train_authored_documents(&fixture.0, &recovered.corpus);
    fs::write(
        fixture.0.join("doc00-s.xml"),
        format!(r#"<graph><region anchors="16 {}"/></graph>"#, text.encode_utf16().count()),
    )
    .unwrap();
    let outside = read_document(&header).unwrap();
    assert!(!outside.unplaced_entities().is_empty());
    assert_eq!(masc_ne::document(&outside).unwrap().unwrap().labels, corpus.splits.train[0].labels);
    let ne_path = fixture.0.join("doc00-ne.xml");
    let ne = fs::read_to_string(&ne_path).unwrap();
    fs::write(
        &ne_path,
        ne.replace("</graph>", r#"<a ref="ne-n99" label="org"/><edge from="ne-n99" to="penn-n2"/></graph>"#),
    )
    .unwrap();
    assert!(matches!(masc_ne::read(&fixture.0,true),Err(ReadError::Unsupported(e))
        if e.entity == 99 && e.start == 7 && e.end == 15 && e.reason == "overlapping original annotations"));
    fs::write(&ne_path, &ne).unwrap();
    let seg_path = fixture.0.join("doc00-seg.xml");
    let seg = fs::read_to_string(&seg_path).unwrap();
    fs::write(&seg_path, seg.replace("anchors=\"7 15\"", "anchors=\"7 7\"")).unwrap();
    let zero_doc = read_document(&header).unwrap();
    assert_eq!(zero_doc.document_tokens().unwrap().tokens().unwrap().len(), 7);
    assert!(matches!(masc_ne::document(&zero_doc), Err(ReadError::Unsupported(e))
        if e.entity == 7 && e.start == 3 && e.end == 7 && e.reason == "entity references a zero-width source token"));
    fs::write(&seg_path, seg).unwrap();
    fs::write(&ne_path, ne.replace("from=\"ne-n7\" to=\"penn-n2\"", "from=\"ne-n7\" to=\"penn-n3\"")).unwrap();
    let discontinuous = read_document(&header).unwrap();
    assert!(matches!(masc_ne::document(&discontinuous), Err(ReadError::Unsupported(e))
        if e.entity == 7 && e.reason == "discontinuous original annotation"));
    fs::write(&ne_path, &ne).unwrap();
    let header_text = fs::read_to_string(&header).unwrap();
    fs::write(&header, header_text.replace(r#"<annotation f.id="f.ne" loc="doc00-ne.xml"/>"#, "")).unwrap();
    let unannotated = masc_ne::read_with_documents(&fixture.0, true).unwrap();
    assert!(!unannotated.train_documents.iter().any(|d| d.path == "doc00.hdr"));
    assert_eq!(ids(&unannotated.corpus.splits.dev), dev);
    assert_eq!(ids(&unannotated.corpus.splits.test), test);
    fs::write(&header, header_text.replace(r#"<annotation f.id="f.penn" loc="doc00-penn.xml"/>"#, "")).unwrap();
    let no_penn = read_document(&header).unwrap();
    assert!(masc_ne::document(&no_penn).unwrap_err().to_string().contains("NE document has no Penn tokens"));
}

fn train_authored_documents(data: &Path, corpus: &turbo_trainer_recipes::corpus::Corpus) {
    let out = data.with_extension("trained");
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["train", "--corpus", "masc-ne", "--dir"])
        .arg(data)
        .arg("--bundle")
        .arg(bundle)
        .arg("--out")
        .arg(&out)
        .args([
            "--unrecorded-corpus",
            "--crf",
            "--threads",
            "2",
            "--epochs",
            "2",
            "--batch",
            "2",
            "--commit",
            "fixture",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let receipt: turbo_trainer_recipes::receipt::Receipt =
        serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert!(!receipt.corpus.recorded);
    assert_eq!(receipt.config.as_ref().unwrap().dev_left_out, 0);
    assert_eq!(receipt.timing_s.feature_words, (corpus.record.train.words + corpus.record.dev.words) as u64);
    assert_eq!(receipt.head.as_ref().unwrap().labels.len(), 9);
    assert!(receipt.eval.model.is_some());
    assert!(fs::metadata(out.join("head.safetensors")).unwrap().len() > 0);
    fs::remove_dir_all(out).unwrap();
}
