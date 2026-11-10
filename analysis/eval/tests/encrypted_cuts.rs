#[path = "encrypted/common.rs"]
mod common;
use common::*;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_embed::{Device, Precision};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences};
use turbo_chunker::{Budget, Costs};
use turbo_eval::analysis::{Analysis, Sentencer};
use turbo_eval::artifact::EvaluationArtifactStore;
use turbo_eval::artifact::encrypted::{KeyFile, PrincipalId, new_artifact_key};
use turbo_eval::chunkers::{Signals, TurboChunker};
use turbo_eval::cuts::{
    CUTS_CLI_PRINCIPAL, Dump, DumpConfig, Header, Writer, compare_dirs, compare_dumps, compare_encrypted_dirs,
    cuts_path, encrypted_store,
};
use turbo_eval::embedder::TurboEmbedder;
use turbo_eval::metrics::{Qrel, Qrels};
use turbo_eval::pipeline::{Pipeline, Source, Titles};
use turbo_eval::run::{record_path, run_sets};
use turbo_eval::set::{Doc, LoadedSet, Query};
use turbo_structure::Syntax;

fn header() -> Header {
    Header {
        set: "private-set-name".into(),
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
fn document() -> Document {
    let mut doc = Document::new("秘密 café 🦀.");
    let end = doc.source().len() as u32;
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, end));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: spans.clone() }, Provenance::by("fixture")).unwrap();
    doc.add(
        &Key::<Boundaries>::DEFAULT,
        Boundaries { cost: vec![0], reason: vec![Cut::Sentence] },
        Provenance::by("fixture"),
    )
    .unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, end));
    doc.add(
        &Key::<Chunks>::DEFAULT,
        Chunks {
            spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![2],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("fixture"),
    )
    .unwrap();
    doc
}

#[test]
fn explicitly_keyed_cut_exports_encrypt_metadata_and_preserve_comparison() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let a = new_artifact_key().unwrap();
    let b = new_artifact_key().unwrap();
    let header = header();
    let document = document();
    // The legacy entry point still requires set-derived filenames. It does
    // not silently change names, store plaintext, or bypass adapter policy.
    assert!(Writer::create_in(&store, &header).is_err());
    for key in [&a, &b] {
        let mut writer = Writer::create_at(&store, key, &header).unwrap();
        writer.document("private-document-id", &document).unwrap();
        let (publication, count) = writer.publish().unwrap();
        assert_eq!(publication.key, *key);
        assert_eq!(count, 1);
        let bytes = fs::read(fixture.physical(1, key)).unwrap();
        for private in [document.source().as_bytes(), header.set.as_bytes(), b"private-document-id"] {
            assert!(!bytes.windows(private.len()).any(|w| w == private));
        }
    }
    let a_dump = Dump::read_at(&store, &a, &header.set).unwrap();
    let b_dump = Dump::read_at(&store, &b, &header.set).unwrap();
    assert_eq!(a_dump.docs[0].1.source(), document.source());
    assert_eq!(a_dump.docs[0].0, "private-document-id");
    assert_eq!(a_dump.header, header);
    assert!(compare_dumps(&a_dump, &b_dump).unwrap().is_identical());
    assert!(Dump::read_at(&store, &a, "wrong expected set").is_err());
    assert!(Dump::read_from(&store, &a).is_err());
    assert_eq!(store.list().unwrap().len(), 2);
}

#[test]
fn keyed_cut_decoding_errors_never_reveal_private_identifiers() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let key = key(1);
    let header = header();
    let mut writer = Writer::create_at(&store, &key, &header).unwrap();
    // This graph lacks required sentence/chunk layers: the storage framing
    // is authentic, but the consumer must refuse it without leaking the ID.
    writer.document("private-document-id", &Document::new("sensitive source")).unwrap();
    writer.publish().unwrap();
    let error = Dump::read_at(&store, &key, &header.set).err().unwrap();
    assert_eq!(error, "cut artifact decoding failed");
    assert!(!error.contains("private-document-id"));
    assert!(!error.contains("sensitive source"));
}

fn toy_set() -> LoadedSet {
    LoadedSet {
        name: "secret-eval-set".into(),
        sha256: "cd".repeat(32),
        docs: vec![Doc {
            id: "secret-doc-alpha".into(),
            title: None,
            text: "cat dog fox\ncat dog hen\n\nred sky sun\nred sky sea\n".into(),
        }],
        queries: vec![Query { id: "q1".into(), text: "cat".into() }],
        qrels: Qrels::new([Qrel { query: "q1".into(), doc: "secret-doc-alpha".into(), grade: 1 }]).unwrap(),
        spans: None,
        unplaced: Vec::new(),
        absent: Vec::new(),
    }
}

fn make_pipeline(costs: Costs, dump_cuts: DumpConfig) -> Pipeline {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let embedder = TurboEmbedder::open(&bundle, Device::Cpu, Precision::Exact).unwrap();
    let chunker = TurboChunker {
        budget: Budget { max: 9, target: 8, size_weight: 1000, overhead: 2 },
        costs,
        signals: Signals::Structure,
        cohesion: None,
        clauses: false,
    };
    Pipeline {
        analysis: Analysis { sentences: Sentencer::Newline, structure: Some(Syntax::Plain) },
        chunker: Box::new(chunker),
        embedder: Box::new(embedder),
        titles: Titles::Ignore,
        lexical: None,
        hybrid: None,
        cascade: None,
        static_embedder: None,
        dump_cuts: Some(dump_cuts),
    }
}

fn run_pipeline_with_cuts(set: &LoadedSet, costs: Costs, dump_cuts: DumpConfig) {
    let mut pipeline = make_pipeline(costs, dump_cuts);
    pipeline.evaluate(set).unwrap();
}

#[test]
fn pipeline_and_cli_export_and_compare_encrypted_cuts_with_caller_keys() {
    let fixture = Fixture::new();
    let set = toy_set();
    let plain_dir = fixture.root.join("plain");
    run_pipeline_with_cuts(&set, Costs::default(), DumpConfig::plain(&plain_dir, Precision::Exact));
    assert!(cuts_path(&plain_dir, &set.name).is_file());
    assert!(compare_dirs(&plain_dir, &plain_dir).unwrap()[0].is_identical());

    let key_file = fixture.root.join("keys.txt");
    fs::write(
        &key_file,
        "01010101010101010101010101010101:4747474747474747474747474747474747474747474747474747474747474747\n",
    )
    .unwrap();
    let keys = Arc::new(KeyFile::load(&key_file).unwrap());
    let enc_a = fixture.root.join("enc-a");
    let enc_b = fixture.root.join("enc-b");
    let enc_diff = fixture.root.join("enc-diff");
    run_pipeline_with_cuts(
        &set,
        Costs::default(),
        DumpConfig::encrypted(&enc_a, keys.clone(), Precision::Exact).unwrap(),
    );
    run_pipeline_with_cuts(
        &set,
        Costs::default(),
        DumpConfig::encrypted(&enc_b, keys.clone(), Precision::Exact).unwrap(),
    );
    let dear = Costs { paragraph: 50_000, sentence: 0, ..Costs::default() };
    run_pipeline_with_cuts(&set, dear, DumpConfig::encrypted(&enc_diff, keys.clone(), Precision::Exact).unwrap());

    for dir in [&enc_a, &enc_b] {
        assert!(!cuts_path(dir, &set.name).exists());
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(entries.len(), 1);
        let filename = entries[0].file_name().unwrap().to_str().unwrap();
        let expected_prefix =
            format!("{}.", CUTS_CLI_PRINCIPAL.0.iter().map(|b| format!("{b:02x}")).collect::<String>());
        assert!(filename.starts_with(&expected_prefix), "{filename}");
        assert!(!filename.contains(&set.name));
        let ciphertext = fs::read(&entries[0]).unwrap();
        assert!(ciphertext.starts_with(b"TEAEAD01"));
        for secret in [set.name.as_bytes(), b"secret-doc-alpha", b"cat dog fox"] {
            assert!(!ciphertext.windows(secret.len()).any(|w| w == secret));
        }
        let store = encrypted_store(dir, keys.clone()).unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        let decoded = Dump::read_at(&store, &listed[0], &set.name).unwrap();
        assert_eq!(decoded.docs[0].0, "secret-doc-alpha");
        assert_eq!(decoded.docs[0].1.source(), set.docs[0].text);
    }

    assert!(compare_dirs(&enc_a, &enc_b).is_err());
    let compared = compare_encrypted_dirs(&enc_a, &enc_b, keys).unwrap();
    assert_eq!(compared.len(), 1);
    assert!(compared[0].is_identical());

    let bin = env!("CARGO_BIN_EXE_turbo-eval");
    let ok =
        Command::new(bin).arg("cuts").arg(&enc_a).arg(&enc_b).arg("--cuts-key-file").arg(&key_file).output().unwrap();
    assert_eq!(ok.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&ok.stderr));
    assert!(String::from_utf8_lossy(&ok.stdout).contains(&set.name));

    let diff = Command::new(bin)
        .arg("cuts")
        .arg(&enc_a)
        .arg(&enc_diff)
        .arg("--cuts-key-file")
        .arg(&key_file)
        .output()
        .unwrap();
    assert_eq!(diff.status.code(), Some(1));

    let no_flag = Command::new(bin).arg("cuts").arg(&enc_a).arg(&enc_b).output().unwrap();
    assert_eq!(no_flag.status.code(), Some(2));

    let wrong_key = fixture.root.join("wrong-key.txt");
    fs::write(
        &wrong_key,
        "01010101010101010101010101010101:9999999999999999999999999999999999999999999999999999999999999999\n",
    )
    .unwrap();
    let bad_auth =
        Command::new(bin).arg("cuts").arg(&enc_a).arg(&enc_b).arg("--cuts-key-file").arg(&wrong_key).output().unwrap();
    assert_eq!(bad_auth.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad_auth.stderr).contains("artifact authentication or completeness failed"));

    let missing_id = fixture.root.join("missing-id.txt");
    fs::write(
        &missing_id,
        "02020202020202020202020202020202:4747474747474747474747474747474747474747474747474747474747474747\n",
    )
    .unwrap();
    let unavail =
        Command::new(bin).arg("cuts").arg(&enc_a).arg(&enc_b).arg("--cuts-key-file").arg(&missing_id).output().unwrap();
    assert_eq!(unavail.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unavail.stderr).contains("artifact key unavailable"));

    let revoked = fixture.root.join("revoked.txt");
    fs::write(&revoked, "revoked:01010101010101010101010101010101\n").unwrap();
    let rev =
        Command::new(bin).arg("cuts").arg(&enc_a).arg(&enc_b).arg("--cuts-key-file").arg(&revoked).output().unwrap();
    assert_eq!(rev.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&rev.stderr).contains("artifact key revoked"));

    let wrong_principal = Command::new(bin)
        .arg("cuts")
        .arg(&enc_a)
        .arg(&enc_b)
        .arg("--cuts-key-file")
        .arg(&key_file)
        .arg("--cuts-principal")
        .arg(PrincipalId([9; 16]).0.iter().map(|b| format!("{b:02x}")).collect::<String>())
        .output()
        .unwrap();
    assert_eq!(wrong_principal.status.code(), Some(2));
}

#[test]
fn run_sets_and_cli_validate_and_compare_encrypted_cuts_end_to_end() {
    let fixture = Fixture::new();
    let key_file = fixture.root.join("cli-keys.txt");
    fs::write(
        &key_file,
        "01010101010101010101010101010101:4747474747474747474747474747474747474747474747474747474747474747\n",
    )
    .unwrap();
    let keys = Arc::new(KeyFile::load(&key_file).unwrap());
    let out_dir = fixture.root.join("records");
    let cuts_dir = fixture.root.join("cli-cuts");
    let mut pipeline =
        make_pipeline(Costs::default(), DumpConfig::encrypted(&cuts_dir, keys, Precision::Exact).unwrap());
    let source =
        Source { commit: "0123456789abcdef0123456789abcdef01234567".into(), pushed_to: vec!["origin/main".into()] };
    fn loader(_: &str, _: &Path) -> turbo_eval::Result<LoadedSet> {
        Ok(toy_set())
    }
    let written =
        run_sets(&mut pipeline, &["secret-eval-set".into()], &fixture.root, loader, &out_dir, &source).unwrap();
    assert_eq!(written, [record_path(&out_dir, "secret-eval-set")]);
    assert!(!cuts_path(&cuts_dir, "secret-eval-set").exists());

    let bin = env!("CARGO_BIN_EXE_turbo-eval");
    let cmp = Command::new(bin)
        .arg("cuts")
        .arg(&cuts_dir)
        .arg(&cuts_dir)
        .arg("--cuts-key-file")
        .arg(&key_file)
        .output()
        .unwrap();
    assert_eq!(cmp.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&cmp.stderr));
    assert!(String::from_utf8_lossy(&cmp.stdout).contains("secret-eval-set: 1 documents"));

    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let missing_dump = Command::new(bin)
        .args(["run", "--bundle"])
        .arg(&bundle)
        .args(["--sets", "scifact", "--cache"])
        .arg(&fixture.root)
        .arg("--out")
        .arg(&out_dir)
        .args(["--chunker", "turbo-chunker", "--max", "32", "--target", "20", "--cuts-key-file"])
        .arg(&key_file)
        .output()
        .unwrap();
    assert_eq!(missing_dump.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&missing_dump.stderr).contains("--cuts-key-file needs --dump-cuts <dir>"));
}
