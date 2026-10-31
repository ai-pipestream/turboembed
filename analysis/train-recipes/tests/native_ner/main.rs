mod common;

use std::collections::HashSet;
use std::fs;

use common::{Scratch, corpus, document, tiny, trained_bundle};
use turbo_analysis::{Annotator, Ix, Layer};
use turbo_analysis_embed::{EncoderOptions, Precision};
use turbo_tagger::{EntityTagger, WordEncoder};
use turbo_train_recipes::native_ner::{Partition, fit_cpu, head, prepare};
use turbo_trainer::Config;

#[test]
fn split_preserves_test_and_separates_development_before_annotation_filtering() {
    let root = Scratch::new("split");
    corpus(&root.0);
    let data = prepare(&root.0).unwrap();
    let test: Vec<_> =
        data.documents().iter().filter(|d| d.partition == Partition::Test).map(|d| d.path.as_str()).collect();
    assert_eq!(test, ["doc009.hdr", "doc019.hdr", "doc029.hdr"]);
    let unique: HashSet<_> = data.documents().iter().map(|d| &d.path).collect();
    assert_eq!(unique.len(), 30);
    assert_eq!((data.samples(Partition::Train).len(), data.samples(Partition::Dev).len()), (25, 2));
    let header = root.0.join("doc000.hdr");
    let original = fs::read_to_string(&header).unwrap();
    let needle = r#"<annotation f.id="f.ne" loc="doc000-ne.xml"/>"#;
    assert!(original.contains(needle));
    fs::write(&header, original.replace(needle, "")).unwrap();
    let without = prepare(&root.0).unwrap();
    assert_eq!(without.samples(Partition::Train).len(), 24);
    assert!(!without.documents()[0].annotated);
    assert_ne!(data.corpus_sha256(), without.corpus_sha256());
    assert_ne!(data.membership_sha256(), without.membership_sha256());
    assert_eq!(
        without.documents().iter().filter(|d| d.partition == Partition::Test).map(|d| &d.path).collect::<Vec<_>>(),
        test
    );
}

#[test]
fn unsupported_and_discontinuous_annotations_are_refused() {
    let root = Scratch::new("refuse");
    corpus(&root.0);
    for (kind, positions, message) in
        [("money", "0", "unsupported"), ("person", "0 3", "discontinuous"), ("person", "0 1 2", "overlapping entity")]
    {
        document(&root.0, 0, kind, positions);
        let error = prepare(&root.0).err().expect("invalid annotation must fail").to_string();
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn invalid_unicode_token_offsets_are_refused() {
    let root = Scratch::new("offsets");
    corpus(&root.0);
    let path = root.0.join("doc000-seg.xml");
    let xml = fs::read_to_string(&path).unwrap();
    // Cut the crab token inside
    // the surrogate pair rather than at a valid character boundary.
    let start = common::TEXT[..common::TEXT.find('🦀').unwrap()].encode_utf16().count();
    let valid = format!("anchors=\"{start} {}\"", start + 2);
    assert!(xml.contains(&valid));
    fs::write(path, xml.replace(&valid, &format!("anchors=\"{} {}\"", start + 1, start + 2))).unwrap();
    assert!(prepare(&root.0).is_err());
}

#[test]
fn licence_exclusions_do_not_move_test_documents() {
    let root = Scratch::new("excluded");
    corpus(&root.0);
    let header = root.0.join("doc000.hdr");
    let text = fs::read_to_string(&header).unwrap();
    fs::write(header, text.replace("</documentHeader>", "<textClass catRef=\"TW\"/></documentHeader>")).unwrap();
    let prepared = prepare(&root.0).unwrap();
    assert_eq!(prepared.excluded().len(), 1);
    assert_eq!(prepared.excluded()[0].path, "doc000.hdr");
    let test: Vec<_> =
        prepared.documents().iter().filter(|d| d.partition == Partition::Test).map(|d| d.path.as_str()).collect();
    assert_eq!(test, ["doc009.hdr", "doc019.hdr", "doc029.hdr"]);
}

#[test]
fn generated_encoder_cpu_smoke_trains_deterministically_and_keeps_unicode_offsets() {
    let root = Scratch::new("smoke");
    corpus(&root.0);
    let data = prepare(&root.0).unwrap();
    let config = Config {
        threads: 2,
        lanes: 2,
        epochs: 35,
        batch: 8,
        lr: 0.03,
        weight_decay: 0.0,
        patience: 0,
        ..Config::default()
    };
    let first = fit_cpu(&data, &tiny(), &config).unwrap();
    let again = fit_cpu(&data, &tiny(), &config).unwrap();
    let bytes =
        |m: &turbo_trainer::Tagger<f32>| m.params().iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    assert_eq!(bytes(&first.model), bytes(&again.model));
    assert_eq!(first.identity, again.identity);
    // Wall time is now part of Report; only the learned result is deterministic.
    assert_eq!(first.report.train_loss, again.report.train_loss);
    assert_eq!(first.report.dev, again.report.dev);
    assert_eq!(first.report.best_epoch, again.report.best_epoch);
    for report in [&first.report, &again.report] {
        assert_eq!(report.epoch_seconds.len(), report.dev.len());
        assert!(report.epoch_seconds.iter().all(|t| t.is_finite() && *t >= 0.0));
    }
    assert!(first.report.train_loss.first() > first.report.train_loss.last(), "{:?}", first.report);
    assert_eq!(first.report.dev[first.report.best_epoch], 1.0, "{:?}", first.report);
    assert_eq!(
        first.identity.encoder_sha256,
        turbo::bundle::sha256_hex(&fs::read(tiny().join("manifest.json")).unwrap())
    );
    // An untouched test partition can affect provenance, never fitting.
    let held_out = root.0.join("doc029.txt");
    fs::write(&held_out, common::TEXT.replace("Ánn", "Zoe")).unwrap();
    let changed_data = prepare(&root.0).unwrap();
    let changed_test = fit_cpu(&changed_data, &tiny(), &config).unwrap();
    assert_eq!(bytes(&first.model), bytes(&changed_test.model));
    assert_ne!(first.identity.corpus_sha256, changed_test.identity.corpus_sha256);
    assert_ne!(first.identity.membership_sha256, changed_test.identity.membership_sha256);
    let changed_seed = fit_cpu(&data, &tiny(), &Config { seed: config.seed + 1, ..config }).unwrap();
    assert_ne!(first.identity.config_sha256, changed_seed.identity.config_sha256);
    let options = EncoderOptions { precision: Precision::Exact, cpu_threads: 1, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(&tiny(), &options).unwrap();
    let s = &data.samples(Partition::Test)[0];
    let seq = vec![(0..s.labels().len()).collect()];
    let features = encoder.token_features(s.view(), s.tokens(), &seq, None, head().word_pooling).unwrap();
    let core = first.model.to_token_tagger().unwrap();
    let mut scores = Vec::new();
    let mut tags = Vec::new();
    let words: Vec<u32> = (0..s.labels().len() as u32).collect();
    core.scores(&features[0], &words, &mut scores, &mut Default::default()).unwrap();
    core.decode(&scores, words.len(), &mut tags).unwrap();
    assert_eq!(tags, s.labels());
    let mut training_work = turbo_trainer::tagger::Work::default();
    first.model.forward(&features[0], &mut training_work).unwrap();
    assert!(scores.iter().zip(training_work.scores()).all(|(a, b)| (a - b).abs() < 1e-5));
    let bundle = Scratch::new("trained-fixture");
    trained_bundle(&first.model, &bundle.0);
    let annotator = EntityTagger::open(&bundle.0, None, &options).unwrap();
    let (entities,) = annotator.annotate((s.view(), s.tokens(), None, None)).unwrap();
    entities.check().unwrap();
    let text: Vec<_> = entities.spans.iter().map(|(_, p)| &s.view().text()[p.range()]).collect();
    assert_eq!(text, ["Ánn", "Bob", "Paris", "Acme", "May", "Zoë"]);
    assert_eq!(entities.spans.get(Ix::new(0)).end, 4);
    for (_, span) in entities.spans.iter() {
        assert_eq!(s.view().span_from_utf16(s.view().span_to_utf16(span)).unwrap(), span);
        assert_eq!(s.view().span_from_code_point(s.view().span_to_code_point(span)).unwrap(), span);
    }
}
