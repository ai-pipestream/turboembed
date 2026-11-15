//! Verification of the committed Nemotron-PII records and zoo entry (#1888).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use turbo::manifest::{HeadKind, LicenseClass, Scheme};
use turbo_tagger::NEMOTRON_PII_TEMPERATURE;
use turbo_trainer_recipes::eval::{Metrics, SpanScore};
use turbo_trainer_recipes::nemotron_pii::{ID, README, TEST, TRAIN, TYPES};
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};
use turbo_trainer_recipes::tagger_calibrate::{SCHEMA as CALIBRATION_SCHEMA, TaggerCalibrationReport};

const HEAD_SHA256: &str = "b8cb82e3708629e101990487add3dea807b2d56fb2cda15cbcfd06dcb054f9e6";
const HEAD_JSON_SHA256: &str = "144017d505e46ed88b51b25443c992518a119d63608563a1ca01adc3e1e48696";
const HEAD_RECEIPT_SHA256: &str = "87b4482b49a7f2cb60c4a006bef7e961fe38948b1d1a760c550e84b165c05f2b";
const SPACY_PRED_SHA256: &str = "623bff9da10b8289729a0d59b421a158c49d49912c87b68835c93d02db6306da";
const STANZA_PRED_SHA256: &str = "d1d5f9542e5e18cbdeb7bba0dd80faf13f28e9b9a39636dea59e2a7a2ed3ff40";
const ZOO_ID: &str = "turbo-en-nemotron-pii-modernbert-base-1";

fn records_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("records/nemotron-pii-1888")
}

fn read_receipt(name: &str) -> Receipt {
    let p = records_dir().join(name);
    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn per_type(m: &Metrics) -> &BTreeMap<String, SpanScore> {
    let Metrics::Spans { per_type, .. } = m else {
        panic!("expected Metrics::Spans");
    };
    per_type
}

#[test]
fn nemotron_pii_receipts_and_peers_match_recorded_release_and_hygiene_rules() {
    let dir = records_dir();
    let mut json_count = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if !name.ends_with(".json") {
            continue;
        }
        json_count += 1;
        let bytes = std::fs::read(dir.join(&name)).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(raw.get("host").is_none(), "{name}: host field present");
        let cmd = raw["command"].as_array().expect("command array");
        assert_eq!(cmd[0].as_str(), Some("turbo-train-head"), "{name}");
        assert!(cmd.iter().all(|c| !c.as_str().unwrap_or("").starts_with('/')), "{name}: absolute path in command");
        let text = String::from_utf8(bytes).unwrap();
        let lower = text.to_lowercase();
        assert!(!text.contains('\u{2014}'), "{name}: em dash present");
        let forbidden_ref = format!("open{}", "nlp");
        assert!(!lower.contains(&forbidden_ref), "{name}: forbidden reference");
        for root in ["work", "home", "users"] {
            let prefix = format!("/{root}/");
            assert!(!lower.contains(&prefix), "{name}: host path {prefix}");
        }
    }
    assert_eq!(json_count, 6);

    let majority = read_receipt("majority.json");
    let spacy = read_receipt("spacy-lg.json");
    let stanza = read_receipt("stanza-en.json");
    let head = read_receipt("head.json");

    let recorded_shas = [README.sha256, TRAIN.sha256, TEST.sha256];
    for (name, r) in [("majority", &majority), ("spacy", &spacy), ("stanza", &stanza), ("head", &head)] {
        assert_eq!(r.schema, SCHEMA, "{name}");
        assert_eq!(r.corpus.id, ID, "{name}");
        assert!(r.corpus.recorded, "{name}");
        let file_shas: Vec<&str> = r.corpus.files.iter().map(|f| f.sha256.as_str()).collect();
        assert_eq!(file_shas, recorded_shas, "{name}");
        assert_eq!((r.corpus.train.sentences, r.corpus.train.words), (90_000, 17_631_946), "{name}");
        assert_eq!((r.corpus.dev.sentences, r.corpus.dev.words), (10_000, 1_969_973), "{name}");
        assert_eq!((r.corpus.test.sentences, r.corpus.test.words), (100_000, 18_710_937), "{name}");
        assert_eq!(round4(r.eval.baseline.score()), 0.1677, "{name}");
    }

    assert!(majority.eval.model.is_none() && majority.peer.is_none());

    for (r, peer_name, sha, score) in [
        (&spacy, "spacy-3.8.7-en_core_web_lg-3.8.0", SPACY_PRED_SHA256, 0.1319),
        (&stanza, "stanza-1.10.1-en", STANZA_PRED_SHA256, 0.1765),
    ] {
        let p = r.peer.as_ref().unwrap();
        assert_eq!(p.name, peer_name);
        assert_eq!(p.predictions.sha256, sha);
        assert_eq!(p.map.len(), 8);
        assert_eq!(round4(r.eval.model.as_ref().unwrap().score()), score);
        assert!(r.head.is_none() && r.encoder.is_none());
    }

    let head_score = head.eval.model.as_ref().unwrap().score();
    assert_eq!(round4(head_score), 0.6096);
    let cmp = head.eval.comparison.as_ref().unwrap();
    assert!(cmp.delta > 0.44 && cmp.low > 0.44);
    assert!(head_score > spacy.eval.model.as_ref().unwrap().score());
    assert!(head_score > stanza.eval.model.as_ref().unwrap().score());

    // Our head beats majority and both peers on every one of the 4 shared OntoNotes labels.
    let head_spans = per_type(head.eval.model.as_ref().unwrap());
    let maj_spans = per_type(&head.eval.baseline);
    let spacy_spans = per_type(spacy.eval.model.as_ref().unwrap());
    let stanza_spans = per_type(stanza.eval.model.as_ref().unwrap());
    for shared in ["company_name", "date", "time", "language"] {
        let ours = head_spans[shared].f1;
        assert!(ours > maj_spans[shared].f1, "{shared}: {ours} vs majority {}", maj_spans[shared].f1);
        assert!(ours > spacy_spans[shared].f1, "{shared}: {ours} vs spacy {}", spacy_spans[shared].f1);
        assert!(ours > stanza_spans[shared].f1, "{shared}: {ours} vs stanza {}", stanza_spans[shared].f1);
    }

    let spec = head.head.as_ref().unwrap();
    assert_eq!(spec.kind, HeadKind::TokenTagger);
    assert_eq!(spec.scheme, Scheme::Bio);
    assert_eq!(spec.license_class, Some(LicenseClass::Permissive));
    assert_eq!(spec.labels.len(), 1 + 2 * TYPES.len());
    assert_eq!(head.training.as_ref().unwrap().best_epoch, 11);

    let enc = head.encoder.as_ref().unwrap();
    assert_eq!(enc.model, "answerdotai/ModernBERT-base");
    assert_eq!(enc.manifest_sha256, "65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8");

    let weights = head.outputs.iter().find(|f| f.path == "head.safetensors").unwrap();
    assert_eq!((weights.size, weights.sha256.as_str()), (6_201_372, HEAD_SHA256));
    let spec_out = head.outputs.iter().find(|f| f.path == "head.json").unwrap();
    assert_eq!((spec_out.size, spec_out.sha256.as_str()), (2_671, HEAD_JSON_SHA256));

    let pins =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../analysis/pins/models.txt")).unwrap();
    assert!(pins.lines().any(|l| l.starts_with(&format!("blobs/sha256/{HEAD_SHA256}  file  "))));
}

#[test]
fn nemotron_pii_calibration_and_zoo_entry_match_head_receipt() {
    let dir = records_dir();
    let cal_bytes = std::fs::read(dir.join("calibration.json")).unwrap();
    let cal: TaggerCalibrationReport = serde_json::from_slice(&cal_bytes).unwrap();
    assert_eq!(cal.schema, CALIBRATION_SCHEMA);
    assert_eq!(cal.corpus_id, ID);
    assert_eq!(cal.split, "dev");
    assert_eq!(cal.layer, 22);
    assert_eq!((cal.dev_sentences, cal.dev_tokens, cal.dev_entity_tokens), (10_000, 1_969_973, 380_348));
    assert!((cal.temperature as f32 - NEMOTRON_PII_TEMPERATURE).abs() < 1e-6);
    assert!(cal.calibrated.nll < cal.uncalibrated.nll);
    assert!(cal.calibrated.ece < cal.uncalibrated.ece);
    assert!(cal.calibrated.entity_ece < cal.uncalibrated.entity_ece);

    let head_bytes = std::fs::read(dir.join("head.json")).unwrap();
    assert_eq!(turbo_pins::sha256_hex(&head_bytes), HEAD_RECEIPT_SHA256);
    assert_eq!(cal.head_files.iter().find(|f| f.path == "head.safetensors").unwrap().sha256, HEAD_SHA256);
    assert_eq!(cal.head_files.iter().find(|f| f.path == "head.json").unwrap().sha256, HEAD_JSON_SHA256);
    assert_eq!(cal.head_files.iter().find(|f| f.path == "receipt.json").unwrap().sha256, HEAD_RECEIPT_SHA256);

    let fit_raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("fit.json")).unwrap()).unwrap();
    assert_eq!(fit_raw["schema"].as_str(), Some("turboembed.fit-cache.v1"));
    assert_eq!(fit_raw["layers"].as_array().unwrap(), &[serde_json::json!(22)]);

    let head = read_receipt("head.json");
    let index = turbo_zoo::Index::builtin();
    let found: Vec<_> = index.find("en", turbo_zoo::Component::PiiDetector).collect();
    assert_eq!(found.len(), 1);
    let a = found[0];
    assert_eq!(a.id.as_str(), ZOO_ID);
    assert_eq!((a.kind, a.format), (turbo_zoo::Kind::Head, turbo_zoo::Format::Safetensors));
    assert_eq!(a.class, turbo_zoo::LicenceClass::Permissive);
    assert_eq!(a.computed_class(), turbo_zoo::LicenceClass::Permissive);
    assert_eq!(a.trained_on.len(), 1);
    assert_eq!(a.trained_on[0].name, head.corpus.name);
    assert_eq!(a.trained_on[0].license.to_string(), head.corpus.spdx);
    let score = format!("test span_f1 {:.4}", head.eval.model.as_ref().unwrap().score());
    let attribution = a.attribution.as_deref().unwrap();
    assert!(attribution.starts_with(head.corpus.attribution.trim()), "{attribution}");
    assert!(attribution.contains(&score), "{attribution} lacks {score}");
    let weights = a.files.iter().find(|f| f.path.as_str() == "weights/model.safetensors").unwrap();
    let head_file = head.outputs.iter().find(|o| o.path == "head.safetensors").unwrap();
    assert!(weights.size > head_file.size);
}
