//! The committed MASC Penn POS records (#1649): each is a receipt of the
//! recorded release, the peer's predictions pinned, and the numbers the
//! README gives.

use std::path::Path;

use turbo::manifest::{HeadKind, LicenseClass};
use turbo_train_recipes::masc::{RELEASE, Release};
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};

fn receipt(name: &str) -> Receipt {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/masc-pos-1649").join(name);
    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[test]
fn the_masc_pos_rows_are_receipts_of_the_recorded_release() {
    let Release::Recorded(release) = RELEASE else { panic!("MASC has a recorded release") };
    let (base, raw, mapped) =
        (receipt("majority.json"), receipt("spacy-trf-raw.json"), receipt("spacy-trf-mapped.json"));
    for r in [&base, &raw, &mapped] {
        assert_eq!(r.schema, SCHEMA);
        assert_eq!(r.corpus.id, "masc-pos");
        assert!(r.corpus.recorded);
        assert_eq!(r.corpus.corpus_sha256, release.corpus_sha256);
        assert_eq!((r.corpus.test.sentences, r.corpus.test.words), (6_384, 85_292));
        assert_eq!(round(r.eval.baseline.score()), 0.8964);
        assert!(r.head.is_none() && r.encoder.is_none(), "no model of ours or anyone's in a peer row");
    }
    for (r, score, mapped) in [(&raw, 0.8753, false), (&mapped, 0.8853, true)] {
        let p = r.peer.as_ref().unwrap();
        assert_eq!(p.name, "spacy-3.8.7-en_core_web_trf-3.8.0");
        assert_eq!(p.predictions.sha256, "0fd96b59eaca3bb84d8a577af73098ae2b7cc19456972ec0f285550da60ff8cb");
        assert_eq!(p.map.is_empty(), !mapped);
        assert_eq!(round(r.eval.model.as_ref().unwrap().score()), score);
        let c = r.eval.comparison.as_ref().unwrap();
        assert!(c.high < 0.0, "the peer is below the majority baseline on MASC's gold");
    }
    assert!(base.peer.is_none() && base.eval.model.is_none());
    for r in [&base, &raw, &mapped] {
        let text = serde_json::to_string(r).unwrap().to_lowercase();
        assert!(!text.contains("opennlp") && !text.contains("en-pos"), "no OpenNLP row (#1509)");
    }
}

/// The digest of the published head file, as `analysis/pins/models.txt`
/// pins it.
const HEAD_B_SHA256: &str = "76fac774d60210b9279a47ebcb565e12e2b353897f560afa4806e43771e8a3b6";

/// The #1825 peer rows: EWT peers, LitBank coref peers and the Re-DocRED
/// dependency-path rule, each a receipt of the recorded release with the
/// predictions pinned, the README's numbers, and no machine or path in it.
#[test]
fn the_1825_rows_are_receipts_of_the_recorded_releases() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/1825");
    let read = |name: &str| -> Receipt {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let ewt = [
        (
            "ewt-spacy-trf-upos.json",
            "ud-en-ewt-upos",
            0.9326,
            "a49403f2c17cdc3d3ee8d3d5f5e5f1d40d43f4e31c5c6a8893b6341605ba6905",
        ),
        (
            "ewt-spacy-trf-xpos.json",
            "ud-en-ewt-xpos",
            0.9532,
            "a49403f2c17cdc3d3ee8d3d5f5e5f1d40d43f4e31c5c6a8893b6341605ba6905",
        ),
        (
            "ewt-spacy-trf-feats.json",
            "ud-en-ewt-feats",
            0.6766,
            "83de81b75d12bd8ed9ccc62739309e94398a13c93061393546bfbbfdc2e7bc2a",
        ),
        (
            "ewt-spacy-trf-lemma.json",
            "ud-en-ewt-lemma",
            0.9531,
            "83de81b75d12bd8ed9ccc62739309e94398a13c93061393546bfbbfdc2e7bc2a",
        ),
        (
            "ewt-spacy-trf-deps.json",
            "ud-en-ewt-deps",
            0.3795,
            "83de81b75d12bd8ed9ccc62739309e94398a13c93061393546bfbbfdc2e7bc2a",
        ),
        (
            "ewt-stanza-upos.json",
            "ud-en-ewt-upos",
            0.9677,
            "53963d47819b1121cc523dcc7b95068e306a20f23c52df72f240e7a295276b18",
        ),
        (
            "ewt-stanza-xpos.json",
            "ud-en-ewt-xpos",
            0.9624,
            "53963d47819b1121cc523dcc7b95068e306a20f23c52df72f240e7a295276b18",
        ),
        (
            "ewt-stanza-feats.json",
            "ud-en-ewt-feats",
            0.9622,
            "53963d47819b1121cc523dcc7b95068e306a20f23c52df72f240e7a295276b18",
        ),
        (
            "ewt-stanza-lemma.json",
            "ud-en-ewt-lemma",
            0.9683,
            "53963d47819b1121cc523dcc7b95068e306a20f23c52df72f240e7a295276b18",
        ),
        (
            "ewt-stanza-deps.json",
            "ud-en-ewt-deps",
            0.8835,
            "53963d47819b1121cc523dcc7b95068e306a20f23c52df72f240e7a295276b18",
        ),
        (
            "ewt-trankit-upos.json",
            "ud-en-ewt-upos",
            0.9649,
            "8a6567908b6547475f30d802ed6fc1fc930218e40d50655ca6546a87a2a829d6",
        ),
        (
            "ewt-trankit-xpos.json",
            "ud-en-ewt-xpos",
            0.9700,
            "8a6567908b6547475f30d802ed6fc1fc930218e40d50655ca6546a87a2a829d6",
        ),
        (
            "ewt-trankit-feats.json",
            "ud-en-ewt-feats",
            0.8534,
            "8a6567908b6547475f30d802ed6fc1fc930218e40d50655ca6546a87a2a829d6",
        ),
        (
            "ewt-trankit-lemma.json",
            "ud-en-ewt-lemma",
            0.9138,
            "8a6567908b6547475f30d802ed6fc1fc930218e40d50655ca6546a87a2a829d6",
        ),
        (
            "ewt-trankit-deps.json",
            "ud-en-ewt-deps",
            0.8578,
            "8a6567908b6547475f30d802ed6fc1fc930218e40d50655ca6546a87a2a829d6",
        ),
    ];
    for (file, corpus, score, sha) in ewt {
        let r = read(file);
        assert_eq!(r.corpus.id, corpus, "{file}");
        assert!(r.corpus.recorded, "{file}");
        assert_eq!(r.corpus.test.words, 25_094, "{file}");
        assert_eq!(round(r.eval.model.as_ref().unwrap().score()), score, "{file}");
        assert!(r.peer.as_ref().unwrap().predictions.sha256 == sha, "{file}");
        assert_eq!(r.head, None, "{file}: a peer row carries no head of ours");
    }
    let singleton = read("litbank-singleton.json");
    assert!(singleton.eval.model.is_none());
    assert_eq!(round(singleton.eval.baseline.score()), 0.3009);
    for (file, score, sha) in [
        ("litbank-spacy-coref.json", 0.5058, "b0c515a6021bc6c77c56e1a90b81555f8a5fa66e379ad691fd7e542565a4da5b"),
        ("litbank-stanza-coref.json", 0.8051, "cfb02c756747d62eda9971567656aeefccacb55367e98a8a850ac4caaec22690"),
    ] {
        let r = read(file);
        assert_eq!(r.corpus.id, "litbank", "{file}");
        assert!(r.corpus.recorded, "{file}");
        assert_eq!(r.corpus.test.sentences, 759, "{file}");
        assert_eq!(round(r.eval.model.as_ref().unwrap().score()), score, "{file}");
        assert_eq!(r.peer.as_ref().unwrap().predictions.sha256, sha, "{file}");
        assert!(r.eval.comparison.as_ref().unwrap().low > 0.0, "{file}: above the singleton baseline");
    }
    let rule = read("redocred-dependency-path.json");
    assert_eq!(rule.corpus.id, "re-docred");
    assert!(rule.corpus.recorded);
    assert_eq!((rule.corpus.test.sentences, rule.corpus.test.words), (500, 98_734));
    assert_eq!(round(rule.eval.baseline.score()), 0.0640);
    assert!(rule.eval.model.is_none(), "a baseline row");
    assert_eq!(rule.head.as_ref().unwrap().kind, HeadKind::BiaffineParser, "the parser the rule ran on");
    assert_eq!(
        rule.encoder.as_ref().unwrap().manifest_sha256,
        "65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8"
    );
    assert_eq!(
        rule.outputs.iter().find(|o| o.path == "predictions.json").unwrap().sha256,
        "9ea91257dda03b94da53773c7327e2a6c1720c1f5dc45c79ee9702655c6d867d"
    );
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".json") {
            let r = read(&name);
            assert_eq!(r.host, "cpu-32-core-machine", "{name}: host {}", r.host);
            assert!(r.command.iter().all(|c| !c.starts_with('/')), "{name}: an absolute path in the command");
            let text = serde_json::to_string(&r).unwrap().to_lowercase();
            assert!(!text.contains("opennlp"), "{name}: no OpenNLP row (#1509)");
        }
    }
}

#[test]
fn the_head_row_clears_the_gate_on_the_recorded_release() {
    let Release::Recorded(release) = RELEASE else { panic!("MASC has a recorded release") };
    let (head, mapped) = (receipt("head-B.json"), receipt("spacy-trf-mapped.json"));
    assert_eq!(head.schema, SCHEMA);
    assert_eq!(head.corpus.corpus_sha256, release.corpus_sha256);
    assert_eq!((head.corpus.test.sentences, head.corpus.test.words), (6_384, 85_292));
    assert_eq!(round(head.eval.baseline.score()), 0.8964);
    let score = head.eval.model.as_ref().unwrap().score();
    assert_eq!(round(score), 0.9581);
    assert!(score >= 0.95, "the #1649 gate");
    assert!(score > mapped.eval.model.as_ref().unwrap().score(), "above the spaCy row");
    assert!(head.eval.comparison.as_ref().unwrap().low > 0.0, "above the majority baseline");
    let spec = head.head.as_ref().unwrap();
    assert_eq!(spec.kind, HeadKind::TokenTagger);
    assert_eq!(spec.license_class, Some(LicenseClass::Permissive));
    let encoder = head.encoder.as_ref().unwrap();
    assert_eq!(encoder.model, "answerdotai/ModernBERT-base");
    assert_eq!(encoder.manifest_sha256, "65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8");
    let file = head.outputs.iter().find(|f| f.path == "head.safetensors").unwrap();
    assert_eq!((file.size, file.sha256.as_str()), (13_426_024, HEAD_B_SHA256));
    let pins =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../analysis/pins/models.txt")).unwrap();
    assert!(pins.lines().any(|l| l.starts_with(&format!("blobs/sha256/{HEAD_B_SHA256}  file  "))), "pinned");
    let text = serde_json::to_string(&head).unwrap().to_lowercase();
    assert!(!text.contains("opennlp") && !text.contains("/work/"), "no OpenNLP row, no host paths");
}

#[test]
fn committed_receipts_name_no_machine() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/masc-pos-1649");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".json") {
            let r = receipt(&name);
            assert!(["cloud-container", "rtx-4080-machine"].contains(&r.host.as_str()), "{name}: host {}", r.host);
            assert!(r.command.iter().all(|c| !c.starts_with('/')), "{name}: an absolute path in the command");
        }
    }
}

/// The zoo's English part-of-speech entry is the head-B bundle: our own,
/// permissive, trained on MASC 3.0.0, and attributed with the receipt's
/// test score.
#[test]
fn the_builtin_english_pos_entry_is_the_head_b_bundle() {
    let head = receipt("head-B.json");
    let index = turbo_zoo::Index::builtin();
    let found: Vec<_> = index.find("en", turbo_zoo::Component::Pos).collect();
    assert_eq!(found.len(), 1, "one English POS entry");
    let a = found[0];
    assert_eq!(a.id.as_str(), "turbo-en-masc-pos-modernbert-base-1");
    assert_eq!((a.kind, a.format), (turbo_zoo::Kind::Head, turbo_zoo::Format::Safetensors));
    assert_eq!(a.class, turbo_zoo::LicenceClass::Permissive);
    assert_eq!(a.computed_class(), turbo_zoo::LicenceClass::Permissive);
    assert_eq!(a.trained_on.len(), 1);
    assert_eq!(a.trained_on[0].name, head.corpus.name);
    assert_eq!(a.trained_on[0].license.to_string(), head.corpus.spdx);
    let score = format!("test accuracy {:.4}", head.eval.model.as_ref().unwrap().score());
    let attribution = a.attribution.as_deref().unwrap();
    assert!(attribution.starts_with(head.corpus.attribution.trim()), "{attribution}");
    assert!(attribution.contains(&score), "{attribution} lacks {score}");
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"]
    );
    // The head adds its tensors to the encoder's weights file.
    let weights = a.files.iter().find(|f| f.path.as_str() == "weights/model.safetensors").unwrap();
    let head_file = head.outputs.iter().find(|o| o.path == "head.safetensors").unwrap();
    assert!(weights.size > head_file.size);
}
