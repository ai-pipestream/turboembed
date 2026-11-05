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
    let readme = std::fs::read_to_string(dir.join("README.md")).unwrap();
    assert!(
        readme.contains("| Stanza 1.10.1 coref (contaminated) |") && readme.contains("`udcoref_xlm-roberta-lora`"),
        "the Stanza coref row names its package and is labelled contaminated (CorefUD carries LitBank)"
    );
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
            let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(&name)).unwrap()).unwrap();
            assert!(raw.get("host").is_none(), "{name}: host field present");
            let r = read(&name);
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
    assert_eq!(head.training.as_ref().unwrap().best_epoch, 32);
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
            let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(&name)).unwrap()).unwrap();
            assert!(raw.get("host").is_none(), "{name}: host field present");
            let r = receipt(&name);
            assert!(r.command.iter().all(|c| !c.starts_with('/')), "{name}: an absolute path in the command");
        }
    }
}

/// The #1803 MarkIT split-head sweep records: receipts of the recorded
/// r2.16 files, the gate cleared by the two runs the README names, and
/// no machine or path leaked into a committed receipt.
#[test]
fn the_markit_split_sweep_records_are_receipts_of_the_recorded_release() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/1803");
    let read = |name: &str| -> Receipt {
        serde_json::from_slice(&std::fs::read(dir.join(name)).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", dir.join(name).display()))
    };
    let full = read("full1.json");
    assert_eq!(full.schema, SCHEMA);
    assert_eq!(full.corpus.id, "ud-it-markit-split");
    assert!(full.corpus.recorded);
    assert_eq!(full.corpus.train.sentences, 611);
    assert_eq!((full.corpus.test.sentences, full.corpus.test.words), (340, 9_654));
    let h = full.head.as_ref().unwrap();
    assert_eq!(h.kind, HeadKind::TokenSplit);
    assert_eq!(h.license_class, Some(LicenseClass::Permissive));
    let seen = h.seen.as_ref().expect("the seen lookup is the head's");
    assert_eq!(seen.min_count, 1);
    assert_eq!(seen.tokens.len(), 4_122);
    assert!(seen.tokens.windows(2).all(|w| w[0].token < w[1].token), "sorted by token");
    assert!(seen.tokens.iter().all(|e| h.labels.contains(&e.label)), "every label is the head's");
    let c = full.eval.comparison.as_ref().unwrap();
    assert!(c.delta > 0.0 && c.low > 0.0, "the gate: the whole interval above zero");
    assert_eq!(round(full.eval.baseline.score()), 0.9925);
    assert_eq!(round(full.eval.model.as_ref().unwrap().score()), 0.9939);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".json") {
            let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(&name)).unwrap()).unwrap();
            assert!(raw.get("host").is_none(), "{name}: host field present");
            let r = read(&name);
            assert!((1..=r.training.as_ref().unwrap().dev.len()).contains(&r.training.as_ref().unwrap().best_epoch));
            assert!(r.command.iter().all(|c| !c.starts_with('/')), "{name}: an absolute path in the command");
            let text = serde_json::to_string(&r).unwrap().to_lowercase();
            assert!(!text.contains("opennlp"), "{name}: no OpenNLP row (#1509)");
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

/// The UD head records of #1523 (EWT, share-alike), #1522 (the CC BY
/// treebanks, default tier) and #1651 (the CC BY-SA ones, share-alike): receipts of the recorded releases, each head's
/// licence class, the gate cleared with the whole interval above zero,
/// the scores the READMEs give, and no machine or path in a receipt.
#[test]
fn the_ud_head_records_are_receipts_of_the_recorded_releases() {
    let rows = [
        ("ewt-1523", "deps.json", "ud-en-ewt-deps", LicenseClass::ShareAlike, 0.7901, 0.3322),
        ("ewt-1523", "feats.json", "ud-en-ewt-feats", LicenseClass::ShareAlike, 0.9531, 0.8292),
        ("ewt-1523", "lemma.json", "ud-en-ewt-lemma", LicenseClass::ShareAlike, 0.9543, 0.9503),
        ("ewt-1523", "upos-l12.json", "ud-en-ewt-upos", LicenseClass::ShareAlike, 0.9427, 0.8617),
        ("ewt-1523", "upos.json", "ud-en-ewt-upos", LicenseClass::ShareAlike, 0.9457, 0.8617),
        ("ewt-1523", "xpos.json", "ud-en-ewt-xpos", LicenseClass::ShareAlike, 0.9408, 0.8381),
        ("ud-1522", "ca-ancora-deps.json", "ud-ca-ancora-deps", LicenseClass::Permissive, 0.9128, 0.4087),
        ("ud-1522", "ca-ancora-lemma.json", "ud-ca-ancora-lemma", LicenseClass::Permissive, 0.9861, 0.9701),
        ("ud-1522", "ca-ancora-upos.json", "ud-ca-ancora-upos", LicenseClass::Permissive, 0.9882, 0.9198),
        ("ud-1522", "es-ancora-deps.json", "ud-es-ancora-deps", LicenseClass::Permissive, 0.8993, 0.3873),
        ("ud-1522", "es-ancora-lemma.json", "ud-es-ancora-lemma", LicenseClass::Permissive, 0.9864, 0.9677),
        ("ud-1522", "es-ancora-upos.json", "ud-es-ancora-upos", LicenseClass::Permissive, 0.9878, 0.9176),
        ("ud-1522", "fi-ftb-deps.json", "ud-fi-ftb-deps", LicenseClass::Permissive, 0.8537, 0.3849),
        ("ud-1522", "fi-ftb-lemma.json", "ud-fi-ftb-lemma", LicenseClass::Permissive, 0.8208, 0.7928),
        ("ud-1522", "fi-ftb-upos.json", "ud-fi-ftb-upos", LicenseClass::Permissive, 0.9463, 0.82),
        ("ud-1522", "it-markit-deps.json", "ud-it-markit-deps", LicenseClass::Permissive, 0.7733, 0.3305),
        ("ud-1522", "it-markit-lemma.json", "ud-it-markit-lemma", LicenseClass::Permissive, 0.831, 0.8172),
        ("ud-1522", "it-markit-upos.json", "ud-it-markit-upos", LicenseClass::Permissive, 0.9555, 0.8441),
        ("ud-1651", "de-gsd-lemma.json", "ud-de-gsd-lemma", LicenseClass::ShareAlike, 0.9516, 0.9424),
        ("ud-1651", "de-gsd-upos.json", "ud-de-gsd-upos", LicenseClass::ShareAlike, 0.962, 0.8865),
        ("ud-1651", "fr-gsd-lemma.json", "ud-fr-gsd-lemma", LicenseClass::ShareAlike, 0.9741, 0.9593),
        ("ud-1651", "fr-gsd-upos.json", "ud-fr-gsd-upos", LicenseClass::ShareAlike, 0.9803, 0.9132),
        ("ud-1651", "nl-alpino-lemma.json", "ud-nl-alpino-lemma", LicenseClass::ShareAlike, 0.916, 0.9076),
        ("ud-1651", "nl-alpino-upos.json", "ud-nl-alpino-upos", LicenseClass::ShareAlike, 0.9624, 0.8806),
        ("ud-1651", "pt-gsd-lemma.json", "ud-pt-gsd-lemma", LicenseClass::ShareAlike, 0.9033, 0.8761),
        ("ud-1651", "pt-gsd-upos.json", "ud-pt-gsd-upos", LicenseClass::ShareAlike, 0.9738, 0.8803),
        ("ud-1651", "ru-gsd-lemma.json", "ud-ru-gsd-lemma", LicenseClass::ShareAlike, 0.8996, 0.7969),
        ("ud-1651", "ru-gsd-upos.json", "ud-ru-gsd-upos", LicenseClass::ShareAlike, 0.9721, 0.8072),
    ];
    for (dir, name, corpus, class, model, baseline) in rows {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("records").join(dir).join(name);
        let bytes = std::fs::read(&p).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(raw.get("host").is_none(), "{name}: host field present");
        let r: Receipt = serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(r.schema, SCHEMA, "{name}");
        assert_eq!(r.corpus.id, corpus, "{name}");
        assert!(r.corpus.recorded, "{name}: not the recorded release");
        assert_eq!(r.command[0], "turbo-train-head", "{name}");
        assert!(r.command.iter().all(|c| !c.starts_with('/')), "{name}: an absolute path in the command");
        assert_eq!(r.head.as_ref().unwrap().license_class, Some(class), "{name}");
        let t = r.training.as_ref().unwrap();
        assert!((1..=t.dev.len()).contains(&t.best_epoch), "{name}: best epoch {}", t.best_epoch);
        let c = r.eval.comparison.as_ref().unwrap();
        assert!(c.delta > 0.0 && c.low > 0.0, "{name}: the gate, the whole interval above zero");
        assert_eq!(round(r.eval.model.as_ref().unwrap().score()), model, "{name}");
        assert_eq!(round(r.eval.baseline.score()), baseline, "{name}");
        let text = String::from_utf8(bytes).unwrap().to_lowercase();
        assert!(!text.contains("opennlp") && !text.contains("/work/"), "{name}: no OpenNLP row, no host paths");
    }
}
