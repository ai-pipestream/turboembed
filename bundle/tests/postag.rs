//! `postag-convert` and `postag-verify` on the six BiLSTM runs of the
//! `postag-neural` goldens: each converted bundle tags the training
//! sentences and the fixed token list exactly as the `ONLP-BLPT` file
//! does, with either numerics, and a bundle whose digests do not match is
//! refused.

#[allow(dead_code)]
#[path = "../../analysis/postag/tests/neural_goldens/golden.rs"]
mod golden;
#[allow(dead_code)]
#[path = "../../analysis/postag/tests/neural_goldens/run.rs"]
mod run;

use std::fs;
use std::path::PathBuf;
use turbo_bundle::postag;
use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, Numerics};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-postag-bundle-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn tags(model: &BilstmPosModel, numerics: Numerics, sentences: &[Vec<String>]) -> Vec<Vec<String>> {
    let tagger = BilstmPosTagger::with_numerics(model.clone(), numerics).unwrap();
    sentences.iter().map(|s| tagger.tag(s).unwrap()).collect()
}

#[test]
fn a_converted_bundle_tags_every_bilstm_golden_as_the_model_file() {
    let mut converted = 0;
    for r in run::runs().iter().filter(|r| r.is_bilstm()) {
        let dir = scratch(&r.name);
        postag::convert(&r.bytes, "Apache-2.0", &dir).unwrap_or_else(|e| panic!("{}: {e}", r.name));
        let m = postag::verify(&dir, Some(&r.bytes)).unwrap();
        assert_eq!(m.source_sha256, golden::sha256_hex(&r.bytes), "{}", r.name);
        let source = BilstmPosModel::load(r.bytes.as_slice()).unwrap();
        assert_eq!((m.crf, m.stacked, m.adapter), (source.is_crf(), source.is_stacked(), source.has_adapter()));
        let bundle = postag::load(&dir, false).unwrap();
        assert_eq!(bundle.vocabularies(), source.vocabularies(), "{}: vocabularies", r.name);
        let mut sentences = r.train_sentences.clone();
        sentences.extend(golden::sentences(&r.tokens));
        for numerics in [Numerics::JvmParity, Numerics::Packed] {
            assert_eq!(
                tags(&bundle, numerics, &sentences),
                tags(&source, numerics, &sentences),
                "{}: {numerics:?} tags",
                r.name
            );
        }
        // The F16 copy loads into the same shapes; its tags are not held.
        postag::load(&dir, true).unwrap();
        fs::remove_dir_all(&dir).unwrap();
        converted += 1;
    }
    assert_eq!(converted, 6, "the BiLSTM runs");
}

#[test]
fn verify_refuses_a_bundle_whose_digests_do_not_match() {
    let r = run::runs().into_iter().find(|r| r.name == "bilstm-crf").unwrap();
    let dir = scratch("tampered");
    postag::convert(&r.bytes, "Apache-2.0", &dir).unwrap();
    assert!(postag::convert(&r.bytes, "Apache-2.0", &dir).is_err(), "writes only a new directory");

    let mut other = r.bytes.clone();
    *other.last_mut().unwrap() ^= 1;
    let e = postag::verify(&dir, Some(&other)).unwrap_err();
    assert!(e.contains("converted from"), "{e}");

    let path = dir.join(postag::MANIFEST);
    let text = fs::read_to_string(&path).unwrap();
    let m = postag::verify(&dir, None).unwrap();
    let sha = &m.files["weights.safetensors"];
    fs::write(&path, text.replace(sha.as_str(), &"0".repeat(64))).unwrap();
    let e = postag::verify(&dir, None).unwrap_err();
    assert!(e.contains("weights.safetensors: SHA-256"), "{e}");
    assert!(postag::load(&dir, false).is_err());

    fs::write(&path, &text).unwrap();
    fs::write(dir.join("vocab.json"), b"{}").unwrap();
    assert!(postag::verify(&dir, None).unwrap_err().contains("vocab.json"));
    fs::remove_dir_all(&dir).unwrap();
}
