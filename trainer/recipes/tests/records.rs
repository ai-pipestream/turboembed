//! The committed MASC Penn POS records (#1649): each is a receipt of the
//! recorded release, the peer's predictions pinned, and the numbers the
//! README gives.

use std::path::Path;

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
