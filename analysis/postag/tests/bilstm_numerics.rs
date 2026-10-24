//! The two numerics of `BilstmPosTagger` through its public API: CRF
//! weights too large for plain f64 still give the exact scorer's
//! probabilities, never a non-finite one, and each numerics has its own
//! stable configuration hash.

mod bilstm_support;

use bilstm_support::{model, model_with};
use turbo_analysis::Annotator;
use turbo_postag::neural::{BilstmPosTagger, Numerics};

fn taggers(transition: f64) -> [BilstmPosTagger; 2] {
    [BilstmPosTagger::new(model(transition)), BilstmPosTagger::jvm_parity(model(transition))]
}

fn assert_halves(probs: &[f64], case: &str) {
    assert_eq!(probs.len(), 3, "{case}");
    for p in probs {
        assert!((p - 0.5).abs() < 1e-12, "{case}: {probs:?}");
    }
}

#[test]
fn huge_crf_weights_give_the_exact_marginals_in_both_numerics() {
    for transition in [1.0, 1e300, f64::MAX, -1e300, -f64::MAX] {
        for tagger in taggers(transition) {
            let case = format!("{:?}, transitions {transition:e}", tagger.numerics());
            let sentence = ["x", "y", "z"];
            let best = tagger.top_k_sequences(&sentence).unwrap().remove(0);
            assert_halves(best.probs(), &case);
            let batch = tagger.best_sequences(&[&sentence[..], &["w"], &sentence[..]]).unwrap();
            assert_halves(batch[0].probs(), &case);
            assert_halves(batch[2].probs(), &case);
            assert_eq!(tagger.tag(&sentence).unwrap(), best.outcomes(), "{case}");
        }
    }
}

#[test]
fn config_hash_separates_the_numerics_and_repeats() {
    let [packed, parity] = taggers(1.0);
    assert_eq!((packed.numerics(), parity.numerics()), (Numerics::Packed, Numerics::JvmParity));
    assert_ne!(packed.config_hash(), parity.config_hash());
    let [packed2, parity2] = taggers(1.0);
    assert_eq!((packed.config_hash(), parity.config_hash()), (packed2.config_hash(), parity2.config_hash()));
    let other = BilstmPosTagger::with_numerics(model(2.0), Numerics::Packed).unwrap();
    assert_ne!(packed.config_hash(), other.config_hash());
    // The same in every run and build: FNV-1a, no random state.
    let sha = packed.model().model_ref().sha256.clone();
    let fnv = |tag: &[u8]| turbo_analysis::Fnv1a::new().bytes(tag).bytes(b"\0").bytes(sha.as_bytes()).finish();
    assert_eq!((packed.config_hash(), parity.config_hash()), (fnv(b"packed"), fnv(b"jvm-parity")));
}

#[test]
fn weights_past_f32_fall_back_to_f64_instead_of_failing() {
    // A char LSTM bias f32 cannot hold: the weights do not pack, so `new`
    // takes the f64 reference and `with_numerics` says why.
    let unpackable = model_with(0.0, 1e39, 0.0, 0.0);
    assert!(BilstmPosTagger::with_numerics(unpackable.clone(), Numerics::Packed).is_err());
    let tagger = BilstmPosTagger::new(unpackable.clone());
    assert_eq!(tagger.numerics(), Numerics::JvmParity);
    assert_eq!(tagger.config_hash(), BilstmPosTagger::jvm_parity(unpackable).config_hash());
    assert_halves(tagger.top_k_sequences(&["x", "y", "z"]).unwrap()[0].probs(), "unpackable");

    // A char embedding of 1e300 with zero char weights: infinite in f32, and
    // a NaN projection once multiplied by zero. `new` must not panic.
    let huge_char = model_with(0.0, 0.0, 0.0, 1e300);
    assert!(BilstmPosTagger::with_numerics(huge_char.clone(), Numerics::Packed).is_err());
    let tagger = BilstmPosTagger::new(huge_char.clone());
    assert_eq!(tagger.numerics(), Numerics::JvmParity);
    let parity = BilstmPosTagger::jvm_parity(huge_char);
    assert_eq!(tagger.tag(&["x"]).unwrap(), parity.tag(&["x"]).unwrap());
    assert_halves(tagger.top_k_sequences(&["x", "y", "z"]).unwrap()[0].probs(), "huge char embedding");

    // An output bias that packs but overflows f32 in the scores: that
    // call's emissions come from the f64 reference, and tag A wins.
    let overflowing = model_with(0.0, 0.0, 1e39, 0.0);
    let packed = BilstmPosTagger::new(overflowing.clone());
    assert_eq!(packed.numerics(), Numerics::Packed);
    let parity = BilstmPosTagger::jvm_parity(overflowing);
    for tagger in [&packed, &parity] {
        let best = tagger.best_sequences(&[["x", "y"]]).unwrap().remove(0);
        assert_eq!(best.outcomes(), ["A", "A"]);
        assert_eq!(best.probs(), [1.0, 1.0]);
        assert_eq!(tagger.emissions(&["x"]).unwrap(), vec![vec![1e39, 0.0]]);
    }
}
