//! The two numerics of `BilstmPosTagger` through its public API: CRF
//! weights too large for plain f64 still give the exact scorer's
//! probabilities, never a non-finite one, and each numerics has its own
//! stable configuration hash.

use turbo_analysis::Annotator;
use turbo_postag::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::lstm::LstmLayer;
use turbo_postag::neural::{BilstmPosTagger, Numerics};

fn lstm(input: usize, hidden: usize) -> LstmLayer {
    let gates = 4 * hidden;
    LstmLayer::of_weights(
        input,
        hidden,
        vec![vec![0.0; input]; gates],
        vec![vec![0.0; hidden]; gates],
        vec![0.0; gates],
    )
    .unwrap()
}

/// Two tags, every weight zero but a uniform CRF transition matrix of
/// `transition`: every emission is zero, so each token's marginal is 1/2.
fn model(transition: f64) -> BilstmPosModel {
    BilstmPosModel::new(
        vec![(UNKNOWN.to_owned(), 0)],
        char_vocabulary_from_strs([(UNKNOWN, 0)]),
        vec!["A".into(), "B".into()],
        vec![vec![0.0; 2]],
        vec![vec![0.0; 3]],
        lstm(3, 2),
        lstm(3, 2),
        lstm(6, 2),
        lstm(6, 2),
        None,
        None,
        vec![vec![0.0; 4]; 2],
        vec![0.0; 2],
        5,
        None,
        None,
        Some(vec![vec![transition; 2]; 2]),
        Some(vec![0.0; 2]),
        Some(vec![0.0; 2]),
        None,
        None,
    )
    .unwrap()
}

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
    let fnv = |tag: &[u8]| {
        [tag, b"\0", sha.as_bytes()]
            .into_iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    };
    assert_eq!((packed.config_hash(), parity.config_hash()), (fnv(b"packed"), fnv(b"jvm-parity")));
}
