//! The default numerics against the JVM: the packed f32 forward pass on
//! the CPU backend (`Numerics::Packed`) tags every golden sentence as the
//! JVM did, with emissions and probabilities within [`PACKED_TOLERANCE`],
//! and so does the same forward pass with its BiLSTMs on a GPU
//! ([`test_device`]), alone and as one batch.

use turbo::device::Device;
use turbo_analysis::Annotator;
use turbo_goldens::Golden;
use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, Numerics};

use crate::golden::*;

/// The largest difference allowed between an emission or a tag
/// probability of the packed f32 forward pass and the JVM's f64 value.
pub(crate) const PACKED_TOLERANCE: f64 = 1e-5;

/// The GPU the packed numerics are checked on as well: TURBO_TEST_DEVICE
/// (cuda, levelzero, metal, or any with :N), else cuda with
/// TURBO_TEST_REQUIRE_CUDA=1, levelzero with TURBO_TEST_REQUIRE_LEVELZERO=1
/// or metal with TURBO_TEST_REQUIRE_METAL=1. A device
/// the tagger cannot be made on fails the test; it is never skipped.
pub(crate) fn test_device() -> Option<Device> {
    let device = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(name) => Device::parse(&name).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={name}: {}", e.message)),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| v == "1") => Device::Cuda(0),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| v == "1") => Device::LevelZero(0),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_METAL").is_ok_and(|v| v == "1") => Device::Metal(0),
        Err(_) => return None,
    };
    (device != Device::Cpu).then_some(device)
}

/// Tags the golden's sentences with the default tagger, and on
/// [`test_device`] when one is named, and compares the tags (exactly), the
/// emissions and the probabilities with the JVM's.
pub(crate) fn check_packed(what: &str, g: &Golden, model: &BilstmPosModel, totals: &mut Totals) {
    let tagger = BilstmPosTagger::new(model.clone());
    assert_eq!((tagger.numerics(), tagger.device()), (Numerics::Packed, Device::Cpu), "{what}: default numerics");
    let (t, e, p) = (&mut totals.packed_tags, &mut totals.packed_emissions, &mut totals.packed_probabilities);
    compare(what, g, model, &tagger, t, e, p);
    if let Some(device) = test_device() {
        let on = BilstmPosTagger::on_device(model.clone(), device)
            .unwrap_or_else(|e| panic!("{what}: the tagger on {device:?}: {e}"));
        assert_eq!((on.numerics(), on.device()), (Numerics::Packed, device), "{what}: numerics on {device:?}");
        assert_ne!(on.config_hash(), tagger.config_hash(), "{what}: config_hash on {device:?}");
        let what = format!("{what} on {device:?}");
        let (t, e, p) = (&mut totals.device_tags, &mut totals.device_emissions, &mut totals.device_probabilities);
        compare(&what, g, model, &on, t, e, p);
        // FASTEST runs on CUDA alone; Level Zero refuses it.
        if let Device::Cuda(_) = device {
            fastest(&what, g, model, device, totals);
        }
    }
}

/// The tagger at FASTEST (F16 weights) on `device`: no bits are claimed,
/// so its tags are counted against the JVM's, and the test asks for at
/// least 99.9% of them over every golden (main.rs).
fn fastest(what: &str, g: &Golden, model: &BilstmPosModel, device: Device, totals: &mut Totals) {
    let fast = BilstmPosTagger::on_device_at(model.clone(), device, turbo::device::Precision::Fastest)
        .unwrap_or_else(|e| panic!("{what}: the tagger at FASTEST: {e}"));
    let sentences = sentences(g);
    let tagged = layer(g, "turbo:neural-tagged");
    let jvm_tags = strs(tagged, "tag");
    let batch = fast.best_sequences(&sentences).unwrap_or_else(|e| panic!("{what}: FASTEST batch: {e}"));
    let flat: Vec<&String> = batch.iter().flat_map(|s| s.outcomes()).collect();
    assert_eq!(flat.len(), jvm_tags.len(), "{what}: FASTEST tags");
    totals.fastest_tags += flat.len();
    totals.fastest_equal += flat.iter().zip(&jvm_tags).filter(|(a, b)| **a == *b).count();
}

fn compare(
    what: &str,
    g: &Golden,
    model: &BilstmPosModel,
    tagger: &BilstmPosTagger,
    tags: &mut usize,
    emitted: &mut Count,
    probabilities: &mut Count,
) {
    let sentences = sentences(g);
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, model.tags().len());
    let tagged = layer(g, "turbo:neural-tagged");
    let (jvm_tags, jvm_probs) = (strs(tagged, "tag"), f64s(tagged, "probability"));
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let rust = tagger.emissions(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        for (index, (r, j)) in rust.iter().zip(&emissions[k]).enumerate() {
            for (tag, (&r, &j)) in r.iter().zip(j).enumerate() {
                emitted.compare(r, j, PACKED_TOLERANCE, &|| {
                    format!("{what}: sentence {k} token {index} tag {tag}: packed emission")
                });
            }
        }
        let best = tagger.top_k_sequences(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        assert_eq!(tagger.tag(sentence).unwrap(), best[0].outcomes(), "{what}: sentence {k}: packed tag and best");
        for (index, (tag, &p)) in best[0].outcomes().iter().zip(best[0].probs()).enumerate() {
            assert_eq!(tag, &jvm_tags[row], "{what}: sentence {k} token {index}: packed tag");
            probabilities.compare(p, jvm_probs[row], PACKED_TOLERANCE, &|| {
                format!("{what}: sentence {k} token {index}: packed probability")
            });
            *tags += 1;
            row += 1;
        }
    }
    assert_eq!(row, jvm_tags.len(), "{what}: turbo:neural-tagged rows past the last token");
    // The whole golden as one batch tags as the sentences alone do.
    let batch = tagger.best_sequences(&sentences).unwrap_or_else(|e| panic!("{what}: batch: {e}"));
    let flat: Vec<&String> = batch.iter().flat_map(|s| s.outcomes()).collect();
    assert!(flat.len() == jvm_tags.len() && flat.iter().zip(&jvm_tags).all(|(a, b)| *a == b), "{what}: batched tags");
}
