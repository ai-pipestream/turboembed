//! The default numerics against the JVM: the packed f32 forward pass on
//! the CPU backend (`Numerics::Packed`) tags every golden sentence as the
//! JVM did, with emissions and probabilities within [`PACKED_TOLERANCE`].

use turbo_goldens::Golden;
use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, Numerics};

use crate::golden::*;

/// The largest difference allowed between an emission or a tag
/// probability of the packed f32 forward pass and the JVM's f64 value.
pub(crate) const PACKED_TOLERANCE: f64 = 1e-5;

/// Tags the golden's sentences with the default tagger and compares the
/// tags (exactly), the emissions and the probabilities with the JVM's.
pub(crate) fn check_packed(what: &str, g: &Golden, model: &BilstmPosModel, totals: &mut Totals) {
    let tagger = BilstmPosTagger::new(model.clone());
    assert_eq!(tagger.numerics(), Numerics::Packed, "{what}: default numerics");
    let sentences = sentences(g);
    let emissions = table(g, "turbo:neural-emissions", "score", &sentences, model.tags().len());
    let tagged = layer(g, "turbo:neural-tagged");
    let (jvm_tags, jvm_probs) = (strs(tagged, "tag"), f64s(tagged, "probability"));
    let mut row = 0;
    for (k, sentence) in sentences.iter().enumerate() {
        let rust = tagger.emissions(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        for (index, (r, j)) in rust.iter().zip(&emissions[k]).enumerate() {
            for (tag, (&r, &j)) in r.iter().zip(j).enumerate() {
                totals.packed_emissions.compare(r, j, PACKED_TOLERANCE, &|| {
                    format!("{what}: sentence {k} token {index} tag {tag}: packed emission")
                });
            }
        }
        let best = tagger.top_k_sequences(sentence).unwrap_or_else(|e| panic!("{what}: sentence {k}: {e}"));
        assert_eq!(tagger.tag(sentence).unwrap(), best[0].outcomes(), "{what}: sentence {k}: packed tag and best");
        for (index, (tag, &p)) in best[0].outcomes().iter().zip(best[0].probs()).enumerate() {
            assert_eq!(tag, &jvm_tags[row], "{what}: sentence {k} token {index}: packed tag");
            totals.packed_probabilities.compare(p, jvm_probs[row], PACKED_TOLERANCE, &|| {
                format!("{what}: sentence {k} token {index}: packed probability")
            });
            totals.packed_tags += 1;
            row += 1;
        }
    }
    assert_eq!(row, jvm_tags.len(), "{what}: turbo:neural-tagged rows past the last token");
}
