//! Optional model scores on mentions (#1903): a candidate with no score
//! is `None`, not zero; a score names its model in the detector table;
//! and `PiiDetector::with_score_threshold` rejects scored candidates below
//! the threshold, leaving unscored ones alone and recording the rejection
//! in `turbo:pii-candidates`.

mod support;

use support::*;
use turbo_analysis::{Annotator, Layer, Span, ViewId};
use turbo_pii::{PiiDetector, PiiEvidence, PiiKind, PiiScore, PiiScoreKind, PiiSink, Scanner};

const BADGE_KIND: PiiKind = test_kind(3);
const SHAPE: &[PiiEvidence] = &[PiiEvidence::Shape];

#[test]
fn scores_are_optional_and_the_threshold_rejects_only_scored_candidates() {
    use turbo_analysis::Ix;
    use turbo_pii::{PiiDisposition, RejectReason};
    fn scored(text: &str, sink: &mut PiiSink<'_>) {
        let model = sink.model("badge-head", "2026.1").unwrap();
        let mut at = 0;
        for (i, word) in text.split(' ').enumerate() {
            let span = Span::new(at as u32, (at + word.len()) as u32);
            let score = (i > 0).then_some(PiiScore {
                value: if i == 1 { 0.2 } else { 0.9 },
                kind: PiiScoreKind::Probability,
                model,
                calibrated: i == 2,
            });
            sink.try_add_candidate(span, BADGE_KIND, SHAPE, score, |s| s.push_str(word)).unwrap();
            at += word.len() + 1;
        }
    }
    let detector =
        PiiDetector::new(vec![Scanner::custom_fn("Scored", BADGE_KIND, scored)]).with_custom_kinds(test_kinds());
    let text = "unscored low high";
    let pii = detector.layer(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    let scores: Vec<_> = pii.mentions().map(|m| m.score.map(|s| (s.value, s.calibrated))).collect();
    assert_eq!(scores, [None, Some((0.2, false)), Some((0.9, true))]);
    assert_eq!(pii.detectors.get(pii.score[1].unwrap().model), Some(("badge-head", "2026.1")));
    assert_eq!(pii.detectors.get(pii.detector[1]), Some(("Scored", "0")));

    let gated = detector.with_score_threshold(0.5);
    let (pii, candidates) = gated.layer_with_candidates(text, ViewId::SOURCE);
    assert_eq!(pii.mentions().map(|m| m.normalized).collect::<Vec<_>>(), ["unscored", "high"]);
    assert_eq!(
        candidates.disposition,
        [
            PiiDisposition::Kept { mention: Ix::new(0) },
            PiiDisposition::RejectedByRule { reason: RejectReason::Threshold },
            PiiDisposition::Kept { mention: Ix::new(1) },
        ]
    );
    assert_ne!(gated.config_hash(), PiiDetector::new(vec![]).with_custom_kinds(test_kinds()).config_hash());
}
