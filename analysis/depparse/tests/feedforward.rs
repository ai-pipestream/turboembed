use turbo_analysis_layers::Tokens;
use turbo_depparse::{ArcStandardState, FeatureLayout, FeedforwardContext};

#[test]
fn expanded_configuration_has_18_word_18_pos_and_12_label_slots() {
    let state = ArcStandardState::<Tokens>::new(1).unwrap();
    let f = FeedforwardContext::extract(&state, &["word"], &["N"], FeatureLayout::ChenManning18).unwrap();
    assert_eq!((f.words.len(), f.pos.len(), f.labels.len()), (18, 18, 12));
    assert_eq!(f.words[0].as_deref(), Some("*ROOT*"));
    assert_eq!(f.words[3].as_deref(), Some("word"));
}
#[path = "feedforward/boundaries.rs"]
mod boundaries;
#[path = "feedforward/parity.rs"]
mod parity;
#[path = "feedforward/pipeline.rs"]
mod pipeline;
#[path = "feedforward/support.rs"]
mod support;
