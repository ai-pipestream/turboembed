//! The seed lexicon and emoji table of [`EmotionAnnotator`](super::EmotionAnnotator):
//! a few common cue words and emoji per GoEmotions label, written for this
//! crate, each with a weight in `0..=1`. They seed the lexicon stage and
//! give the learned stage's positives words to point at; they are not a
//! trained or exhaustive resource.

use turbo_analysis_layers::EmotionDimension as D;

/// `(word, label, weight)`, case-folded single tokens, in label order.
pub(super) const WORDS: &[(&str, D, f64)] = &[
    ("admire", D::Admiration, 0.8),
    ("impressive", D::Admiration, 0.75),
    ("brilliant", D::Admiration, 0.7),
    ("amazing", D::Admiration, 0.6),
    ("hilarious", D::Amusement, 0.85),
    ("funny", D::Amusement, 0.75),
    ("lol", D::Amusement, 0.8),
    ("haha", D::Amusement, 0.8),
    ("furious", D::Anger, 0.9),
    ("angry", D::Anger, 0.85),
    ("outraged", D::Anger, 0.85),
    ("annoying", D::Annoyance, 0.8),
    ("annoyed", D::Annoyance, 0.8),
    ("irritating", D::Annoyance, 0.75),
    ("agree", D::Approval, 0.7),
    ("approve", D::Approval, 0.75),
    ("exactly", D::Approval, 0.55),
    ("care", D::Caring, 0.6),
    ("careful", D::Caring, 0.55),
    ("confused", D::Confusion, 0.85),
    ("confusing", D::Confusion, 0.8),
    ("unclear", D::Confusion, 0.6),
    ("curious", D::Curiosity, 0.8),
    ("wonder", D::Curiosity, 0.65),
    ("wish", D::Desire, 0.7),
    ("want", D::Desire, 0.5),
    ("disappointed", D::Disappointment, 0.85),
    ("disappointing", D::Disappointment, 0.8),
    ("disagree", D::Disapproval, 0.75),
    ("wrong", D::Disapproval, 0.45),
    ("disgusting", D::Disgust, 0.9),
    ("gross", D::Disgust, 0.8),
    ("embarrassed", D::Embarrassment, 0.85),
    ("embarrassing", D::Embarrassment, 0.8),
    ("excited", D::Excitement, 0.85),
    ("thrilled", D::Excitement, 0.85),
    ("afraid", D::Fear, 0.85),
    ("scared", D::Fear, 0.85),
    ("terrified", D::Fear, 0.9),
    ("thanks", D::Gratitude, 0.85),
    ("thank", D::Gratitude, 0.85),
    ("grateful", D::Gratitude, 0.85),
    ("grieving", D::Grief, 0.9),
    ("mourn", D::Grief, 0.85),
    ("happy", D::Joy, 0.8),
    ("glad", D::Joy, 0.75),
    ("delighted", D::Joy, 0.85),
    ("love", D::Love, 0.8),
    ("adore", D::Love, 0.85),
    ("nervous", D::Nervousness, 0.85),
    ("anxious", D::Nervousness, 0.85),
    ("hope", D::Optimism, 0.6),
    ("hopeful", D::Optimism, 0.8),
    ("optimistic", D::Optimism, 0.85),
    ("proud", D::Pride, 0.85),
    ("realized", D::Realization, 0.75),
    ("realize", D::Realization, 0.7),
    ("relieved", D::Relief, 0.85),
    ("relief", D::Relief, 0.8),
    ("sorry", D::Remorse, 0.75),
    ("regret", D::Remorse, 0.8),
    ("sad", D::Sadness, 0.85),
    ("unhappy", D::Sadness, 0.8),
    ("heartbroken", D::Sadness, 0.85),
    ("surprised", D::Surprise, 0.85),
    ("wow", D::Surprise, 0.7),
    ("unexpected", D::Surprise, 0.6),
];

/// `(emoji, label, weight)`, matched as a whole token.
pub(super) const EMOJI: &[(&str, D, f64)] = &[
    ("\u{1F602}", D::Amusement, 0.85),
    ("\u{1F923}", D::Amusement, 0.85),
    ("\u{1F620}", D::Anger, 0.85),
    ("\u{1F621}", D::Anger, 0.9),
    ("\u{1F914}", D::Curiosity, 0.6),
    ("\u{1F922}", D::Disgust, 0.85),
    ("\u{1F633}", D::Embarrassment, 0.7),
    ("\u{1F631}", D::Fear, 0.8),
    ("\u{1F64F}", D::Gratitude, 0.7),
    ("\u{1F600}", D::Joy, 0.75),
    ("\u{1F60A}", D::Joy, 0.75),
    ("\u{2764}\u{FE0F}", D::Love, 0.8),
    ("\u{2764}", D::Love, 0.8),
    ("\u{1F60D}", D::Love, 0.85),
    ("\u{1F605}", D::Nervousness, 0.6),
    ("\u{1F625}", D::Relief, 0.6),
    ("\u{1F622}", D::Sadness, 0.8),
    ("\u{1F62D}", D::Sadness, 0.85),
    ("\u{1F62E}", D::Surprise, 0.75),
    ("\u{1F632}", D::Surprise, 0.8),
];

/// Words that negate a cue among the three tokens before it.
pub(super) const NEGATORS: &[&str] =
    &["not", "no", "never", "n't", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "hardly", "without"];

/// How many tokens before a cue a negator reaches.
pub(super) const NEGATION_WINDOW: usize = 3;

/// The weight of `token` (already lowercased and trimmed of punctuation,
/// or an emoji) for each label it cues.
pub(super) fn cues(token: &str) -> impl Iterator<Item = (D, f64)> + '_ {
    WORDS.iter().chain(EMOJI).filter(move |(w, _, _)| *w == token).map(|&(_, d, weight)| (d, weight))
}
