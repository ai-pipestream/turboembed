//! The 28 labels of `turbo:emotion`: GoEmotions' 27 emotions and neutral
//! (Demszky et al., ACL 2020), in the dataset's own label order.

use std::fmt;

/// One label of [`EmotionLayer`](super::EmotionLayer), numbered as
/// GoEmotions numbers its labels.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum EmotionDimension {
    Admiration = 0,
    Amusement = 1,
    Anger = 2,
    Annoyance = 3,
    Approval = 4,
    Caring = 5,
    Confusion = 6,
    Curiosity = 7,
    Desire = 8,
    Disappointment = 9,
    Disapproval = 10,
    Disgust = 11,
    Embarrassment = 12,
    Excitement = 13,
    Fear = 14,
    Gratitude = 15,
    Grief = 16,
    Joy = 17,
    Love = 18,
    Nervousness = 19,
    Optimism = 20,
    Pride = 21,
    Realization = 22,
    Relief = 23,
    Remorse = 24,
    Sadness = 25,
    Surprise = 26,
    Neutral = 27,
}

/// The number of [`EmotionDimension`]s.
pub const EMOTION_DIMENSIONS: usize = 28;

impl EmotionDimension {
    pub const ALL: [EmotionDimension; EMOTION_DIMENSIONS] = [
        EmotionDimension::Admiration,
        EmotionDimension::Amusement,
        EmotionDimension::Anger,
        EmotionDimension::Annoyance,
        EmotionDimension::Approval,
        EmotionDimension::Caring,
        EmotionDimension::Confusion,
        EmotionDimension::Curiosity,
        EmotionDimension::Desire,
        EmotionDimension::Disappointment,
        EmotionDimension::Disapproval,
        EmotionDimension::Disgust,
        EmotionDimension::Embarrassment,
        EmotionDimension::Excitement,
        EmotionDimension::Fear,
        EmotionDimension::Gratitude,
        EmotionDimension::Grief,
        EmotionDimension::Joy,
        EmotionDimension::Love,
        EmotionDimension::Nervousness,
        EmotionDimension::Optimism,
        EmotionDimension::Pride,
        EmotionDimension::Realization,
        EmotionDimension::Relief,
        EmotionDimension::Remorse,
        EmotionDimension::Sadness,
        EmotionDimension::Surprise,
        EmotionDimension::Neutral,
    ];

    /// The label's position, `0..28`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// GoEmotions' name of the label.
    pub const fn name(self) -> &'static str {
        match self {
            EmotionDimension::Admiration => "admiration",
            EmotionDimension::Amusement => "amusement",
            EmotionDimension::Anger => "anger",
            EmotionDimension::Annoyance => "annoyance",
            EmotionDimension::Approval => "approval",
            EmotionDimension::Caring => "caring",
            EmotionDimension::Confusion => "confusion",
            EmotionDimension::Curiosity => "curiosity",
            EmotionDimension::Desire => "desire",
            EmotionDimension::Disappointment => "disappointment",
            EmotionDimension::Disapproval => "disapproval",
            EmotionDimension::Disgust => "disgust",
            EmotionDimension::Embarrassment => "embarrassment",
            EmotionDimension::Excitement => "excitement",
            EmotionDimension::Fear => "fear",
            EmotionDimension::Gratitude => "gratitude",
            EmotionDimension::Grief => "grief",
            EmotionDimension::Joy => "joy",
            EmotionDimension::Love => "love",
            EmotionDimension::Nervousness => "nervousness",
            EmotionDimension::Optimism => "optimism",
            EmotionDimension::Pride => "pride",
            EmotionDimension::Realization => "realization",
            EmotionDimension::Relief => "relief",
            EmotionDimension::Remorse => "remorse",
            EmotionDimension::Sadness => "sadness",
            EmotionDimension::Surprise => "surprise",
            EmotionDimension::Neutral => "neutral",
        }
    }

    pub fn from_name(name: &str) -> Option<EmotionDimension> {
        EmotionDimension::ALL.into_iter().find(|d| d.name() == name)
    }

    pub fn from_index(i: usize) -> Option<EmotionDimension> {
        EmotionDimension::ALL.get(i).copied()
    }
}

impl fmt::Display for EmotionDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The 28 scores of one unit, `0.0..=1.0` each, indexed by
/// [`EmotionDimension::index`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EmotionScores(pub [f64; EMOTION_DIMENSIONS]);

impl Default for EmotionScores {
    fn default() -> Self {
        EmotionScores([0.0; EMOTION_DIMENSIONS])
    }
}

impl EmotionScores {
    pub fn get(&self, d: EmotionDimension) -> f64 {
        self.0[d.index()]
    }

    pub fn set(&mut self, d: EmotionDimension, v: f64) {
        self.0[d.index()] = v;
    }

    /// Every score finite and in `0.0..=1.0`.
    pub fn is_valid(&self) -> bool {
        self.0.iter().all(|s| s.is_finite() && (0.0..=1.0).contains(s))
    }

    /// The labels at or above `threshold`, in label order.
    pub fn active(&self, threshold: f64) -> impl Iterator<Item = EmotionDimension> + '_ {
        EmotionDimension::ALL.into_iter().filter(move |d| self.get(*d) >= threshold)
    }
}
