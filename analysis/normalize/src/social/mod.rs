//! Social media markup: hashtags, mentions, retweet markers, emoticons and
//! laughter.
//!
//! - [`SocialMedia`] keeps the output of OpenNLP's
//!   `SocialMediaCharSequenceNormalizer`, which the existing language
//!   detector models were trained with. Its tags are ASCII-whitespace
//!   delimited and its emoticons match inside words.
//! - [`BoundedSocialMedia`] finds the same kinds of markup by Unicode
//!   character properties and word boundaries, the way the platforms that
//!   define hashtags and mentions delimit them.

mod bounded;
mod opennlp;
mod shared;

pub use bounded::BoundedSocialMedia;
pub use opennlp::SocialMedia;
