//! The text preparation OpenNLP's language detector applies before it
//! counts character n-grams.

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::emoji::EmojiRemove;
use crate::{Chain, Numbers, Shrink, SocialMedia, Urls, apply};

/// The chain `LanguageDetectorFactory.getContextGenerator()` builds, in
/// its order: supplementary-plane characters (the deprecated
/// `EmojiCharSequenceNormalizer`), URLs and e-mail addresses, social media
/// markup, ASCII digit runs, and repeated characters and whitespace.
///
/// The models OpenNLP distributes were trained on text prepared this way,
/// so each step keeps OpenNLP's output text exactly; the improved
/// variants (`BoundedUrls`, `EmojiRemove::sequences`) are not used here.
pub fn chain() -> Chain {
    Chain::new().then(EmojiRemove::supplementary()).then(Urls).then(SocialMedia).then(Numbers).then(Shrink)
}

/// `text` prepared as OpenNLP's language detector prepares it, borrowed
/// when nothing changed.
pub fn prepare(text: &str) -> Cow<'_, str> {
    static CHAIN: OnceLock<Chain> = OnceLock::new();
    apply(CHAIN.get_or_init(chain), text)
}
