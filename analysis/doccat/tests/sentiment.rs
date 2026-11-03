//! Context extraction and factory naming for the classical sentiment factories.

#![cfg(feature = "classical")]

use turbo_doccat::sentiment::{EMOJI_FACTORY, SentimentFactory, TOKEN_FACTORY};

#[test]
fn the_token_context_is_the_tokens_as_they_are() {
    let mut extra = Vec::new();
    let tokens = ["good", "\u{1F600}", "caf\u{E9}"];
    assert_eq!(SentimentFactory::Tokens.context(&tokens, &mut extra), tokens);
    assert!(extra.is_empty());
}

#[test]
fn the_emoji_context_appends_each_token_s_features_in_turn() {
    let mut extra = Vec::new();
    let tokens = ["\u{1F622}", "good\u{1F600}", "\u{1F1EF}\u{1F1F5}", "\u{2764}\u{FE0F}", "\u{1F600}\u{1F600}"];
    assert_eq!(
        SentimentFactory::Emoji.context(&tokens, &mut extra),
        [
            "\u{1F622}",
            "good\u{1F600}",
            "\u{1F1EF}\u{1F1F5}",
            "\u{2764}\u{FE0F}",
            "\u{1F600}\u{1F600}",
            "emojiSentiment=-2",
            "emojiType=FACE",
            "emojiCategory=SMILEYS_AND_EMOTION",
            "emojiType=FLAG",
            "emojiCategory=FLAGS",
            "emojiRegion=JP",
            "emojiSentiment=2",
            "emojiType=HEART",
            "emojiCategory=SMILEYS_AND_EMOTION",
        ]
    );
    // A token that starts with ASCII, or holds two emoji, has none.
    assert_eq!(SentimentFactory::Emoji.context(&["good\u{1F600}"], &mut extra), ["good\u{1F600}"]);
}

#[test]
fn the_factory_names_its_class() {
    assert_eq!(SentimentFactory::from_class(TOKEN_FACTORY), Some(SentimentFactory::Tokens));
    assert_eq!(SentimentFactory::from_class(EMOJI_FACTORY), Some(SentimentFactory::Emoji));
    assert_eq!(SentimentFactory::from_class("opennlp.tools.doccat.DoccatFactory"), None);
    assert_eq!(SentimentFactory::Emoji.class(), EMOJI_FACTORY);
}
