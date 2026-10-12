//! `language_detection::prepare`: the five normalizers OpenNLP's
//! LanguageDetectorFactory chains, in its order.

use std::borrow::Cow;

use turbo_normalize::emoji::EmojiRemove;
use turbo_normalize::language_detection::{chain, prepare};
use turbo_normalize::{Normalizer, Numbers, Shrink, SocialMedia, Urls, normalize};

/// The five normalizers applied one after another, each to the previous
/// one's output.
fn one_after_another(text: &str) -> String {
    let mut current = text.to_string();
    current = normalize(&EmojiRemove::supplementary(), &current).0;
    current = normalize(&Urls, &current).0;
    current = normalize(&SocialMedia, &current).0;
    current = normalize(&Numbers, &current).0;
    normalize(&Shrink, &current).0
}

#[test]
fn the_chain_is_in_the_factory_order() {
    let ids: Vec<&str> = chain().ids().collect();
    assert_eq!(
        ids,
        [
            <EmojiRemove as Normalizer>::ID,
            <Urls as Normalizer>::ID,
            <SocialMedia as Normalizer>::ID,
            <Numbers as Normalizer>::ID,
            <Shrink as Normalizer>::ID,
        ]
    );
}

#[test]
fn prepares_text_as_the_detector_does() {
    assert_eq!(prepare("Hi \u{1F600}\u{1F600} @ana see http://x.y 2024!!!!  ok"), "Hi see !! ok");
    assert!(matches!(prepare("plain words only"), Cow::Borrowed(_)));
    for text in [
        "",
        "  RT @user: check #cool :-) hahaha http://a.b/c?d=1 mail me@x.org 555-0199  ",
        "\u{1D400}\u{1D401} caf\u{E9} na\u{EF}ve 12.5% \u{20000}text",
        "Bahahaha!!! sooooo gooood \t\t xD :P",
        "\u{1F1E9}\u{1F1EA}http://x.y\u{1F600}a@b.cd\u{1F600}",
    ] {
        assert_eq!(prepare(text), one_after_another(text), "{text:?}");
    }
}
