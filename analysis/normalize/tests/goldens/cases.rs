use super::*;

#[test]
fn whitespace_collapse() {
    check("whitespace", Edits::Reported, &Whitespace::collapse(), &[]);
}

#[test]
fn whitespace_keep_line_breaks() {
    check("whitespace-keep-lines", Edits::Reported, &Whitespace::keep_line_breaks(), &[]);
}

#[test]
fn whitespace_keep_paragraphs() {
    check("whitespace-keep-paragraphs", Edits::Reported, &Whitespace::keep_paragraphs(), &[]);
}

#[test]
fn dashes() {
    check("dash", Edits::Reported, &Dashes::new(), &[]);
}

#[test]
fn quotes() {
    check("quote", Edits::Reported, &Quotes, QUOTE);
}

#[test]
fn bullets() {
    check("bullet", Edits::Reported, &Bullets, BULLET);
}

#[test]
fn ellipsis() {
    check("ellipsis", Edits::Reported, &Ellipsis, ELLIPSIS);
}

#[test]
fn digits() {
    check("digit", Edits::Reported, &Digits, &[]);
}

#[test]
fn invisible() {
    check("invisible", Edits::Reported, &Invisible, INVISIBLE);
}

#[test]
fn dehyphenation() {
    check("dehyphenation", Edits::Reported, &Dehyphenate::new(), DEHYPHENATION);
}

#[test]
fn nfc() {
    check("nfc", Edits::Reconstructed, &UnicodeForm::nfc(), &[]);
}

#[test]
fn nfkc() {
    check("nfkc", Edits::Reconstructed, &UnicodeForm::nfkc(), &[]);
}

#[test]
fn case_fold_lowercase() {
    check("case-fold", Edits::Reconstructed, &CaseFold::lowercase(), &[]);
}

#[test]
fn case_fold_full() {
    check("full-case-fold", Edits::Reported, &CaseFold::full(), &[]);
}

#[test]
fn accent_fold() {
    check("accent-fold", Edits::Reconstructed, &AccentFold::new(), &[]);
}

#[test]
fn german_umlauts() {
    check("german-umlaut", Edits::Reported, &GermanUmlauts::new(), &[]);
}

#[test]
fn confusable_skeleton() {
    check("confusable-skeleton", Edits::Reconstructed, &ConfusableSkeleton, CONFUSABLE_SKELETON);
}

#[test]
fn urls() {
    check("url", Edits::Reconstructed, &Urls, &[]);
}

#[test]
fn bounded_urls() {
    check("bounded-url", Edits::Reconstructed, &BoundedUrls, BOUNDED_URL);
}

#[test]
fn social_media() {
    check("social-media", Edits::Reconstructed, &SocialMedia, &[]);
}

#[test]
fn numbers() {
    check("number", Edits::Reconstructed, &Numbers, &[]);
}

#[test]
fn shrink() {
    check("shrink", Edits::Reconstructed, &Shrink, &[]);
}

#[test]
fn english_contractions() {
    check("english-contraction", Edits::Reported, &EnglishContractions, ENGLISH_CONTRACTION);
}

#[test]
fn emoji() {
    check("emoji", Edits::Reconstructed, &EmojiRemove::supplementary(), EMOJI);
}

#[test]
fn emoji_to_emoticon() {
    check("emoji-to-emoticon", Edits::Reported, &EmojiToEmoticon, &[]);
}

#[test]
fn emoticon_to_emoji() {
    check("emoticon-to-emoji", Edits::Reported, &EmoticonToEmoji, &[]);
}

#[test]
fn symbol_words() {
    check("symbol-joiner", Edits::Reconstructed, &SymbolWords, SYMBOL_JOINER);
}

#[test]
fn default_chain() {
    let chain = Chain::new()
        .then(Invisible)
        .then(UnicodeForm::nfc())
        .then(Whitespace::collapse())
        .then(Quotes)
        .then(Dashes::new())
        .then(CaseFold::lowercase())
        .then(AccentFold::new());
    check("default-chain", Edits::Reconstructed, &chain, DEFAULT_CHAIN);
}

#[test]
fn case_fold_turkish() {
    check("case-fold-tr", Edits::Reconstructed, &CaseFold::turkish_lowercase(), &[]);
}

#[test]
fn case_fold_azerbaijani() {
    check("case-fold-tr", Edits::Reconstructed, &CaseFold::azerbaijani_lowercase(), &[]);
}

#[test]
fn case_fold_turkic_remains_context_free() {
    check("case-fold-tr", Edits::Reconstructed, &CaseFold::turkic_simple(), CASE_FOLD_TR);
}

// The texts on which a normalizer is documented to differ from OpenNLP, with the output it is
// documented to give. Every difference other than the Unicode version of the data is an addition
// that docs/normalizers.md lists under "Differences from OpenNLP"; the Unicode data differences are
// the 18.0 data against the JVM's.
