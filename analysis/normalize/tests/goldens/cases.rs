use super::*;

#[test]
fn whitespace_collapse() {
    check("whitespace", &Whitespace::collapse(), &[]);
}

#[test]
fn whitespace_keep_line_breaks() {
    check("whitespace-keep-lines", &Whitespace::keep_line_breaks(), &[]);
}

#[test]
fn whitespace_keep_paragraphs() {
    check("whitespace-keep-paragraphs", &Whitespace::keep_paragraphs(), &[]);
}

#[test]
fn dashes() {
    check("dash", &Dashes::new(), &[]);
}

#[test]
fn quotes() {
    check("quote", &Quotes, QUOTE);
}

#[test]
fn bullets() {
    check("bullet", &Bullets, BULLET);
}

#[test]
fn ellipsis() {
    check("ellipsis", &Ellipsis, ELLIPSIS);
}

#[test]
fn digits() {
    check("digit", &Digits, &[]);
}

#[test]
fn invisible() {
    check("invisible", &Invisible, INVISIBLE);
}

#[test]
fn dehyphenation() {
    check("dehyphenation", &Dehyphenate::new(), DEHYPHENATION);
}

#[test]
fn nfc() {
    check("nfc", &UnicodeForm::nfc(), &[]);
}

#[test]
fn nfkc() {
    check("nfkc", &UnicodeForm::nfkc(), &[]);
}

#[test]
fn case_fold_lowercase() {
    check("case-fold", &CaseFold::lowercase(), &[]);
}

#[test]
fn case_fold_full() {
    check("full-case-fold", &CaseFold::full(), &[]);
}

#[test]
fn accent_fold() {
    check("accent-fold", &AccentFold::new(), &[]);
}

#[test]
fn german_umlauts() {
    check("german-umlaut", &GermanUmlauts::new(), &[]);
}

#[test]
fn confusable_skeleton() {
    check("confusable-skeleton", &ConfusableSkeleton, CONFUSABLE_SKELETON);
}

#[test]
fn urls() {
    check("url", &Urls, &[]);
}

#[test]
fn bounded_urls() {
    check("bounded-url", &BoundedUrls, BOUNDED_URL);
}

#[test]
fn social_media() {
    check("social-media", &SocialMedia, &[]);
}

#[test]
fn numbers() {
    check("number", &Numbers, &[]);
}

#[test]
fn shrink() {
    check("shrink", &Shrink, &[]);
}

#[test]
fn english_contractions() {
    check("english-contraction", &EnglishContractions, ENGLISH_CONTRACTION);
}

#[test]
fn emoji() {
    check("emoji", &EmojiRemove::supplementary(), EMOJI);
}

#[test]
fn emoji_to_emoticon() {
    check("emoji-to-emoticon", &EmojiToEmoticon, &[]);
}

#[test]
fn emoticon_to_emoji() {
    check("emoticon-to-emoji", &EmoticonToEmoji, &[]);
}

#[test]
fn symbol_words() {
    check("symbol-joiner", &SymbolWords, SYMBOL_JOINER);
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
    check("default-chain", &chain, DEFAULT_CHAIN);
}

#[test]
fn case_fold_turkish() {
    check("case-fold-tr", &CaseFold::turkish_lowercase(), &[]);
}

#[test]
fn case_fold_azerbaijani() {
    check("case-fold-tr", &CaseFold::azerbaijani_lowercase(), &[]);
}

#[test]
fn case_fold_turkic_remains_context_free() {
    check("case-fold-turkic", &CaseFold::turkic_simple(), CASE_FOLD_TR);
}

// The texts on which a normalizer is documented to differ from OpenNLP, with the output it is
// documented to give. Every difference other than the Unicode version of the data is an addition
// that docs/normalizers.md lists under "Differences from OpenNLP"; the Unicode data differences are
// the 18.0 data against the JVM's.
