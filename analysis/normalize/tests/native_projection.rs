//! Native boundary contracts for normalizers whose output adds, deletes or
//! changes the byte length of characters: every scalar of the normalized
//! view sits on a character boundary, its UTF-16 and code point offsets
//! agree with the text, and spans of the normalized view project back to
//! the source bytes that produced them.

use turbo_analysis::{Document, Key, Provenance, Span, View};
use turbo_goldens::invariants::{check_document, check_spans};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};

const NORM: Key<View> = Key::named("norm");

fn run<N: Normalizer>(normalizer: &N, text: &str) -> Document {
    let mut doc = Document::new(text);
    let (out, alignment) = normalize(normalizer, text);
    let view = View::derived(doc.get::<View>().expect("the source view"), out, alignment);
    doc.add(&NORM, view, Provenance::by("native-projection")).unwrap();
    doc
}

fn normalized(doc: &Document) -> &View {
    doc.get_key::<View>(&NORM).unwrap()
}

fn scalar_contract(doc: &Document, view: &View) {
    let starts: Vec<_> = view.text().char_indices().map(|(offset, _)| offset as u32).collect();
    let ends: Vec<_> = starts.iter().skip(1).copied().chain([view.text().len() as u32]).take(starts.len()).collect();
    let violations = check_spans(doc, "native scalar boundaries", view.id(), &starts, &ends);
    assert!(violations.is_empty(), "{violations:?}");
    let offsets = doc.offsets(view.id());
    for byte in starts.into_iter().chain([view.text().len() as u32]) {
        let prefix = &view.text()[..byte as usize];
        assert_eq!(offsets.to_utf16(view.text(), byte).0 as usize, prefix.encode_utf16().count());
        assert_eq!(offsets.to_code_point(view.text(), byte).0 as usize, prefix.chars().count());
    }
}

#[test]
fn turkish_lowercase_has_exact_native_text_and_source_boundaries() {
    let turkish = CaseFold::turkish_lowercase();
    for (input, expected) in [
        ("Iİiı", "ıiiı"),
        ("I\u{307}", "i"),
        ("İSTANBUL IĞDIR", "istanbul ığdır"),
        ("😀 I İ café 東京", "😀 ı i café 東京"),
        ("", ""),
    ] {
        let doc = run(&turkish, input);
        let view = normalized(&doc);
        assert_eq!(view.text(), expected, "{input:?}");
        scalar_contract(&doc, view);
        // The trailing combining dot is deleted and contributes no output bytes.
        let contributing_end = if input == "I\u{307}" { 1 } else { input.len() as u32 };
        assert_eq!(doc.to_source(view.id(), Span::new(0, expected.len() as u32)), Span::new(0, contributing_end));
    }
    let doc = run(&turkish, "Iİiı");
    let view = normalized(&doc);
    for (child, source) in [((0, 2), (0, 1)), ((2, 3), (1, 3)), ((3, 4), (3, 4)), ((4, 6), (4, 6))] {
        assert_eq!(doc.to_source(view.id(), Span::new(child.0, child.1)), Span::new(source.0, source.1));
    }
}

#[test]
fn native_expansion_and_deletion_keep_source_projection() {
    let cases: [(&str, Document, &str); 4] = [
        ("ellipsis", run(&Ellipsis, "…\u{fe19}"), "......"),
        ("invisible", run(&Invisible, "a\u{200b}b"), "ab"),
        ("full-case-fold", run(&CaseFold::full(), "Straße"), "strasse"),
        ("whitespace", run(&Whitespace::collapse(), "😀  beta"), "😀 beta"),
    ];
    for (name, doc, expected) in &cases {
        let view = normalized(doc);
        assert_eq!(view.text(), *expected, "{name}");
        scalar_contract(doc, view);
        let input_len = doc.get::<View>().unwrap().text().len() as u32;
        assert_eq!(doc.to_source(view.id(), Span::new(0, expected.len() as u32)), Span::new(0, input_len), "{name}");
    }
    let doc = run(&Whitespace::collapse(), "😀  beta");
    assert_eq!(doc.to_source(normalized(&doc).id(), Span::new(5, 9)), Span::new(6, 10));
}

/// Every shipped normalizer and the two chains the analysis pipeline uses,
/// each run on mixed emoji, combining-mark, CJK and contact text: the
/// normalized view keeps every document invariant and every scalar of it
/// starts and ends on a character boundary that projects to the source.
#[test]
fn every_normalizer_keeps_scalar_boundaries_on_unicode_text() {
    let text = "😀 Zoë  e\u{301} 東京.\n\nEmail a@example.org costs 12 EUR.";
    let docs: Vec<(&str, Document)> = vec![
        ("whitespace", run(&Whitespace::collapse(), text)),
        ("whitespace-keep-lines", run(&Whitespace::keep_line_breaks(), text)),
        ("whitespace-keep-paragraphs", run(&Whitespace::keep_paragraphs(), text)),
        ("dash", run(&Dashes::new(), text)),
        ("quote", run(&Quotes, text)),
        ("bullet", run(&Bullets, text)),
        ("ellipsis", run(&Ellipsis, text)),
        ("digit", run(&Digits, text)),
        ("invisible", run(&Invisible, text)),
        ("dehyphenation", run(&Dehyphenate::new(), text)),
        ("nfc", run(&UnicodeForm::nfc(), text)),
        ("nfkc", run(&UnicodeForm::nfkc(), text)),
        ("case-fold", run(&CaseFold::lowercase(), text)),
        ("case-fold-tr", run(&CaseFold::turkish_lowercase(), text)),
        ("full-case-fold", run(&CaseFold::full(), text)),
        ("accent-fold", run(&AccentFold::new(), text)),
        ("german-umlaut", run(&GermanUmlauts::new(), text)),
        ("confusable-skeleton", run(&ConfusableSkeleton, text)),
        ("url", run(&Urls, text)),
        ("bounded-url", run(&BoundedUrls, text)),
        ("social-media", run(&SocialMedia, text)),
        ("number", run(&Numbers, text)),
        ("shrink", run(&Shrink, text)),
        ("english-contraction", run(&EnglishContractions, text)),
        ("emoji", run(&EmojiRemove::supplementary(), text)),
        ("emoji-to-emoticon", run(&EmojiToEmoticon, text)),
        ("emoticon-to-emoji", run(&EmoticonToEmoji, text)),
        ("symbol-joiner", run(&SymbolWords, text)),
        (
            "default-chain",
            run(
                &Chain::new()
                    .then(Invisible)
                    .then(UnicodeForm::nfc())
                    .then(Whitespace::collapse())
                    .then(Quotes)
                    .then(Dashes::new())
                    .then(CaseFold::lowercase())
                    .then(AccentFold::new()),
                text,
            ),
        ),
        (
            "umlaut-full-case-whitespace-chain",
            run(&Chain::new().then(GermanUmlauts::new()).then(CaseFold::full()).then(Whitespace::collapse()), text),
        ),
    ];
    assert_eq!(docs.len(), 30);
    for (name, doc) in &docs {
        let (_, violations) = check_document(doc);
        assert!(violations.is_empty(), "{name}: {violations:?}");
        scalar_contract(doc, normalized(doc));
    }
}
