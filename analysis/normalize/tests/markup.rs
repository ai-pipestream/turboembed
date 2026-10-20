//! Markup stripping and character-reference decoding:
//! adversary fixtures, byte-exact projection of normalized
//! word spans back to the raw markup, and the WHATWG numeric
//! replacement rules.

use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_normalize::{HtmlEntityDecodeNormalizer, HtmlTagStripNormalizer, NormalizeStep, apply, normalize};

fn stripped(text: &str) -> String {
    apply(&HtmlTagStripNormalizer::new(), text).into_owned()
}

fn decoded(text: &str) -> String {
    apply(&HtmlEntityDecodeNormalizer::new(), text).into_owned()
}

#[test]
fn tags_attributes_and_inline_boundaries() {
    assert_eq!(stripped(r#"<a title="x>y">Hello</a>"#), "Hello");
    assert_eq!(stripped("<a title='x>y'>Hello</a>"), "Hello");
    assert_eq!(stripped("un<b>break</b>able"), "unbreakable");
    assert_eq!(stripped("<p>One</p><p>Two</p>"), "One Two ");
    assert_eq!(stripped("One<br>Two"), "One Two");
    assert_eq!(stripped("One<br/>Two"), "One Two");
    assert_eq!(apply(&HtmlTagStripNormalizer::without_boundary_spaces(), "<p>One</p><p>Two</p>"), "OneTwo");
    // A stray < is text, and so is </ followed by space.
    assert_eq!(stripped("1 < 2 and a </ b"), "1 < 2 and a </ b");
    assert_eq!(stripped("a <3 b"), "a <3 b");
    // Unterminated constructs drop to the end.
    assert_eq!(stripped("keep <div class=\"x"), "keep ");
    assert_eq!(stripped("keep <!-- open"), "keep ");
}

#[test]
fn comments_doctype_pi_and_cdata() {
    assert_eq!(stripped("a <!-- x > y --> b"), "a  b");
    assert_eq!(stripped("a <!--> b"), "a  b");
    assert_eq!(stripped("a <!---> b"), "a  b");
    assert_eq!(stripped("<!DOCTYPE html>text"), "text");
    assert_eq!(stripped("<!DOCTYPE root [ <!ENTITY x \"a>b\"> ]>text"), "text");
    assert_eq!(stripped("<?xml version=\"1.0\"?>text"), "text");
    assert_eq!(stripped("<![CDATA[a <b> &amp; c]]>"), "a <b> &amp; c");
    assert_eq!(stripped("x <![CDATA[unterminated"), "x unterminated");
    assert_eq!(stripped("a </> b"), "a  b");
}

#[test]
fn raw_text_bodies_drop_and_rcdata_stays() {
    assert_eq!(stripped("<script>if (a<b) { go(\"</p>\"); }</script>after"), "after");
    assert_eq!(stripped("<style>.a > .b { color: red }</style>after"), "after");
    assert_eq!(stripped("<SCRIPT>x</ScRiPt>after"), "after");
    assert_eq!(stripped("before <script>never closed"), "before ");
    assert_eq!(stripped("<script/>after"), "after");
    assert_eq!(stripped("<title>a < b</title>after"), "a < bafter");
    assert_eq!(stripped("<textarea><b>not bold</b></textarea>after"), "<b>not bold</b>after");
    // The closing name must be a whole tag name.
    assert_eq!(stripped("<script>var s = \"</scriptx>\";</script>after"), "after");
}

#[test]
fn character_references_decode_once() {
    assert_eq!(decoded("Hello &amp;lt;world&eacute;!"), "Hello &lt;world\u{e9}!");
    assert_eq!(decoded("&lt;tag&gt; &quot;q&quot; &apos;a&apos;"), "<tag> \"q\" 'a'");
    assert_eq!(decoded("&nbsp;x"), "\u{a0}x");
    assert_eq!(decoded("&CounterClockwiseContourIntegral;"), "\u{2233}");
    assert_eq!(decoded("&unknown; & plain"), "&unknown; & plain");
    // Legacy names without the semicolon, longest match wins.
    assert_eq!(decoded("&ampx"), "&x");
    assert_eq!(decoded("&copy2026"), "\u{a9}2026");
    assert_eq!(decoded("&notit;"), "\u{ac}it;");
    // Decimal and hex, with and without semicolons.
    assert_eq!(decoded("&#160;&#xA0;&#xa0"), "\u{a0}\u{a0}\u{a0}");
    assert_eq!(decoded("&#65B"), "AB");
    assert_eq!(decoded("&#x1F600;"), "\u{1F600}");
    assert_eq!(decoded("&#; &#xG;"), "&#; &#xG;");
}

#[test]
fn numeric_reference_replacements_follow_whatwg() {
    assert_eq!(decoded("&#0;"), "\u{FFFD}");
    assert_eq!(decoded("&#xD800;"), "\u{FFFD}");
    assert_eq!(decoded("&#x110000;"), "\u{FFFD}");
    assert_eq!(decoded("&#99999999999999999999;"), "\u{FFFD}");
    // The C1 table: 0x80 is the euro sign, 0x96 an en dash,
    // and 0x81 keeps its value.
    assert_eq!(decoded("&#128;"), "\u{20AC}");
    assert_eq!(decoded("&#150;"), "\u{2013}");
    assert_eq!(decoded("&#129;"), "\u{81}");
    assert_eq!(decoded("&#x9F;"), "\u{178}");
}

#[test]
fn normalized_words_project_back_to_raw_markup_bytes() {
    let raw = "<a title=\"x>y\">Hello &amp;lt;world&eacute;!</a>";
    let mut doc = Document::new(raw);
    let strip = Bound::new(NormalizeStep::new(HtmlTagStripNormalizer::new())).write(0, "stripped");
    let decode =
        Bound::new(NormalizeStep::new(HtmlEntityDecodeNormalizer::new())).read(0, "stripped").write(0, "decoded");
    let analyzer = Analyzer::builder().step(strip).step(decode).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    // Word spans over the decoded view, as a tokenizer on
    // that view would produce them.
    let decoded_id = doc.get_key::<View>(&Key::named("decoded")).unwrap().id();
    let clean = "Hello &lt;world\u{e9}!";
    let mut spans = Spans::new(decoded_id);
    for word in ["Hello", "&lt;world\u{e9}"] {
        let at = clean.find(word).unwrap();
        spans.push(Span::new(at as u32, (at + word.len()) as u32));
    }
    let tokens = Tokens { spans, sentence: None };
    assert_eq!(
        doc.get_key::<View>(&Key::named("decoded")).map(|v| v.text()),
        Some(clean),
        "the decoded view holds the expected text"
    );
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let back = doc.project_spans(&tokens.spans, ViewId::SOURCE).unwrap();
    let raw_of = |i: u32| {
        let s = back.get(Ix::new(i));
        &raw[s.start as usize..s.end as usize]
    };
    assert_eq!(raw_of(0), "Hello");
    assert_eq!(raw_of(1), "&amp;lt;world&eacute;");
}

#[test]
fn alignment_edits_are_exact() {
    let (clean, alignment) = normalize(&HtmlTagStripNormalizer::new(), "<p>a</p>b");
    assert_eq!(clean, "a b");
    // One edit for <p> (to nothing at the start), one for
    // </p> (to the boundary space).
    assert_eq!(alignment.edits().len(), 2);
    let (decoded_text, alignment) = normalize(&HtmlEntityDecodeNormalizer::new(), "a &amp; b");
    assert_eq!(decoded_text, "a & b");
    assert_eq!(alignment.edits().len(), 1);
    assert_eq!(alignment.edits()[0].parent, Span::new(2, 7));
    assert_eq!(alignment.edits()[0].child, Span::new(2, 3));
}

#[test]
fn config_hashes_are_deterministic_and_distinct() {
    use turbo_analysis::Annotator;
    let a = NormalizeStep::new(HtmlTagStripNormalizer::new());
    let b = NormalizeStep::new(HtmlTagStripNormalizer::without_boundary_spaces());
    assert_eq!(a.config_hash(), NormalizeStep::new(HtmlTagStripNormalizer::new()).config_hash());
    assert_ne!(a.config_hash(), b.config_hash());
    assert_eq!(NormalizeStep::new(HtmlEntityDecodeNormalizer::new()).config_hash(), 0);
}
