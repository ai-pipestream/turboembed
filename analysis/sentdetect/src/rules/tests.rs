//! The rule splitter end to end, one test per tailoring, each also run
//! with the tailoring off to show what it changes.

use super::*;
use turbo_analysis::{Document, Span};

fn split(splitter: RuleSplitter, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    splitter.split(text, &mut out);
    out.iter().map(|s: &Span| text[s.start as usize..s.end as usize].to_owned()).collect()
}

fn en(text: &str) -> Vec<String> {
    split(RuleSplitter::new("en"), text)
}

fn without(tag: &str, text: &str, off: impl Fn(&mut Rules)) -> Vec<String> {
    let mut rules = Rules::ALL;
    off(&mut rules);
    split(RuleSplitter::new(tag).with_rules(rules), text)
}

#[test]
fn plain_uax29() {
    let s = RuleSplitter::uax29();
    assert_eq!(split(s, "Hello World. My name is Jonas."), ["Hello World.", "My name is Jonas."]);
    assert_eq!(split(s, "Mr. Smith left.\nHe came back."), ["Mr.", "Smith left.", "He came back."]);
    assert_eq!(split(s, "etc. and so on"), ["etc. and so on"]);
    assert_eq!(split(s, "  \n \t "), Vec::<String>::new());
    assert_eq!(split(s, ""), Vec::<String>::new());
}

#[test]
fn abbreviations_and_starters() {
    assert_eq!(en("Mr. Smith left. He came back."), ["Mr. Smith left.", "He came back."]);
    assert_eq!(without("en", "Mr. Smith left.", |r| r.abbreviations = false), ["Mr.", "Smith left."]);
    // `Co.` is a CLDR suppression; `It` starts sentences.
    assert_eq!(
        en("They closed the deal with Pitt, Briggs & Co. at noon."),
        ["They closed the deal with Pitt, Briggs & Co. at noon."]
    );
    assert_eq!(
        en("They closed the deal with Pitt, Briggs & Co. It closed yesterday."),
        ["They closed the deal with Pitt, Briggs & Co.", "It closed yesterday."]
    );
    // German suppressions are case-sensitive and carry their full stop.
    let de = |t| split(RuleSplitter::new("de"), t);
    assert_eq!(de("Wir trafen Dr. Meyer. Er lachte."), ["Wir trafen Dr. Meyer.", "Er lachte."]);
    // A lower case list matches in any case (Polish, sentencex).
    let pl = |t| split(RuleSplitter::new("pl"), t);
    assert_eq!(pl("Jest np. Kowalski. Potem"), ["Jest np. Kowalski.", "Potem"]);
    assert_eq!(pl("Jest NP. Kowalski."), ["Jest NP. Kowalski."]);
    // The root rules have no abbreviations.
    assert_eq!(split(RuleSplitter::new("qaa"), "Mr. Smith left."), ["Mr.", "Smith left."]);
}

#[test]
fn initials() {
    assert_eq!(en("My name is Jonas E. Smith."), ["My name is Jonas E. Smith."]);
    assert_eq!(
        without("en", "My name is Jonas E. Smith.", |r| (r.initials, r.abbreviations) = (false, false)),
        ["My name is Jonas E.", "Smith."]
    );
    assert_eq!(
        en("We make a good team, you and I. Did you see Albert I. Jones yesterday?"),
        ["We make a good team, you and I.", "Did you see Albert I. Jones yesterday?"]
    );
    // Several initials in one token, in any script.
    assert_eq!(split(RuleSplitter::new("ru"), "Писал Л.Н. Толстой. Потом"), ["Писал Л.Н. Толстой.", "Потом"]);
    assert_eq!(en("I live in the U.S. How about you?"), ["I live in the U.S.", "How about you?"]);
    assert_eq!(en("I work for the U.S. Government in Virginia."), ["I work for the U.S. Government in Virginia."]);
    // An initial before a word in lower case is SB8's already.
    assert_eq!(en("He is A. not B."), ["He is A. not B."]);
}

#[test]
fn continuation_after_exclamation_and_question() {
    assert_eq!(
        en("She works at Yahoo! in the accounting department."),
        ["She works at Yahoo! in the accounting department."]
    );
    assert_eq!(en("'Why?' she asked. Nobody knew."), ["'Why?' she asked.", "Nobody knew."]);
    assert_eq!(
        without("en", "She works at Yahoo! in the department.", |r| r.continuation = false),
        ["She works at Yahoo!", "in the department."]
    );
}

#[test]
fn numbered_references() {
    assert_eq!(en("It is big.[1][2] Next one."), ["It is big.[1][2]", "Next one."]);
    assert_eq!(en("It is big. [3] Next one."), ["It is big. [3]", "Next one."]);
    assert_eq!(en("It is big.[1]"), ["It is big.[1]"]);
    // Without the tailoring, UAX #29 breaks after the bracket, and the
    // bracket rule then keeps the sentence whole.
    assert_eq!(without("en", "It is big.[1] Next one.", |r| r.references = false), ["It is big.[1] Next one."]);
    assert_eq!(
        without("en", "It is big.[1] Next one.", |r| (r.references, r.brackets) = (false, false)),
        ["It is big.[", "1] Next one."]
    );
    // Not references: glued to a word, or not a number.
    assert_eq!(en("See [a]. Next."), ["See [a].", "Next."]);
}

#[test]
fn brackets_and_quotes() {
    assert_eq!(en("He said \"Stop. Go.\" Then he left."), ["He said \"Stop. Go.\"", "Then he left."]);
    assert_eq!(en("Er sagte „Ja. Nein.“ Dann ging er."), ["Er sagte „Ja. Nein.“", "Dann ging er."]);
    assert_eq!(
        without("en", "He said \"Stop. Go.\" Then he left.", |r| r.brackets = false),
        ["He said \"Stop.", "Go.\"", "Then he left."]
    );
    // A paragraph break always ends a sentence, even inside a quotation.
    assert_eq!(en("He said \"Stop.\n\nGo.\" Then"), ["He said \"Stop.", "Go.\"", "Then"]);
}

#[test]
fn lists() {
    assert_eq!(en("1. The first item 2. The second item"), ["1. The first item", "2. The second item"]);
    assert_eq!(
        en("a. The first item b. The second item c. The third"),
        ["a. The first item", "b. The second item", "c. The third"]
    );
    assert_eq!(en("• 9. The first item • 10. The second item"), ["• 9. The first item", "• 10. The second item"]);
    assert_eq!(en("Shopping:\n- milk\n- bread"), ["Shopping:", "- milk", "- bread"]);
    assert_eq!(split(RuleSplitter::new("de"), "1. Dies ist eine Punkteliste."), ["1. Dies ist eine Punkteliste."]);
    assert_eq!(
        without("en", "1. The first item 2. The second", |r| r.lists = false),
        ["1.", "The first item 2.", "The second"]
    );
    // One number with a full stop inside a line is no marker.
    assert_eq!(en("It was 1990. Then it was over."), ["It was 1990.", "Then it was over."]);
}

#[test]
fn prose_with_single_letters_and_numbers_is_no_list() {
    // Two consecutive letters or numbers inside a line are not a list
    // unless the sequence reaches back to a line start or a bullet.
    assert_eq!(en("J. K. Rowling wrote it. Then she left."), ["J. K. Rowling wrote it.", "Then she left."]);
    assert_eq!(en("John A. Smith and Mary B. Jones came."), ["John A. Smith and Mary B. Jones came."]);
    assert_eq!(en("We tried plan A. Then plan B. Both failed."), ["We tried plan A.", "Then plan B.", "Both failed."]);
    assert_eq!(en("I was 7. Then I turned 8. Life went on."), ["I was 7.", "Then I turned 8.", "Life went on."]);
    let de = |t| split(RuleSplitter::new("de"), t);
    assert_eq!(de("Der 1. und 2. Platz. Dann"), ["Der 1. und 2. Platz.", "Dann"]);
    // A single letter that is not the first of its sequence does not start
    // a list at a line start, so the break after it stays.
    assert_eq!(en("x. A"), ["x.", "A"]);
    // A sequence that does start at a line start still chains inside it.
    assert_eq!(en("Steps:\na. Mix b. Bake"), ["Steps:", "a. Mix", "b. Bake"]);
}

#[test]
fn line_breaks() {
    let text = "It was a cold \nnight in the city.\n\nNext paragraph";
    assert_eq!(en(text), ["It was a cold \nnight in the city.", "Next paragraph"]);
    let lines = RuleSplitter::new("en").with_line_breaks(LineBreaks::Sentence);
    assert_eq!(split(lines, text), ["It was a cold", "night in the city.", "Next paragraph"]);
    assert_eq!(en("one\r\n\r\ntwo\u{2029}three"), ["one", "two", "three"]);
}

#[test]
fn ordinals() {
    let de = |t| split(RuleSplitter::new("de"), t);
    assert_eq!(de("Was geschah am 12. Juni? Nichts."), ["Was geschah am 12. Juni?", "Nichts."]);
    assert_eq!(de("Gefäße aus dem 1. -3. Jahrhundert."), ["Gefäße aus dem 1. -3. Jahrhundert."]);
    assert_eq!(de("Es waren 1990. Dann"), ["Es waren 1990.", "Dann"]);
    // English writes no ordinal full stop.
    assert_eq!(en("He was 12. Then he grew."), ["He was 12.", "Then he grew."]);
}

#[test]
fn bracketed_ellipsis() {
    let text = "\"Bohr [...] used the analogy of parallel stairways [...]\" (Smith 55).";
    assert_eq!(en(text), [text]);
    assert_eq!(
        without("en", text, |r| r.terminators = false),
        ["\"Bohr [...] used the analogy of parallel stairways [...]\"", "(Smith 55)."]
    );
}

#[test]
fn language_terminators() {
    let el = |t| split(RuleSplitter::new("el"), t);
    assert_eq!(el("Πού είναι; Εδώ."), ["Πού είναι;", "Εδώ."]);
    assert_eq!(split(RuleSplitter::new("en"), "Where; Here."), ["Where; Here."]);
    let hy = |t| split(RuleSplitter::new("hy"), t);
    assert_eq!(hy("Բարև Ձեզ: Իմ անունն է:"), ["Բարև Ձեզ:", "Իմ անունն է:"]);
    assert_eq!(en("I never meant that\u{2026} She left."), ["I never meant that\u{2026}", "She left."]);
    assert_eq!(
        without("en", "I meant that\u{2026} She left.", |r| r.terminators = false),
        ["I meant that\u{2026} She left."]
    );
    let ja = |t| split(RuleSplitter::new("ja"), t);
    assert_eq!(ja("今日は晴れ。明日は雨！"), ["今日は晴れ。", "明日は雨！"]);
}

#[test]
fn offsets_are_exact_outside_ascii() {
    // 😀 is four bytes; spans are byte offsets on character boundaries.
    let text = "Ça va 😀. Ok.";
    let mut out = Vec::new();
    RuleSplitter::new("fr").split(text, &mut out);
    assert_eq!(out, [Span::new(0, 12), Span::new(13, 16)]);
    for s in &out {
        assert!(text.is_char_boundary(s.start as usize) && text.is_char_boundary(s.end as usize));
    }
}

#[test]
fn scratch_reuse_gives_the_same_spans() {
    let splitter = RuleSplitter::new("en");
    let mut scratch = RuleScratch::default();
    let texts = ["One. Two.", "1. A 2. B", "\"A. B.\" C.", "", "x"];
    for text in texts.iter().chain(texts.iter()) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        splitter.split_with(text, &mut scratch, &mut a);
        splitter.split(text, &mut b);
        assert_eq!(a, b, "{text}");
    }
}

#[test]
fn annotator() {
    let doc = Document::new("Mr. Smith left. He came back.");
    let view = doc.get::<View>().unwrap();
    let detector = RuleSentenceDetector::new("en");
    let (sentences,) = detector.annotate((view,)).unwrap();
    assert_eq!(sentences.spans.columns(), (&[0, 16][..], &[15, 29][..]));
    let other = RuleSentenceDetector::new("de");
    assert_ne!(detector.config_hash(), other.config_hash());
    // de-AT uses the German rules, so it is the same configuration.
    assert_eq!(other.config_hash(), RuleSentenceDetector::new("de-AT").config_hash());
    let plain = RuleSentenceDetector { splitter: RuleSplitter::new("en").with_rules(Rules::NONE) };
    assert_ne!(detector.config_hash(), plain.config_hash());
}

#[test]
fn many_markers_and_long_tokens() {
    // Many inline list items in one paragraph (a marker has at most three
    // digits): each is a sentence.
    let text: String = (1..=999).map(|n| format!("{n}. item ")).collect();
    let mut out = Vec::new();
    RuleSplitter::new("en").split(&text, &mut out);
    assert_eq!(out.len(), 999);
    assert_eq!(&text[out[998].start as usize..out[998].end as usize], "999. item");
    // A text without spaces, a boundary after every `1.` (SB11; SB7 keeps
    // `a.B` together, but not `1.B`): the word before each full stop is
    // looked at only up to 64 characters back.
    let text = "1.B".repeat(5000);
    let mut out = Vec::new();
    RuleSplitter::new("en").split(&text, &mut out);
    assert_eq!(out.len(), 5001);
    // Initials are recognized up to that length; a longer run is not one.
    let initials = |n: usize| format!("{} Smith came.", "A.".repeat(n));
    assert_eq!(en(&initials(32)).len(), 1);
    assert_eq!(en(&initials(33)), ["A.".repeat(33), "Smith came.".to_owned()]);
}

#[test]
fn unpunctuated_scripts_are_one_sentence_per_paragraph() {
    // Thai separates sentences with a space and no punctuation; UAX #29 has
    // nothing to break at, and no rule here finds the end.
    let th = |t| split(RuleSplitter::new("th"), t);
    assert_eq!(th("วันนี้อากาศดี ฉันจะไปตลาด"), ["วันนี้อากาศดี ฉันจะไปตลาด"]);
    assert_eq!(th("วันนี้อากาศดี\n\nฉันจะไปตลาด"), ["วันนี้อากาศดี", "ฉันจะไปตลาด"]);
}
