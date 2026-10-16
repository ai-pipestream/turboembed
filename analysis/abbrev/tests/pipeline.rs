//! The step in a pipeline: definitions and mentions with exact byte
//! offsets, checked against the source text, and the links between them.

use turbo_abbrev::{AbbreviationAnnotator, AbbreviationForm, Abbreviations, DEFINITION, MENTION, detect};
use turbo_analysis::{
    Analyzer, Bound, Document, Edges, Error, Ix, Key, Layer, Provenance, Span, Spans, Status, ViewId,
};
use turbo_analysis_layers::{Sentences, Tokens};

type Links = Edges<Abbreviations, Abbreviations>;

/// Sentences ending at each ". " of `text`.
fn sentences(text: &str) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for (i, _) in text.match_indices(". ") {
        spans.push(Span::new(start as u32, (i + 1) as u32));
        start = i + 2;
    }
    spans.push(Span::new(start as u32, text.len() as u32));
    Sentences { spans }
}

/// Tokens at the spaces of `text`.
fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for piece in text.split(' ') {
        spans.push(Span::new(start as u32, (start + piece.len()) as u32));
        start += piece.len() + 1;
    }
    Tokens { spans, sentence: None }
}

fn run(text: &str, with_tokens: bool) -> Document {
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, sentences(text), Provenance::by("test")).unwrap();
    if with_tokens {
        doc.add(&Key::DEFAULT, tokens(text), Provenance::by("test")).unwrap();
    }
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(AbbreviationAnnotator))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    docs.pop().unwrap()
}

/// Every item as (start, end, form, its text, the text of its long form),
/// with each span's text read from the source by its offsets.
fn items(doc: &Document) -> Vec<(u32, u32, AbbreviationForm, &str, Option<&str>)> {
    let layer = doc.get::<Abbreviations>().unwrap();
    let links = doc.get::<Links>().unwrap();
    let text = doc.source();
    layer
        .spans
        .iter()
        .map(|(i, s)| {
            let long = layer.long_form(links, i).map(|l| &text[layer.spans.get(l).range()]);
            (s.start, s.end, layer.form(i), &text[s.range()], long)
        })
        .collect()
}

const TEXT: &str = "The World Health Organization (WHO) met in Geneva. \
WHO staff and the WHO Regional Office (WRO) agreed. \
Later, IL-2 (interleukin 2) rose; WHOLE grains and who else. \
The WRO said WHO's plan for IL-2.";

#[test]
fn definitions_and_mentions_have_exact_offsets() {
    use AbbreviationForm::*;
    let doc = run(TEXT, false);
    assert_eq!(
        items(&doc),
        [
            (4, 29, Long, "World Health Organization", None),
            (31, 34, Short, "WHO", Some("World Health Organization")),
            (51, 54, Mention, "WHO", Some("World Health Organization")),
            (69, 72, Mention, "WHO", Some("World Health Organization")),
            (69, 88, Long, "WHO Regional Office", None),
            (90, 93, Short, "WRO", Some("WHO Regional Office")),
            (110, 114, Short, "IL-2", Some("interleukin 2")),
            (116, 129, Long, "interleukin 2", None),
            (168, 171, Mention, "WRO", Some("WHO Regional Office")),
            (177, 180, Mention, "WHO", Some("World Health Organization")),
            (192, 196, Mention, "IL-2", Some("interleukin 2")),
        ]
    );
    // The offsets are the source's own bytes.
    assert_eq!(&TEXT[177..182], "WHO's");
}

#[test]
fn links_say_definition_or_mention() {
    let doc = run(TEXT, false);
    let layer = doc.get::<Abbreviations>().unwrap();
    let links = doc.get::<Links>().unwrap();
    assert_eq!(links.len(), 8, "one link per short form and mention");
    for (_, e) in links.iter() {
        let expected = if layer.form(e.from) == AbbreviationForm::Mention { MENTION } else { DEFINITION };
        assert_eq!(links.kinds.label(e.kind), expected);
        assert_eq!(layer.form(e.to), AbbreviationForm::Long);
        assert_eq!(e.score, 1.0);
    }
    // The defining "WHO" and its three mentions point at "World Health
    // Organization".
    assert_eq!(links.incoming(Ix::new(0)).count(), 4);
    let key = Key::<Links>::DEFAULT.id();
    assert_eq!(doc.made_on(key), [Key::<Abbreviations>::DEFAULT.id()]);
}

#[test]
fn tokens_make_mentions_whole_tokens() {
    let doc = run(TEXT, true);
    let mentions: Vec<(u32, &str)> =
        items(&doc).into_iter().filter(|i| i.2 == AbbreviationForm::Mention).map(|i| (i.0, i.3)).collect();
    // "WHO's" and "IL-2." are single tokens, so neither holds a mention.
    assert_eq!(mentions, [(51, "WHO"), (69, "WHO"), (168, "WRO")]);
    let provenance = doc.provenance(Key::<Abbreviations>::DEFAULT.id()).unwrap();
    assert!(provenance.reads.contains(&Key::<Tokens>::DEFAULT.id()));
}

#[test]
fn a_redefinition_takes_over() {
    let text = "Alpha beta (AB) one. AB two. Apple banana (AB) three. AB four.";
    let doc = run(text, false);
    let found: Vec<(u32, &str, Option<&str>)> =
        items(&doc).into_iter().filter(|i| i.2 != AbbreviationForm::Long).map(|i| (i.0, i.3, i.4)).collect();
    assert_eq!(
        found,
        [
            (12, "AB", Some("Alpha beta")),
            (21, "AB", Some("Alpha beta")),
            (43, "AB", Some("Apple banana")),
            (54, "AB", Some("Apple banana")),
        ]
    );
}

#[test]
fn a_short_form_inside_another_is_not_a_mention() {
    // "IL" is in the defining "IL-2" and in the later mention of "IL-2".
    let text = "The interleukin (IL) family. Then IL-2 (interleukin 2) differs. IL-2 and IL rose.";
    let doc = run(text, false);
    let found: Vec<(u32, AbbreviationForm, &str)> = items(&doc).into_iter().map(|i| (i.0, i.2, i.3)).collect();
    assert_eq!(
        found,
        [
            (4, AbbreviationForm::Long, "interleukin"),
            (17, AbbreviationForm::Short, "IL"),
            (34, AbbreviationForm::Short, "IL-2"),
            (40, AbbreviationForm::Long, "interleukin 2"),
            (64, AbbreviationForm::Mention, "IL-2"),
            (73, AbbreviationForm::Mention, "IL"),
        ]
    );
}

#[test]
fn partially_overlapping_mentions_and_mentions_crossing_a_long_form_do_not_cross() {
    // "AB-CD" and "CD-EF" are both defined; in "AB-CD-EF", "AB-CD" (0..5) and
    // "CD-EF" (3..8) partially cross, so only the earlier mention is kept.
    let text = "Alpha beta charlie delta (AB-CD) first. \
Charlie delta echo foxtrot (CD-EF) second. \
Delta echo (D E) third. \
Later AB-CD-EF and Alpha Delta echo (ADE) appeared.";
    let doc = run(text, false);
    let layer = doc.get::<Abbreviations>().unwrap();
    assert_eq!(layer.check(), Ok(()));
    let mentions: Vec<&str> =
        items(&doc).into_iter().filter(|i| i.2 == AbbreviationForm::Mention).map(|i| i.3).collect();
    assert_eq!(mentions, ["AB-CD"]);

    // In "Later B C delta (CD) y.", a candidate mention of "B C" starts before
    // and ends inside the long form "C delta", so it must be refused rather
    // than crossing the long form.
    let cross_long = "Bob Carl (B C) x. Later B C delta (CD) y. End.";
    let doc2 = run(cross_long, false);
    let layer2 = doc2.get::<Abbreviations>().unwrap();
    assert_eq!(layer2.check(), Ok(()));
    let mentions2: Vec<&str> =
        items(&doc2).into_iter().filter(|i| i.2 == AbbreviationForm::Mention).map(|i| i.3).collect();
    assert!(mentions2.is_empty(), "{mentions2:?}");
}

#[test]
fn mentions_before_a_definition_are_not_linked() {
    let text = "WHO is first. Then the World Health Organization (WHO) speaks.";
    let doc = run(text, false);
    let found: Vec<AbbreviationForm> = items(&doc).into_iter().map(|i| i.2).collect();
    assert_eq!(found, [AbbreviationForm::Long, AbbreviationForm::Short]);
}

#[test]
fn non_ascii_offsets_are_on_character_boundaries() {
    let text = "La Organización Mundial de la Salud (OMS) habló. La OMS añadió más.";
    let doc = run(text, false);
    let found: Vec<(u32, u32, &str)> = items(&doc).into_iter().map(|i| (i.0, i.1, i.3)).collect();
    assert_eq!(found, [(3, 36, "Organización Mundial de la Salud"), (38, 41, "OMS"), (54, 57, "OMS")]);
}

#[test]
fn a_document_with_no_definitions_has_empty_layers() {
    let doc = run("Nothing here (really) at all.", false);
    assert!(doc.get::<Abbreviations>().unwrap().spans.is_empty());
    assert_eq!(doc.get::<Links>().unwrap().len(), 0);
    assert_eq!(doc.status(Key::<Abbreviations>::DEFAULT.id()), Some(&Status::Complete));
}

#[test]
fn named_instances_are_linked() {
    let mut docs = vec![Document::new(TEXT)];
    docs[0].add(&Key::DEFAULT, sentences(TEXT), Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .step(Bound::new(AbbreviationAnnotator).write(0, "sh").write(1, "sh"))
        .build()
        .unwrap();
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let key = Key::<Links>::named("sh").id();
    assert_eq!(docs[0].made_on(key), [Key::<Abbreviations>::named("sh").id()]);
    assert_eq!(docs[0].get_key(&Key::<Abbreviations>::named("sh")).unwrap().spans.len(), 11);
}

#[test]
fn sentences_in_another_view_fail_the_document() {
    use turbo_analysis::Annotator;
    let doc = Document::new(TEXT);
    let view = doc.get::<turbo_analysis::View>().unwrap();
    let other = Sentences { spans: Spans::new(ViewId(7)) };
    let Err(Error::Annotator { detail, .. }) = AbbreviationAnnotator.annotate((view, &other, None)) else {
        panic!("sentences in another view are refused");
    };
    assert_eq!(detail, "the sentences are in another view");
}

#[test]
fn many_definitions_are_found_in_one_pass() {
    // 50,000 distinct short forms, each defined once and mentioned once.
    // Searching the rest of the text once per definition took minutes.
    let mut text = String::new();
    let mut sentences = Vec::new();
    for i in 0..50_000u32 {
        let [a, b, c, d] = [i % 26, i / 26 % 26, i / 676 % 26, i / 17_576].map(|n| char::from(b'a' + n as u8));
        let short: String = [a, b, c, d].map(|c| c.to_ascii_uppercase()).iter().collect();
        let start = text.len() as u32;
        text.push_str(&format!("{a}x {b}x {c}x {d}x ({short}) and {short}."));
        sentences.push(Span::new(start, text.len() as u32));
        text.push(' ');
    }
    let (layer, links) = detect(&text, ViewId::SOURCE, sentences, None);
    assert_eq!(layer.check(), Ok(()));
    assert_eq!(links.check(), Ok(()));
    let count = |form| layer.spans.iter().filter(|&(i, _)| layer.form(i) == form).count();
    assert_eq!(count(AbbreviationForm::Short), 50_000);
    assert_eq!(count(AbbreviationForm::Mention), 50_000);
}

#[test]
fn odd_input_gives_valid_layers() {
    // Random strings of parentheses, punctuation, capitals and multi-byte
    // letters, split into sentences at ". " and into tokens at spaces.
    const PIECES: [&str; 18] =
        ["A", "B", "a", "b", " ", " ", "(", ")", ". ", ",", "-", "é", "\"", "[", "AB", "ab", "É", "'s"];
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..20_000 {
        let text: String = (0..next() % 40).map(|_| PIECES[(next() % PIECES.len() as u64) as usize]).collect();
        let sentences = sentences(&text);
        let (mut starts, mut ends) = (Vec::new(), Vec::new());
        for (_, s) in tokens(&text).spans.iter().filter(|(_, s)| !s.is_empty()) {
            starts.push(s.start);
            ends.push(s.end);
        }
        let (layer, links) =
            detect(&text, ViewId::SOURCE, sentences.spans.iter().map(|(_, s)| s), Some((&starts, &ends)));
        assert_eq!(layer.check(), Ok(()), "{text:?}");
        assert_eq!(links.check(), Ok(()), "{text:?}");
        for (i, s) in layer.spans.iter() {
            assert!(text.is_char_boundary(s.start as usize) && text.is_char_boundary(s.end as usize), "{text:?}");
            assert_eq!(layer.long_form(&links, i).is_none(), layer.form(i) == AbbreviationForm::Long, "{text:?}");
        }
    }
}
