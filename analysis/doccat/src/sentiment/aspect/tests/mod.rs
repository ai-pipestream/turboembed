use super::*;
use turbo_analysis::{Analyzer, Document, IxRange, Key, Provenance, Span, Spans, ViewId};

fn tokenize_simple(text: &str) -> Tokens {
    let mut spans = Spans::<Tokens>::new(ViewId(0));
    let mut sentence = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let start = i;
        if bytes[i].is_ascii_alphanumeric() {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'\'') {
                i += 1;
            }
        } else {
            i += 1;
        }
        spans.push(Span::new(start as u32, i as u32));
        sentence.push(Ix::new(0));
    }
    Tokens { spans, sentence: Some(sentence) }
}

fn run_rule_absa(text: &str) -> Document {
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    docs.into_iter().next().unwrap()
}

#[test]
fn battery_life_is_great_but_screen_scratches_gives_opposite_polarity_aspects() {
    // Exact example from #322 summary:
    // "battery life is great but the screen scratches"
    let doc = run_rule_absa("battery life is great but the screen scratches");
    let tokens = doc.get::<Tokens>().unwrap();
    let opinions = doc.get::<Opinions>().unwrap();
    let aspects = doc.get::<Aspects>().unwrap();

    assert_eq!(aspects.len(), 2);
    assert_eq!(opinions.len(), 2);

    let a0 = aspects.get(Ix::new(0));
    let a1 = aspects.get(Ix::new(1));
    assert_eq!(a0.target, AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) });
    assert_eq!(aspects.category_label(Ix::new(0)), Some("battery"));
    assert_eq!(a0.polarity.label(), Polarity::Positive);
    assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)]);
    assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "great"

    assert_eq!(a1.target, AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) });
    assert_eq!(aspects.category_label(Ix::new(1)), Some("display"));
    assert_eq!(a1.polarity.label(), Polarity::Negative);
    assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)]);
    assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(7, 8)); // "scratches"
    let _ = tokens;
}

#[test]
fn negation_with_a_curly_apostrophe_still_negates() {
    // A tokenizer that keeps contractions whole hands the extractor
    // tokens like "isn’t" (U+2019): the negation cues, written with
    // ASCII apostrophes, must still fire.
    let text = "The screen isn\u{2019}t great.";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    let mut spans = Spans::<Tokens>::new(ViewId(0));
    for (a, b) in [(0, 3), (4, 10), (11, 18), (19, 24), (24, 25)] {
        spans.push(Span::new(a, b));
    }
    let tokens = Tokens { spans, sentence: Some(vec![Ix::new(0); 5]) };
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];
    let opinions = doc.get::<Opinions>().unwrap();
    let aspects = doc.get::<Aspects>().unwrap();
    assert_eq!(opinions.len(), 1);
    assert!(opinions.get(Ix::new(0)).negated, "the curly-apostrophe negation must reach the opinion");
    assert_eq!(aspects.len(), 1);
    assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Negative);
}

#[test]
fn double_propagation_over_dependencies_extracts_new_targets_and_conjoined_aspects() {
    // "The sushi and tiramisu were delicious."
    // Notice "tiramisu" is not in SINGLE_WORD_ASPECT_TERMS, so R3 (`conj` from "sushi" or R12 from "delicious")
    // extracts it via Dependencies!
    let text = "The sushi and tiramisu were delicious.";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

    // Tokens: 0:"The", 1:"sushi", 2:"and", 3:"tiramisu", 4:"were", 5:"delicious", 6:"."
    let rels = TagSet::new("ud", &["root", "det", "nsubj", "cc", "conj", "cop", "punct"]);
    let deps = Dependencies::<Tokens> {
        head: vec![
            Some(Ix::new(1)), // The -> sushi
            Some(Ix::new(5)), // sushi -> delicious (nsubj)
            Some(Ix::new(3)), // and -> tiramisu
            Some(Ix::new(1)), // tiramisu -> sushi (conj)
            Some(Ix::new(5)), // were -> delicious (cop)
            None,             // delicious (root)
            Some(Ix::new(5)), // . -> delicious
        ],
        relation: vec![
            rels.tag("det").unwrap(),
            rels.tag("nsubj").unwrap(),
            rels.tag("cc").unwrap(),
            rels.tag("conj").unwrap(),
            rels.tag("cop").unwrap(),
            rels.tag("root").unwrap(),
            rels.tag("punct").unwrap(),
        ],
        relations: rels,
    };
    doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();

    let absa_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    absa_pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let aspects = doc.get::<Aspects>().unwrap();
    assert_eq!(aspects.len(), 2);
    assert_eq!(aspects.get(Ix::new(0)).target.tokens(), Some(IxRange::new(1, 2))); // "sushi"
    assert_eq!(aspects.get(Ix::new(1)).target.tokens(), Some(IxRange::new(3, 4))); // "tiramisu" extracted by R3/R1!
    assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
    assert_eq!(aspects.get(Ix::new(1)).polarity.label(), Polarity::Positive);
}

#[test]
fn aspect_and_opinion_under_the_same_governor_link_without_panicking() {
    // "The battery works well": `nsubj(works, battery)` and `advmod(works, well)` share
    // the governor "works". Linking used to index an empty slice on this shape.
    let text = "The battery works well";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId(0));
    sents.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();
    let rels = TagSet::new("ud", &["root", "det", "nsubj", "advmod"]);
    let deps = Dependencies::<Tokens> {
        head: vec![Some(Ix::new(1)), Some(Ix::new(2)), None, Some(Ix::new(2))],
        relation: vec![
            rels.tag("det").unwrap(),
            rels.tag("nsubj").unwrap(),
            rels.tag("root").unwrap(),
            rels.tag("advmod").unwrap(),
        ],
        relations: rels,
    };
    doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(RuleAspectSentiment::new())
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let aspects = docs[0].get::<Aspects>().unwrap();
    let opinions = docs[0].get::<Opinions>().unwrap();
    assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "well"
    let battery = (0..aspects.len())
        .map(|i| Ix::<Aspects>::new(i as u32))
        .find(|&ix| aspects.get(ix).target.tokens() == Some(IxRange::new(1, 2)))
        .expect("battery is an aspect");
    assert_eq!(aspects.get(battery).polarity.label(), Polarity::Positive);
    assert_eq!(aspects.get(battery).opinions, &[Ix::<Opinions>::new(0)]);
}

#[test]
fn implicit_aspect_anchors_to_sentence_and_scores_category() {
    let doc = run_rule_absa("Way too expensive and completely overpriced.");
    let aspects = doc.get::<Aspects>().unwrap();
    let opinions = doc.get::<Opinions>().unwrap();
    assert_eq!(aspects.len(), 1);
    assert_eq!(opinions.len(), 2);
    let a0 = aspects.get(Ix::new(0));
    assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) });
    assert_eq!(aspects.category_label(Ix::new(0)), Some("price"));
    assert_eq!(a0.polarity.label(), Polarity::Negative);
    assert_eq!(a0.opinions.len(), 2);
}

mod benchmark;
