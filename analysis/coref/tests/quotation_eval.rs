//! A small attribution evaluation of `QuotationAnnotator` without a parse:
//! the speaker found for each direct quote against a gold speaker, on news
//! and dialog lines with quotes before, after and around their cues, and
//! the share of dialog turns given the right chain with and without turn
//! alternation.

use turbo_analysis::{Document, Groups, Ix, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_coref::{
    Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person, QuotationAnnotator, QuotationConfig,
};

/// Text, then the gold speaker of each quote in order (`""` for none).
const LINES: &[(&str, &[&str])] = &[
    ("Smith said , \" We grew . \"", &["Smith"]),
    ("\" Sales fell , \" said Jones .", &["Jones"]),
    ("\" We grew , \" she said , \" and we hired . \"", &["she", "she"]),
    ("Ann said , \" Hi . \" Bob said , \" Hello . \"", &["Ann", "Bob"]),
    ("\" Hi , \" said Ann . \" Hello , \" said Bob .", &["Ann", "Bob"]),
    ("The minister warned : \" Prices will rise . \"", &["minister"]),
    ("\" It is over , \" Garcia told reporters .", &["Garcia"]),
    ("He paused . \" Maybe , \" he added .", &["he"]),
];

fn doc(text: &str) -> Document {
    let mut spans = Spans::new(ViewId(0));
    let mut at = 0;
    for w in text.split(' ') {
        spans.push(Span::new(at as u32, (at + w.len()) as u32));
        at += w.len() + 1;
    }
    let n = spans.len();
    let mut d = Document::new(text);
    let mut sentence = Spans::new(ViewId(0));
    sentence.push(Span::new(0, text.len() as u32));
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentence }, Provenance::by("test:sentences")).unwrap();
    d.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans, sentence: Some(vec![Ix::new(0); n]) },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    d
}

#[test]
fn surface_attribution_finds_most_speakers() {
    let (mut right, mut total) = (0, 0);
    for (text, gold) in LINES {
        let d = doc(text);
        let q = QuotationAnnotator::new().quotations(&d).unwrap();
        let found: Vec<&str> =
            q.iter().map(|q| q.speaker.map_or("", |s| &text[s.start as usize..s.end as usize])).collect();
        assert_eq!(found.len(), gold.len(), "{text}");
        for (f, g) in found.iter().zip(gold.iter()) {
            total += 1;
            right += usize::from(f.ends_with(g));
            if !f.ends_with(g) {
                println!("miss: {text} -> {f:?}, gold {g:?}");
            }
        }
    }
    let accuracy = right as f64 / total as f64;
    println!("speaker accuracy {right}/{total} = {accuracy:.3}");
    assert!(accuracy >= 0.9, "{accuracy}");
}

/// A four-turn dialog where only the first two turns are attributed.
#[test]
fn dialog_alternation_recovers_unattributed_turns() {
    let text = "Ann said , \" Hi . \" Bob said , \" Hello . \" \" How are you ? \" \" Fine . \"";
    let mut d = doc(text);
    let unk = Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown);
    let mut spans = Spans::new(ViewId(0));
    spans.push(Span::new(0, 3));
    spans.push(Span::new(19, 22));
    let mentions = Mentions {
        spans,
        sentence: vec![Ix::new(0); 2],
        tokens: vec![IxRange::new(0, 1), IxRange::new(7, 8)],
        head: vec![Ix::new(0), Ix::new(7)],
        entity: None,
        kind: vec![MentionKind::Proper; 2],
        origin: vec![Origin::NounPhrase; 2],
        agreement: vec![unk; 2],
    };
    d.add(&Key::<Mentions>::DEFAULT, mentions, Provenance::by("test:mentions")).unwrap();
    d.add(
        &Key::<Groups<Mentions>>::DEFAULT,
        Groups::new([vec![Ix::new(0)], vec![Ix::new(1)]]),
        Provenance::by("test:coref"),
    )
    .unwrap();
    let gold = [Some(0), Some(1), Some(0), Some(1)];
    let score = |config: QuotationConfig| {
        let q = QuotationAnnotator::new().with_config(config).quotations(&d).unwrap();
        q.iter().zip(gold).filter(|(q, g)| q.speaker_chain == *g).count()
    };
    let with = score(QuotationConfig::default());
    let without = score(QuotationConfig { alternate_dialog: false, ..Default::default() });
    println!("dialog turns with the right chain: {with}/4 with alternation, {without}/4 without");
    assert_eq!((with, without), (4, 2));
}
