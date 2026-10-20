//! `Layer::check` for [`Mentions`] and [`Speakers`]: a well-formed layer
//! passes and each broken invariant is named.

use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, Tag, TagSet, ViewId};

use super::*;

fn agreement() -> Agreement {
    Agreement::new(Number::Singular, Gender::Male, Animacy::Animate, Person::Third)
}

/// An entity mention "John" (token 0) and a pronoun "he" (token 2).
fn mentions() -> Mentions {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(10, 12));
    Mentions {
        spans,
        sentence: vec![Ix::new(0), Ix::new(0)],
        tokens: vec![IxRange::new(0, 1), IxRange::new(2, 3)],
        head: vec![Ix::new(0), Ix::new(2)],
        entity: Some(vec![Some(Ix::new(0)), None]),
        kind: vec![MentionKind::Proper, MentionKind::Pronominal],
        origin: vec![Origin::Entity, Origin::Pronoun],
        agreement: vec![agreement(), agreement()],
    }
}

fn refused(m: Mentions, needle: &str) {
    let err = m.check().expect_err("the layer must be refused");
    assert!(err.contains(needle), "{err}");
}

#[test]
fn well_formed_mentions_pass() {
    mentions().check().unwrap();
    let mut no_entities = mentions();
    no_entities.entity = None;
    no_entities.check().unwrap();
}

#[test]
fn mentions_columns_must_match_the_spans() {
    let mut m = mentions();
    m.kind.pop();
    refused(m, "for every mention");
    let mut m = mentions();
    m.entity = Some(vec![None]);
    refused(m, "an entity slot for every mention");
}

#[test]
fn mention_tokens_and_head_are_checked() {
    let mut m = mentions();
    m.tokens[1] = IxRange::new(2, 2);
    refused(m, "has no token");
    let mut m = mentions();
    m.head[1] = Ix::new(0);
    refused(m, "outside its tokens");
}

#[test]
fn kind_must_fit_origin() {
    let mut m = mentions();
    m.kind[0] = MentionKind::Pronominal;
    refused(m, "is a entity mention");
    let mut m = mentions();
    m.kind[1] = MentionKind::Nominal;
    refused(m, "mention 1");
    let mut m = mentions();
    m.origin = vec![Origin::Gold, Origin::Gold];
    m.entity = None;
    m.kind = vec![MentionKind::Nominal, MentionKind::Demonstrative];
    m.check().unwrap();
}

#[test]
fn entity_links_must_match_origin() {
    let mut m = mentions();
    m.entity = Some(vec![None, None]);
    refused(m, "names no entity");
    let mut m = mentions();
    m.entity = Some(vec![Some(Ix::new(0)), Some(Ix::new(1))]);
    refused(m, "was not lifted from one");
}

#[test]
fn agreement_bits_round_trip_and_reject_unpacked_values() {
    let a = agreement();
    assert_eq!(Agreement::from_bits(a.bits()), Some(a));
    assert_eq!(a.gender(), Gender::Male);
    assert_eq!(a.person(), Person::Third);
    assert_eq!(Agreement::from_bits(0b11), None);
    assert_eq!(Agreement::from_bits(0b11 << 4), None);
}

fn speakers() -> Speakers {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    spans.push(Span::new(6, 11));
    let speakers = TagSet::new("speakers", &["ann", "bob"]);
    Speakers { spans, speaker: vec![Tag(0), Tag(1)], speakers }
}

#[test]
fn speakers_check() {
    speakers().check().unwrap();
    let mut s = speakers();
    s.speaker.pop();
    assert!(s.check().unwrap_err().contains("a speaker for every turn"));
    let mut s = speakers();
    s.speaker[1] = Tag(2);
    assert!(s.check().unwrap_err().contains("a speaker outside speakers"));
    let mut s = speakers();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    spans.push(Span::new(6, 6));
    s.spans = spans;
    assert!(s.check().unwrap_err().contains("covers no text"));
}
