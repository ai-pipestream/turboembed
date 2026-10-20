//! A `CorefInput` built by hand, outside a `Document`, is validated in
//! full: malformed layers, columns, cross-layer indexes and spans that do
//! not slice the text are refused with an error, never a panic.

use turbo_analysis::{Ix, IxRange, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, PosTags, Sentences, Tokens};
use turbo_coref::{CorefAnnotator, CorefInput, Speakers};

fn spans<L>(items: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in items {
        s.push(Span::new(a, b));
    }
    s
}

/// One sentence over `text`, one token per `(start, end)`, one entity.
struct Layers {
    text: String,
    sentences: Sentences,
    tokens: Tokens,
    tags: PosTags,
    entities: Entities,
    chunks: Option<PhraseChunks>,
    phrases: Option<Phrases>,
    speakers: Option<Speakers>,
}

impl Layers {
    fn new(text: &str, tokens: &[(u32, u32)], entity: (u32, u32)) -> Layers {
        let end = text.len() as u32;
        let n = tokens.len() as u32;
        Layers {
            text: text.to_owned(),
            sentences: Sentences { spans: spans(&[(0, end)]) },
            tokens: Tokens { spans: spans(tokens), sentence: None },
            tags: PosTags { tagset: TagSet::new("penn", &["NNP"]), tags: vec![Tag(0); tokens.len()], prob: None },
            entities: Entities {
                spans: spans(&[entity]),
                tokens: vec![IxRange::new(0, n.min(1))],
                kind: vec![Tag(0)],
                kinds: TagSet::new("types", &["person"]),
                score: vec![1.0],
            },
            chunks: None,
            phrases: None,
            speakers: None,
        }
    }

    fn input(&self) -> CorefInput<'_> {
        CorefInput {
            text: &self.text,
            view: ViewId::SOURCE,
            sentences: &self.sentences,
            tokens: &self.tokens,
            tags: &self.tags,
            entities: &self.entities,
            chunks: self.chunks.as_ref(),
            phrases: self.phrases.as_ref(),
            speakers: self.speakers.as_ref(),
        }
    }

    fn refused(&self, needle: &str) {
        let err = CorefAnnotator::default().resolve(&self.input()).expect_err("the input must be refused");
        assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    }
}

fn ada() -> Layers {
    Layers::new("Ada left", &[(0, 3), (4, 8)], (0, 3))
}

#[test]
fn well_formed_hand_built_input_resolves() {
    let (mentions, _, _) = CorefAnnotator::default().resolve(&ada().input()).unwrap();
    assert_eq!(mentions.spans.len(), 1);
}

#[test]
fn spans_past_the_text_are_refused() {
    Layers::new("A", &[(0, 1)], (0, 5)).refused("does not slice the text");
    let mut l = Layers::new("A", &[(0, 1)], (0, 1));
    l.sentences = Sentences { spans: spans(&[(0, 9)]) };
    l.refused("opennlp:sentences item 0");
}

#[test]
fn spans_off_char_boundaries_are_refused() {
    // "é" is two bytes: a token ending at byte 1 splits it.
    Layers::new("\u{e9}x", &[(0, 1), (2, 3)], (2, 3)).refused("opennlp:tokens item 0");
    let mut l = Layers::new("ab \u{e9}c", &[(0, 2), (3, 6)], (0, 2));
    let mut entity = Spans::new(ViewId::SOURCE);
    entity.try_push_fragments(&[Span::new(0, 1), Span::new(4, 5)]).unwrap();
    l.entities.spans = entity;
    l.refused("opennlp:entities item 0");
}

#[test]
fn tags_and_kinds_outside_their_tag_sets_are_refused() {
    let mut l = ada();
    l.tags.tags[1] = Tag(9);
    l.refused("opennlp:pos");
    let mut l = ada();
    l.entities.kind[0] = Tag(4);
    l.refused("opennlp:entities");
    let mut l = ada();
    l.tags.tags.pop();
    l.refused("aligned");
}

#[test]
fn short_columns_are_refused() {
    let mut l = ada();
    l.entities.kind.clear();
    l.refused("opennlp:entities");
    let mut l = ada();
    l.chunks = Some(PhraseChunks {
        spans: spans(&[(0, 3)]),
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![],
        kinds: TagSet::new("chunks", &["NP"]),
    });
    l.refused("opennlp:chunks");
    let mut l = ada();
    l.speakers = Some(Speakers { spans: spans(&[(0, 8)]), speaker: vec![Tag(3)], speakers: TagSet::new("s", &["a"]) });
    l.refused("opennlp:speakers");
}

#[test]
fn token_indexes_past_the_tokens_are_refused() {
    let mut l = ada();
    l.phrases = Some(Phrases {
        spans: spans(&[(0, 3)]),
        tokens: vec![IxRange::new(7, 8)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("phrases", &["NP"]),
        head: vec![Ix::new(7)],
        parent: vec![None],
    });
    l.refused("opennlp:phrases item 0 has tokens 7..8 past 2 tokens");
    let mut l = ada();
    l.entities.tokens[0] = IxRange::new(0, 5);
    l.refused("opennlp:entities item 0 has tokens 0..5 past 2 tokens");
    let mut l = ada();
    l.chunks = Some(PhraseChunks {
        spans: spans(&[(0, 3)]),
        tokens: vec![IxRange::new(1, 3)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("chunks", &["NP"]),
    });
    l.refused("opennlp:chunks item 0 has tokens 1..3 past 2 tokens");
    let mut l = ada();
    l.tokens.sentence = Some(vec![Ix::new(0), Ix::new(4)]);
    l.refused("token 1 names sentence 4 past 1 sentences");
}

#[test]
fn hand_built_input_is_validated_for_training_too() {
    use turbo_coref::train::{GoldMention, TrainingDocument, pairs};
    let l = Layers::new("A", &[(0, 1)], (0, 5));
    let doc = TrainingDocument { input: l.input(), gold: vec![GoldMention { span: Span::new(0, 1), chain: 0 }] };
    assert!(pairs(&doc, &CorefAnnotator::default()).is_err());
}
