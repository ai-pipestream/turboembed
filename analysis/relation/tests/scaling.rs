//! Annotation cost must grow with the work, not with its square. Each
//! test runs a document and one four times as large and bounds the time
//! ratio: linear work gives about 4, a quadratic pass about 16, so the
//! bound of 8 fails a quadratic pass with room for timing noise. Each
//! size keeps its fastest of three runs.

use std::time::{Duration, Instant};

use turbo_analysis::{Annotator, Document, Edges, Ix, IxRange, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::{RelationAnnotator, RelationPattern, Relations};

/// Layers over `text`: one token per `(start, end, head, label)`, one
/// sentence per range, and one entity per listed token.
struct Input {
    doc: Document,
    sentences: Sentences,
    tokens: Tokens,
    deps: Dependencies,
    entities: Entities,
}

fn input(
    text: &str,
    sentences: &[(u32, u32)],
    tokens: &[(u32, u32, Option<u32>, u16)],
    entity_tokens: &[u32],
) -> Input {
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in sentences {
        sentence_spans.push(Span::new(a, b));
    }
    let mut token_spans = Spans::new(ViewId::SOURCE);
    let mut sentence = Vec::new();
    for &(a, b, _, _) in tokens {
        token_spans.push(Span::new(a, b));
        sentence.push(Ix::new(sentences.partition_point(|&(s, _)| s <= a) as u32 - 1));
    }
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    for &t in entity_tokens {
        let (a, b, _, _) = tokens[t as usize];
        entity_spans.push(Span::new(a, b));
    }
    let n = entity_tokens.len();
    Input {
        doc: Document::new(text),
        sentences: Sentences { spans: sentence_spans },
        tokens: Tokens { spans: token_spans, sentence: Some(sentence) },
        deps: Dependencies {
            head: tokens.iter().map(|t| t.2.map(Ix::new)).collect(),
            relation: tokens.iter().map(|t| Tag(t.3)).collect(),
            relations: TagSet::new("dependency", &["root", "conj", "nsubj", "obj"]),
        },
        entities: Entities {
            spans: entity_spans,
            tokens: entity_tokens.iter().map(|&t| IxRange::new(t, t + 1)).collect(),
            kind: vec![Tag(0); n],
            kinds: TagSet::new("entity", &["item"]),
            score: vec![1.0; n],
        },
    }
}

fn run(input: &Input, path: &str) -> (Relations, Edges<Entities, Entities>, Duration) {
    let annotator: RelationAnnotator = RelationAnnotator::new(vec![RelationPattern::new("r", path).unwrap()]).unwrap();
    let view = input.doc.get::<View>().unwrap();
    let mut best = Duration::MAX;
    let mut out = None;
    for _ in 0..3 {
        let start = Instant::now();
        let layers = annotator
            .annotate((view, &input.sentences, &input.tokens, &input.tokens, &input.deps, &input.entities, None, None))
            .unwrap();
        best = best.min(start.elapsed());
        out = Some(layers);
    }
    let (relations, edges) = out.unwrap();
    (relations, edges, best)
}

fn assert_linear(small: Duration, large: Duration) {
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(1e-9);
    assert!(ratio < 8.0, "four times the work took {ratio:.1} times as long ({small:?} then {large:?})");
}

/// One head with `n` `conj` dependents, all entities: a sibling pattern
/// matches all n(n-1) ordered pairs.
fn coordination(n: u32) -> Input {
    let text: String = std::iter::once("h").chain(std::iter::repeat_n(" e", n as usize)).collect();
    let mut tokens = vec![(0, 1, None, 0)];
    tokens.extend((0..n).map(|i| (2 + 2 * i, 3 + 2 * i, Some(0), 1)));
    input(&text, &[(0, text.len() as u32)], &tokens, &(1..=n).collect::<Vec<_>>())
}

#[test]
fn edges_grow_linearly_with_mentions() {
    let (small, large) = (coordination(150), coordination(300));
    let (_, _, small_time) = run(&small, "<conj >conj");
    let (relations, edges, large_time) = run(&large, "<conj >conj");
    assert_eq!(relations.subject.len(), 300 * 299);
    assert_eq!(edges.from, relations.subject);
    assert_eq!(edges.to, relations.object);
    assert_linear(small_time, large_time);
}

/// `n` sentences "a b c" where b heads a (`nsubj`) and c (`obj`), with
/// entities on a and c: one match per sentence.
fn sentences(n: u32) -> Input {
    let text = "a b c. ".repeat(n as usize);
    let (mut spans, mut tokens, mut entity_tokens) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (at, t) = (7 * i, 3 * i);
        spans.push((at, at + 6));
        tokens.extend([(at, at + 1, Some(t + 1), 2), (at + 2, at + 3, None, 0), (at + 4, at + 5, Some(t + 1), 3)]);
        entity_tokens.extend([t, t + 2]);
    }
    input(&text, &spans, &tokens, &entity_tokens)
}

#[test]
fn pairs_and_heads_grow_linearly_with_the_document() {
    // Sized so the small run lasts tens of milliseconds even in a
    // release build; at millisecond scale a loaded machine can
    // preempt only the large run and fake a quadratic ratio.
    let (small, large) = (sentences(16_000), sentences(64_000));
    let (_, _, small_time) = run(&small, "<nsubj >obj");
    let (relations, _, large_time) = run(&large, "<nsubj >obj");
    assert_eq!(relations.subject.len(), 64_000);
    assert!(relations.subject.iter().zip(&relations.object).all(|(s, o)| s.get() % 2 == 0 && o.get() == s.get() + 1));
    assert_linear(small_time, large_time);
}
