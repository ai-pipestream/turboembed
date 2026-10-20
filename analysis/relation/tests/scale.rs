//! A long document of independent sentences: head finding and the
//! pair loop must stay linear in the document size, not quadratic.

use turbo_analysis::{Annotator, Document, Ix, IxRange, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::{RelationAnnotator, RelationPattern};

#[test]
fn many_sentences_extract_linearly() {
    let n = 8000usize;
    let text = "a b c. ".repeat(n);
    let doc = Document::new(text.as_str());
    let mut token_spans = Spans::new(ViewId::SOURCE);
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    let (mut heads, mut relations, mut token_sent) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..n as u32 {
        let base = 7 * s;
        for off in [0, 2, 4] {
            token_spans.push(Span::new(base + off, base + off + 1));
            token_sent.push(Ix::new(s));
        }
        entity_spans.push(Span::new(base, base + 1));
        entity_spans.push(Span::new(base + 4, base + 5));
        sentence_spans.push(Span::new(base, base + 7));
        let root = Ix::new(3 * s + 1);
        heads.extend([Some(root), None, Some(root)]);
        relations.extend([Tag(1), Tag(0), Tag(2)]);
    }
    let sentences = Sentences { spans: sentence_spans };
    let tokens = Tokens { spans: token_spans, sentence: Some(token_sent) };
    let deps = Dependencies {
        head: heads,
        relation: relations,
        relations: TagSet::new("dependency", &["root", "nsubj", "obj"]),
    };
    let entities = Entities {
        spans: entity_spans,
        tokens: (0..2 * n as u32)
            .map(|e| IxRange::new(3 * (e / 2) + 2 * (e % 2), 3 * (e / 2) + 2 * (e % 2) + 1))
            .collect(),
        kind: vec![Tag(0); 2 * n],
        kinds: TagSet::new("entity", &["item"]),
        score: vec![1.0; 2 * n],
    };
    let annotator: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("acts_on", "<nsubj >obj").unwrap()]).unwrap();
    let view = doc.get::<View>().unwrap();
    let start = std::time::Instant::now();
    let (found, edges) =
        annotator.annotate((view, &sentences, &tokens, &tokens, &deps, &entities, None, None)).unwrap();
    let _elapsed = start.elapsed();
    assert_eq!(found.subject.len(), n);
    assert_eq!(edges.from.len(), n);
    // The quadratic implementation took over a second here in
    // release builds; the linear one takes single-digit
    // milliseconds. Debug builds only check the counts.
    #[cfg(not(debug_assertions))]
    assert!(_elapsed < std::time::Duration::from_millis(500), "extraction over {n} sentences took {_elapsed:?}");
}
