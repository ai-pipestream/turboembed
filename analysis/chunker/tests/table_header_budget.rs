mod table_support;
use table_support::*;
use turbo_analysis::{Annotator, Ix, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Boundaries, Cut, Sentences, Structure, Subwords, Tables};
use turbo_chunker::{Budget, Chunker, Lengths};

#[test]
fn enclosing_block_fit_counts_the_complete_row_syntax() {
    let text = table(2);
    let doc = prepared(&text);
    let tables = doc.get::<Tables>().unwrap();
    // Valid cell-aligned sentences omit the surrounding grid syntax.
    // The detector and subword tokenizer still run on the complete source.
    let mut spans = Spans::new(ViewId::SOURCE);
    for (_, span) in tables.spans.iter() {
        spans.push(span);
    }
    let sentences = Sentences { spans };
    sentences.check().unwrap();
    let subwords = doc.get::<Subwords>().unwrap();
    let max = Lengths::new(&sentences.spans, &subwords.spans).run(0, sentences.spans.len());
    assert!(max < subwords.spans.len() as u32);
    let bounds =
        Boundaries { cost: vec![0; sentences.spans.len()], reason: vec![Cut::Sentence; sentences.spans.len()] };
    let mut quote = Spans::new(ViewId::SOURCE);
    quote.push(Span::new(0, text.len() as u32));
    let structure = Structure { spans: quote, kind: vec![Block::Quote], parent: vec![None], depth: vec![0] };
    structure.check().unwrap();
    let (chunks,) = Chunker { budget: Budget { max, target: max, size_weight: 1000, overhead: 0 } }
        .annotate((&sentences, subwords, &bounds, Some(&structure), None, Some(tables)))
        .unwrap();
    let leaves: Vec<_> = chunks.level.iter().enumerate().filter(|&(_, &level)| level == 0).map(|(i, _)| i).collect();
    assert!(leaves.len() >= 2);
    for &i in &leaves[1..] {
        assert_eq!(chunks.cut[i], Cut::Inside(Block::Quote), "the complete quote exceeds the budget");
    }
    for &i in &leaves {
        let span = chunks.spans.get(Ix::new(i as u32));
        assert_eq!(chunks.model_tokens[i], subwords.spans.overlapping(span).len() as u32);
    }
}
