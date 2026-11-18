use turbo_abbrev::{AbbreviationAnnotator, AbbreviationLatticeAnnotator};
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, TokenGraph};
use turbo_tokenize::WordTokenizer;

#[test]
fn abbreviation_lattice_expands_both_definition_and_later_mentions_with_exact_spans() {
    let text = "The Federal Deposit Insurance Corporation (FDIC) insured banks. Later the FDIC intervened.";
    let s1_end = text.find('.').unwrap() as u32 + 1;
    let s2_start = s1_end + 1;
    let mut doc = Document::new(text);
    let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, s1_end));
    sent_spans.push(Span::new(s2_start, text.len() as u32));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(WordTokenizer::new())
        .add(AbbreviationAnnotator)
        .add(AbbreviationLatticeAnnotator::new().with_lowercase(true))
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();

    let graph = doc.get::<TokenGraph>().unwrap();

    // Sentence 1: "The Federal Deposit Insurance Corporation" also carries spanning "fdic".
    assert!(!graph.matches_phrase(&["the", "fdic"]).is_empty());
    assert!(!graph.matches_phrase(&["the", "federal", "deposit", "insurance", "corporation"]).is_empty());

    // Sentence 2: "Later the FDIC intervened" matches BOTH "the fdic intervened"
    // AND "the federal deposit insurance corporation intervened" with slop=0!
    assert_eq!(graph.matches_phrase(&["later", "the", "fdic", "intervened"]).len(), 1);
    let expanded =
        graph.matches_phrase(&["later", "the", "federal", "deposit", "insurance", "corporation", "intervened"]);
    assert_eq!(expanded.len(), 1);

    // Every expanded word over the mention "FDIC" maps back to the exact "FDIC" bytes in Sentence 2!
    let mention_start = text.rfind("FDIC").unwrap() as u32;
    let mention_span = Span::new(mention_start, mention_start + 4);
    for &ix in &expanded[0][2..6] {
        let arc = graph.get(ix);
        assert_eq!(arc.span, mention_span);
        assert_eq!(&text[arc.span.range()], "FDIC");
    }
}
