use std::sync::Arc;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::{TokenGraph, TokenGraphBuilder, Tokens};
use turbo_glossary::{Case, Glossary, GlossaryAnnotator, GlossaryLatticeAnnotator, GlossaryMatcher, TextMatcher};
use turbo_matcher::{Input, PhraseSynonymExpander};
use turbo_tokenize::WordTokenizer;

#[test]
fn glossary_and_phrase_matcher_multi_token_synonyms_preserve_phrase_continuity() {
    let mut glossary = Glossary::new();
    glossary.push("NYC", "New York City").unwrap();
    glossary.push("NYC", "Big Apple").unwrap();
    glossary.push("NYC", "NYC").unwrap();
    let glossary = Arc::new(glossary);
    let matcher = GlossaryMatcher::Text(TextMatcher::new(&glossary, Case::Insensitive).unwrap());

    let text = "flights to New York City depart daily";
    let analyzer = Analyzer::builder()
        .add(WordTokenizer::new())
        .add(GlossaryAnnotator::new(Arc::clone(&glossary), matcher))
        .add(GlossaryLatticeAnnotator::new().with_lowercase(true))
        .build()
        .unwrap();

    let mut doc = Document::new(text);
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();
    let graph = doc.get::<TokenGraph>().unwrap();

    assert_eq!(graph.matches_phrase(&["flights", "to", "New", "York", "City", "depart", "daily"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["flights", "to", "big", "apple", "depart", "daily"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["flights", "to", "nyc", "depart", "daily"]).len(), 1);
    assert!(graph.matches_phrase(&["flights", "depart"]).is_empty());

    // PhraseSynonymExpander over reverse 1-token -> 3-token expansion ("nyc" -> "new york city").
    let mut syn = PhraseSynonymExpander::lower();
    syn.add_equivalent_group(&[&["nyc"], &["new", "york", "city"], &["big", "apple"]]).unwrap();

    let mut doc2 = Document::new("visit NYC tomorrow");
    Analyzer::builder()
        .add(WordTokenizer::new())
        .build()
        .unwrap()
        .run(std::slice::from_mut(&mut doc2))
        .remove(0)
        .unwrap();
    let tokens = doc2.get::<Tokens>().unwrap();
    let input = Input::new(doc2.text(tokens.spans.view()), tokens).unwrap();
    let mut tb = TokenGraphBuilder::from_tokens(doc2.text(tokens.spans.view()), tokens, None).unwrap();
    syn.emit(&input, tokens, &mut tb).unwrap();
    let g2 = tb.finish().unwrap();

    assert_eq!(g2.matches_phrase(&["visit", "NYC", "tomorrow"]).len(), 1);
    assert_eq!(g2.matches_phrase(&["visit", "new", "york", "city", "tomorrow"]).len(), 1);
    assert_eq!(g2.matches_phrase(&["visit", "big", "apple", "tomorrow"]).len(), 1);
}
