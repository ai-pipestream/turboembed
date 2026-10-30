use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_glossary::{Glossary, GlossaryAnnotator, GlossaryMatcher, TokenMatcher, TokenValue};
use turbo_matcher::FuzzyMatchPolicy;

fn policy(distance: u8, phonetic: bool) -> FuzzyMatchPolicy {
    FuzzyMatchPolicy::new(distance, 4, phonetic, 0.1).unwrap()
}

fn glossary(term: &str) -> Arc<Glossary> {
    let mut glossary = Glossary::new();
    glossary.push("entry", term).unwrap();
    Arc::new(glossary)
}

fn analyze(text: &str, out: &mut Vec<String>) {
    out.extend(text.split(' ').map(str::to_owned));
}

fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        if ch == ' ' {
            spans.push(Span::new(start as u32, index as u32));
            start = index + 1;
        }
    }
    spans.push(Span::new(start as u32, text.len() as u32));
    Tokens { spans, sentence: None }
}

fn find(matcher: TokenMatcher, glossary: Arc<Glossary>, text: &str) -> Vec<Span> {
    let mut doc = Document::new(text);
    let tokens = tokens(text);
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let view = doc.get::<View>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let (terms,) =
        GlossaryAnnotator::new(glossary, GlossaryMatcher::Tokens(matcher)).annotate((view, tokens, None)).unwrap();
    terms.check().unwrap();
    assert_eq!(terms.spans.view(), ViewId::SOURCE);
    terms.spans.iter().map(|(_, span)| span).collect()
}

#[test]
fn fuzzy_tokens_recover_transposition_and_phonetic_variants_without_moving_spans() {
    let g = glossary("California");
    let exact = TokenMatcher::new(&g, TokenValue::Text, analyze).unwrap();
    assert!(find(exact, Arc::clone(&g), "Califronia").is_empty());

    let fuzzy = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(1, false)).unwrap();
    let spans = find(fuzzy, Arc::clone(&g), "Califronia");
    assert_eq!(spans, [Span::new(0, 10)]);
    assert_eq!(&"Califronia"[spans[0].range()], "Califronia");

    let g = glossary("Catherine");
    let phonetic = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(0, true)).unwrap();
    let spans = find(phonetic, g, "Kathryn");
    assert_eq!(spans, [Span::new(0, 7)]);
}

#[test]
fn fuzzy_tokens_cover_multiple_words_and_disabled_policy_is_compatible() {
    let g = glossary("New York");
    let exact = TokenMatcher::new(&g, TokenValue::Text, analyze).unwrap();
    let disabled =
        TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, FuzzyMatchPolicy::default()).unwrap();
    assert_eq!(find(exact, Arc::clone(&g), "New York"), find(disabled, Arc::clone(&g), "New York"));

    let fuzzy = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(1, false)).unwrap();
    assert_eq!(find(fuzzy, Arc::clone(&g), "New Yrok"), [Span::new(0, 8)]);

    let base = TokenMatcher::new(&g, TokenValue::Text, analyze).unwrap();
    let same = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, FuzzyMatchPolicy::default()).unwrap();
    let changed = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(1, false)).unwrap();
    let hash = |m| GlossaryAnnotator::new(Arc::clone(&g), GlossaryMatcher::Tokens(m)).config_hash();
    let base_hash = hash(base);
    assert_eq!(base_hash, hash(same));
    assert_ne!(base_hash, hash(changed));
}

#[test]
fn exact_match_wins_over_fuzzy_match_on_the_same_span() {
    let mut g = Glossary::new();
    g.push("cat", "cat").unwrap();
    g.push("bat", "bat").unwrap();
    let matcher = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(1, false)).unwrap();
    let mut out = Vec::new();
    matcher.find("bat", &tokens("bat"), None, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].entry.0, 1);

    let mut g = Glossary::new();
    g.push("Kathryn", "Kathryn").unwrap();
    g.push("Catherine", "Catherine").unwrap();
    let matcher = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy(0, true)).unwrap();
    let mut out = Vec::new();
    matcher.find("Catherine", &tokens("Catherine"), None, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].entry.0, 1);
}

#[test]
fn fuzzy_matching_cannot_edit_across_token_boundaries() {
    let policy = FuzzyMatchPolicy::new(2, 2, false, 0.1).unwrap();
    let mut g = Glossary::new();
    g.push("ab-cd", "ab cd").unwrap();
    let matcher = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy).unwrap();
    let mut out = Vec::new();
    matcher.find("abcd", &tokens("abcd"), None, &mut out).unwrap();
    assert!(out.is_empty());

    let policy = FuzzyMatchPolicy::new(0, 4, true, 0.1).unwrap();
    let mut g = Glossary::new();
    g.push("short-parts", "Ka th").unwrap();
    let matcher = TokenMatcher::new_with_fuzzy_policy(&g, TokenValue::Text, analyze, policy).unwrap();
    let mut out = Vec::new();
    matcher.find("Ca th", &tokens("Ca th"), None, &mut out).unwrap();
    assert!(out.is_empty());
}
