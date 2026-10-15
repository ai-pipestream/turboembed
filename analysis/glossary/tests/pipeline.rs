//! The step in a pipeline: it matches over a normalized view and its
//! tokens, writes spans in that view that map back to the source, and
//! reads stems when a token matcher matches by them.

use std::borrow::Cow;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Bound, Document, Key, Span, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_glossary::{
    Case, Glossary, GlossaryAnnotator, GlossaryMatcher, GlossaryTerms, TextMatcher, TokenMatcher, TokenValue,
};
use turbo_normalize::{CaseFold, NormalizeStep, Whitespace, apply};
use turbo_stem::{Algorithm, PorterStemmer, StemAnnotator, Stemmer};
use turbo_tokenize::WordTokenizer;

fn glossary(entries: &[(&str, &str)]) -> Arc<Glossary> {
    let mut g = Glossary::new();
    for (id, term) in entries {
        g.push(id, term).unwrap();
    }
    Arc::new(g)
}

/// Whitespace collapsed, then lowercased: the view the step reads and
/// what the terms go through when the matcher is built.
fn fold(text: &str) -> String {
    apply(&CaseFold::lowercase(), &apply(&Whitespace::collapse(), text)).into_owned()
}

/// Folds the source into "folded", tokenizes and stems that view, and
/// runs the step over it.
fn run(step: GlossaryAnnotator, text: &str) -> Document {
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "spaced"))
        .step(Bound::new(NormalizeStep::new(CaseFold::lowercase())).read(0, "spaced").write(0, "folded"))
        .step(Bound::new(WordTokenizer::new()).read(0, "folded").write(0, "folded").write(1, "folded"))
        .step(Bound::new(StemAnnotator::new(Algorithm::Porter)).read(0, "folded").read(1, "folded").write(0, "folded"))
        .step(step.bound("terms").read(0, "folded").read(1, "folded").read(2, "folded"))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    let status = analyzer.run(&mut docs);
    assert_eq!(status[0], Ok(()));
    docs.pop().unwrap()
}

/// Each match as its source text, its tokens' text in the view and its
/// entry's id.
fn found(doc: &Document) -> Vec<(String, Vec<String>, String)> {
    let terms = doc.get_key(&Key::<GlossaryTerms>::named("terms")).unwrap();
    let tokens = doc.get_key(&Key::<Tokens>::named("folded")).unwrap();
    let view = doc.get_key(&Key::<View>::named("folded")).unwrap();
    assert_eq!(terms.spans.view(), view.id());
    terms
        .spans
        .iter()
        .enumerate()
        .map(|(i, (_, span))| {
            let source = doc.to_source(view.id(), span);
            let words = terms.tokens[i].iter().map(|t| view.text()[tokens.spans.get(t).range()].to_string()).collect();
            (doc.text(ViewId::SOURCE)[source.range()].to_string(), words, terms.entry_of(i).id.to_string())
        })
        .collect()
}

fn triples(xs: &[(&str, &[&str], &str)]) -> Vec<(String, Vec<String>, String)> {
    xs.iter().map(|(s, w, id)| (s.to_string(), w.iter().map(|w| w.to_string()).collect(), id.to_string())).collect()
}

const TEXT: &str = "  The Hot\u{a0}  DOG, größe and 東京; hot-dogs and HOT DOGS!  ";

#[test]
fn character_matches_in_a_folded_view_map_to_the_source() {
    let g = glossary(&[("c1", "hot dog"), ("c2", "GRÖßE"), ("c3", "東京"), ("c4", "the")]);
    let m = TextMatcher::with_normalizer(&g, Case::Sensitive, |t| Cow::Owned(fold(t))).unwrap();
    let doc = run(GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Text(m)), TEXT);
    assert_eq!(
        found(&doc),
        triples(&[
            ("The", &["the"], "c4"),
            ("Hot\u{a0}  DOG", &["hot", "dog"], "c1"),
            ("größe", &["größe"], "c2"),
            ("東京", &["東", "京"], "c3"),
        ])
    );
}

fn stems(t: &str, out: &mut Vec<String>) {
    out.extend(fold(t).split(' ').map(|w| PorterStemmer.stem(w)));
}

#[test]
fn token_matches_by_stem_cross_punctuation() {
    let g = glossary(&[("c1", "hot dog")]);
    let m = TokenMatcher::new(&g, TokenValue::Values, stems).unwrap();
    let doc = run(GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Tokens(m)), TEXT);
    assert_eq!(
        found(&doc),
        triples(&[
            ("Hot\u{a0}  DOG", &["hot", "dog"], "c1"),
            ("hot-dogs", &["hot", "dogs"], "c1"),
            ("HOT DOGS", &["hot", "dogs"], "c1"),
        ])
    );
}

#[test]
fn a_composite_prefers_the_matcher_listed_first() {
    let g = glossary(&[("c1", "hot dog"), ("c2", "dogs")]);
    let tokens = TokenMatcher::new(&g, TokenValue::Values, stems).unwrap();
    let exact = TextMatcher::with_normalizer(&g, Case::Sensitive, |t| Cow::Owned(fold(t))).unwrap();
    let m = GlossaryMatcher::composite(vec![GlossaryMatcher::Text(exact), GlossaryMatcher::Tokens(tokens)]).unwrap();
    let doc = run(GlossaryAnnotator::new(g.clone(), m), TEXT);
    // The exact "dogs" keeps the stemmed "hot dog" off "hot-dogs" and
    // "HOT DOGS"; the stemmed "dogs" cannot take "DOG", which the exact
    // "hot dog" holds.
    assert_eq!(
        found(&doc),
        triples(&[("Hot\u{a0}  DOG", &["hot", "dog"], "c1"), ("dogs", &["dogs"], "c2"), ("DOGS", &["dogs"], "c2"),])
    );
}

#[test]
fn stems_are_required_when_a_token_matcher_reads_them() {
    let g = glossary(&[("c1", "dog")]);
    let m = TokenMatcher::new(&g, TokenValue::Values, |t, out| out.push(t.to_string())).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .step(Bound::new(WordTokenizer::new()))
        .step(GlossaryAnnotator::new(g, GlossaryMatcher::Tokens(m)).bound("terms"))
        .build()
        .unwrap();
    let mut docs = vec![Document::new("dog")];
    let status = analyzer.run(&mut docs);
    assert!(status[0].as_ref().is_err_and(|e| e.to_string().contains("turbo:stems")), "{status:?}");
}

#[test]
fn an_empty_document_has_no_matches() {
    let g = glossary(&[("c1", "x")]);
    for text in ["", "   "] {
        let m = TextMatcher::new(&g, Case::Insensitive).unwrap();
        let doc = run(GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Text(m)), text);
        assert_eq!(found(&doc), triples(&[]));
    }
}

#[test]
fn a_match_maps_to_whole_source_characters() {
    // "ǅ" lowercases to "ǆ", same length; "İ" lowercases to two
    // characters, so the view is longer than the source there.
    let g = glossary(&[("c1", "i̇stanbul"), ("c2", "ǆ")]);
    let m = TextMatcher::new(&g, Case::Sensitive).unwrap();
    let doc = run(GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Text(m)), "İstanbul ǅ");
    assert_eq!(found(&doc), triples(&[("İstanbul", &["i̇stanbul"], "c1"), ("ǅ", &["ǆ"], "c2")]));
    let terms = doc.get_key(&Key::<GlossaryTerms>::named("terms")).unwrap();
    let view = doc.get_key(&Key::<View>::named("folded")).unwrap();
    let spans: Vec<Span> = terms.spans.iter().map(|(_, s)| doc.to_source(view.id(), s)).collect();
    assert_eq!(spans, [Span::new(0, 9), Span::new(10, 12)]);
}

#[test]
fn the_step_matches_at_token_edges() {
    // UAX #29 has a boundary between the two plus signs, but neither is a
    // token, so the step does not find "c+" in "C++".
    let g = glossary(&[("c1", "c+")]);
    let m = TextMatcher::with_normalizer(&g, Case::Sensitive, |t| Cow::Owned(fold(t))).unwrap();
    let doc = run(GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Text(m)), "C++ and C+");
    assert_eq!(found(&doc), triples(&[("C+", &["c"], "c1")]));
}

#[test]
fn the_configuration_hash_follows_the_patterns() {
    use turbo_analysis::Annotator;
    let g = glossary(&[("c1", "Hot Dog"), ("c2", "dogs")]);
    let text = |case, norm: bool| {
        let m = if norm {
            TextMatcher::with_normalizer(&g, case, |t| Cow::Owned(fold(t))).unwrap()
        } else {
            TextMatcher::new(&g, case).unwrap()
        };
        GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Text(m)).config_hash()
    };
    // The same automaton shape, other patterns.
    assert_ne!(text(Case::Sensitive, false), text(Case::Sensitive, true));
    assert_ne!(text(Case::Sensitive, false), text(Case::Insensitive, false));
    assert_eq!(text(Case::Insensitive, true), text(Case::Insensitive, true));
    let tokens = |analyze: fn(&str, &mut Vec<String>)| {
        let m = TokenMatcher::new(&g, TokenValue::Values, analyze).unwrap();
        GlossaryAnnotator::new(g.clone(), GlossaryMatcher::Tokens(m)).config_hash()
    };
    let words: fn(&str, &mut Vec<String>) = |t, out| out.extend(t.split(' ').map(str::to_string));
    assert_ne!(tokens(stems), tokens(words));
    assert_eq!(tokens(stems), tokens(stems));
}
