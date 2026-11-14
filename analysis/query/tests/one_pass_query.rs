//! Unit tests for unified one-pass query planning (`#2339`):
//! - exact ordinal-preserving per-field BM25 terms counting empty-normalized tokens,
//! - bidirectional `ViewId::SOURCE` <-> child `Invisible` alignment,
//! - non-backtracking multi-word query synonym expansion (`turbo-matcher`),
//! - structured NLP filters (ECB money, SI quantities, temporal ranges/relations,
//!   geo bbox + containment spine, KB-linked entities) and residual query terms.

use std::sync::Arc;

use turbo_analysis::{Key, View, ViewId};
use turbo_analysis_layers::{Currency, Date};
use turbo_chunker::terms::TermForm;
use turbo_entity_linker::{CatalogProvider, EntityLinker, KbEntity, KnowledgeBase};
use turbo_matcher::PhraseSynonymExpander;
use turbo_namefind::{Dictionary, DictionaryNameFinder};
use turbo_numeric::money::EcbRates;
use turbo_query::{
    FieldQuerySpec, FieldTokenizerKind, OnePassQueryAnalyzer, QueryComparisonOp, StructuredFilterPayload,
};

#[test]
fn empty_normalized_tokens_preserve_exact_ordinals_and_bidirectional_offsets() {
    let analyzer = OnePassQueryAnalyzer::builder()
        .add_field(FieldQuerySpec::new("ws_body").with_tokenizer(FieldTokenizerKind::Whitespace))
        .add_field(FieldQuerySpec::new("uax_body").with_tokenizer(FieldTokenizerKind::Word))
        .build()
        .unwrap();

    // Whitespace field: token 0 has intra-word soft hyphen, token 1 is a standalone
    // zero-width space (normalizes to empty!), token 2 is "match".
    let q_ws = "exact\u{00AD}phrase \u{200B} match";
    let (doc, plan) = analyzer.analyze_document(q_ws).unwrap();
    let ws = &plan.fields[0];
    assert_eq!(ws.total_token_ordinals, 3);
    assert_eq!(ws.empty_normalized_ordinals, vec![1]);
    assert_eq!(ws.bm25_terms.len(), 2);
    assert_eq!(ws.bm25_terms[0].term, "exactphras");
    assert_eq!(ws.bm25_terms[0].ordinal, 0);
    assert_eq!(&q_ws[ws.bm25_terms[0].span.range()], "exact\u{00AD}phrase");
    assert_eq!(ws.bm25_terms[1].term, "match");
    assert_eq!(ws.bm25_terms[1].ordinal, 2, "ordinal 1 was empty-normalized so 'match' stays at 2");

    // Verify bidirectional ViewId::SOURCE <-> child view alignment on Document.
    let clean = doc.get_key(&Key::<View>::named("invisible_clean")).expect("invisible_clean view attached");
    assert_ne!(clean.id(), ViewId::SOURCE);
    let child_span = clean.alignment().span_to_child(ws.bm25_terms[0].span);
    assert_eq!(&clean.text()[child_span.range()], "exactphrase");
    assert_eq!(clean.alignment().span_to_parent(child_span), ws.bm25_terms[0].span);

    // UAX #29 WordTokenizer field: emoji token is emitted into Tokens (ordinal 1)
    // and normalized to empty by NormalizedTermIndexer, preserving ordinal 2 on "match".
    let q_word = "exact\u{00AD}phrase \u{1F680} match";
    let plan_word = analyzer.analyze(q_word).unwrap();
    let uax = &plan_word.fields[1];
    assert_eq!(uax.total_token_ordinals, 3);
    assert_eq!(uax.empty_normalized_ordinals, vec![1]);
    assert_eq!(uax.bm25_terms.len(), 2);
    assert_eq!((uax.bm25_terms[0].ordinal, uax.bm25_terms[0].term.as_str()), (0, "exactphras"));
    assert_eq!((uax.bm25_terms[1].ordinal, uax.bm25_terms[1].term.as_str()), (2, "match"));
}

#[test]
fn multi_token_query_synonyms_match_across_stemmed_and_folded_fields() {
    let mut syns = PhraseSynonymExpander::lower();
    syns.add_equivalent_group(&[&["new", "york", "city"], &["nyc"]]).unwrap();
    syns.add_equivalent_group(&[&["machine", "learning"], &["ml"]]).unwrap();
    let syns = Arc::new(syns);

    let analyzer = OnePassQueryAnalyzer::builder()
        .with_synonyms(syns)
        .add_field(FieldQuerySpec::new("title_stem").with_term_form(TermForm::Stem))
        .add_field(FieldQuerySpec::new("code_folded").with_term_form(TermForm::Folded))
        .build()
        .unwrap();

    let q = "New York City machine learning engineers";
    let plan = analyzer.analyze(q).unwrap();

    let title = &plan.fields[0];
    assert_eq!(title.synonyms.len(), 2);
    assert_eq!((title.synonyms[0].start_ordinal, title.synonyms[0].end_ordinal, title.synonyms[0].pos_len), (0, 3, 3));
    assert_eq!(title.synonyms[0].surface, "New York City");
    assert_eq!(title.synonyms[0].terms, vec!["nyc"]);
    assert_eq!((title.synonyms[1].start_ordinal, title.synonyms[1].end_ordinal, title.synonyms[1].pos_len), (3, 5, 2));
    assert_eq!(title.synonyms[1].surface, "machine learning");
    assert_eq!(title.synonyms[1].terms, vec!["ml"]);

    // Reverse direction: "nyc" expands to ["new", "york", "citi"] on stemmed field
    // and ["new", "york", "city"] on folded field.
    let rev = analyzer.analyze("nyc jobs").unwrap();
    assert_eq!(rev.fields[0].synonyms[0].terms, vec!["new", "york", "citi"]);
    assert_eq!(rev.fields[1].synonyms[0].terms, vec!["new", "york", "city"]);
}

#[test]
fn structured_filters_split_residual_terms_with_ecb_si_geo_and_kb_linking() {
    let anchor = Date::new(2026, 7, 14).unwrap();
    let ecb = Arc::new(EcbRates::parse("Date,USD\n2026-07-14,1.25\n").unwrap());
    let eur = Currency::new(*b"EUR").unwrap();

    let dict = Dictionary::read("case-insensitive\nORG\tAlphabet\n").unwrap();
    let dict_finder = DictionaryNameFinder::new(dict).unwrap();
    let kb = KnowledgeBase::from_entities(
        "wikidata",
        vec![KbEntity::new("Q95", "Alphabet Inc.").with_alias("Alphabet", 1.0).with_types(["ORG"]).with_prior(0.95)],
    )
    .unwrap();
    let linker = Arc::new(EntityLinker::new().add_provider(CatalogProvider::new(kb)));

    let analyzer = OnePassQueryAnalyzer::builder()
        .with_reference_date(anchor)
        .with_ecb_rates(ecb, eur, Some(anchor))
        .with_dictionary_finder(dict_finder)
        .with_entity_linker(linker)
        .build()
        .unwrap();

    let q = "cloud startups over $5M in Berlin since 2024";
    let plan = analyzer.analyze(q).unwrap();

    let res_words: Vec<&str> = plan.residual_terms.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(res_words, vec!["cloud", "startup"]);
    assert_eq!(plan.free_text, "cloud startups");

    assert_eq!(plan.filters.len(), 3, "{:?}", plan.filters);

    // 1. Money filter: "over $5M" -> Gt, $5,000,000 USD == 4,000,000 EUR at 1.25 USD/EUR.
    assert_eq!(plan.filters[0].surface, "over $5M");
    assert_eq!(plan.filters[0].comparison, QueryComparisonOp::Gt);
    match &plan.filters[0].payload {
        StructuredFilterPayload::Money { raw, amount_f64, ecb_converted } => {
            assert_eq!(raw.currency.as_str(), "USD");
            assert!((*amount_f64 - 5_000_000.0).abs() < 1e-3);
            let conv = ecb_converted.as_ref().expect("ECB converted to EUR");
            assert_eq!(conv.currency.as_str(), "EUR");
            assert!((conv.amount_f64 - 4_000_000.0).abs() < 1e-3);
        }
        other => panic!("expected Money filter, got {other:?}"),
    }

    // 2. Place filter: "in Berlin" -> DE/Berlin with WGS84 coordinates.
    assert_eq!(plan.filters[1].surface, "in Berlin");
    match &plan.filters[1].payload {
        StructuredFilterPayload::Place { raw, spine_path, .. } => {
            assert_eq!(raw.canonical_name, "Berlin");
            assert!(spine_path.starts_with("DE/") && spine_path.ends_with("Berlin"), "{spine_path}");
            assert!((raw.location.latitude() - 52.52).abs() < 1.0);
        }
        other => panic!("expected Place filter, got {other:?}"),
    }

    // 3. Temporal filter: "since 2024" -> Gte, open end.
    assert_eq!(plan.filters[2].surface, "since 2024");
    assert_eq!(plan.filters[2].comparison, QueryComparisonOp::Gte);
    match &plan.filters[2].payload {
        StructuredFilterPayload::Temporal(t) => {
            let y2024 = Date::new(2024, 1, 1).unwrap().epoch_day() * 86_400;
            assert_eq!(t.start_epoch_seconds, Some(y2024));
            assert_eq!(t.end_epoch_seconds, None);
        }
        other => panic!("expected Temporal filter, got {other:?}"),
    }

    // SI quantity + EntityLinker KB resolution in one pass.
    let q2 = "Alphabet drones under 500 g";
    let plan2 = analyzer.analyze(q2).unwrap();
    let res2: Vec<&str> = plan2.residual_terms.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(res2, vec!["drone"]);
    assert_eq!(plan2.filters.len(), 2, "{:?}", plan2.filters);
    match &plan2.filters[0].payload {
        StructuredFilterPayload::Entity { kb_id, provider_id, canonical_name, .. } => {
            assert_eq!(kb_id.as_deref(), Some("Q95"));
            assert_eq!(provider_id.as_deref(), Some("wikidata"));
            assert_eq!(canonical_name.as_deref(), Some("Alphabet Inc."));
        }
        other => panic!("expected KB-linked Entity filter, got {other:?}"),
    }
    assert_eq!(plan2.filters[1].comparison, QueryComparisonOp::Lt);
    match &plan2.filters[1].payload {
        StructuredFilterPayload::Quantity { si_base_unit, si_value_f64, .. } => {
            assert_eq!(si_base_unit, "kg");
            assert!((*si_value_f64 - 0.5).abs() < 1e-6);
        }
        other => panic!("expected SI Quantity filter, got {other:?}"),
    }
}
