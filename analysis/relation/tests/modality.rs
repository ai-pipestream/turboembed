//! `ModalityAnnotator` over hand-parsed sentences: negation scope cut at
//! clause boundaries, conditionals, deontic composition, the `may` rule,
//! the window fallback, span lookup, provenance and the graph round trip.

mod modality_support;

use modality_support::{at, modality, parsed, tag_at, token, tokens, tokens_only};
use turbo_analysis::{Analyzer, Annotator, DocumentGraph, Key, export, import};
use turbo_analysis_layers::modality::{DeonticKind as D, EpistemicStatus as E, Modality, Polarity as P};
use turbo_analysis_layers::{Dependencies, Tokens};
use turbo_relation::export::codecs;
use turbo_relation::{ModalityAnnotator, ModalityConfig};

const A: (P, E, D) = (P::Affirmed, E::Asserted, D::None);

fn bought() -> turbo_analysis::Document {
    parsed("Acme did not buy Beta and sold Gamma .", "4:nsubj 4:aux 4:advmod 0:root 4:obj 7:cc 4:conj 7:obj 4:punct")
}

#[test]
fn negation_scope_stops_at_a_conjunct() {
    let d = bought();
    let m = modality(&d);
    assert_eq!(m.len(), 1);
    let (scope, tag) = m.iter().next().unwrap();
    assert_eq!(scope, tokens(&d, 0, 5), "Acme did not buy Beta");
    assert_eq!(tag.cue, Some(token(&d, 2)));
    assert_eq!(tag.confidence, 0.9);
    assert_eq!(at(&m, &d, 3), (P::Negated, E::Asserted, D::None));
    assert_eq!(tag_at(&m, &d, 6), None, "sold is outside the scope");
}

#[test]
fn conditional_scopes_the_antecedent() {
    let d = parsed(
        "If the deal closes , Acme will expand .",
        "4:mark 3:det 4:nsubj 8:advcl 8:punct 8:nsubj 8:aux 0:root 8:punct",
    );
    let m = modality(&d);
    assert_eq!(m.iter().map(|(s, _)| s).collect::<Vec<_>>(), [tokens(&d, 0, 4)]);
    assert_eq!(at(&m, &d, 3), (P::Affirmed, E::Conditional, D::None));
    assert_eq!(at(&m, &d, 7), A);
}

#[test]
fn obligation_and_must_not() {
    let d = parsed("Vendors must encrypt data .", "3:nsubj 3:aux 0:root 3:obj 3:punct");
    let m = modality(&d);
    assert_eq!(at(&m, &d, 2), (P::Affirmed, E::Asserted, D::Obligation));
    assert_eq!(tag_at(&m, &d, 2).unwrap().cue, Some(token(&d, 1)));

    let d = parsed("Vendors must not share customer data .", "4:nsubj 4:aux 4:advmod 0:root 6:compound 4:obj 4:punct");
    let m = modality(&d);
    assert_eq!(m.len(), 1, "both cues share the predicate");
    assert_eq!(at(&m, &d, 3), (P::Negated, E::Asserted, D::Prohibition));

    let d = parsed("Tenants are not required to pay .", "4:nsubj 4:cop 4:advmod 0:root 6:mark 4:xcomp 4:punct");
    assert_eq!(at(&modality(&d), &d, 3), (P::Negated, E::Asserted, D::None), "not required to");
}

#[test]
fn prohibition_reaches_its_adverbial_clause() {
    let d = parsed(
        "Customers are prohibited from reselling tickets .",
        "3:nsubj:pass 3:aux:pass 0:root 5:mark 3:advcl 5:obj 3:punct",
    );
    let m = modality(&d);
    assert_eq!(at(&m, &d, 4), (P::Affirmed, E::Asserted, D::Prohibition));
}

#[test]
fn not_impossible_is_affirmed_and_hedged() {
    let d = parsed("A merger is not impossible .", "2:det 5:nsubj 5:cop 5:advmod 0:root 5:punct");
    assert_eq!(at(&modality(&d), &d, 4), (P::Affirmed, E::Hedged, D::None));
}

#[test]
fn may_is_a_permission_for_a_party_and_a_hedge_otherwise() {
    let d = parsed("Users may cancel at any time .", "3:nsubj 3:aux 0:root 6:case 6:det 3:obl 3:punct");
    let m = modality(&d);
    assert_eq!(at(&m, &d, 2), (P::Affirmed, E::Asserted, D::Permission));
    assert_eq!(tag_at(&m, &d, 2).unwrap().confidence, 0.6);

    let d = parsed("The shares may fall .", "2:det 4:nsubj 4:aux 0:root 4:punct");
    assert_eq!(at(&modality(&d), &d, 3), (P::Affirmed, E::Hedged, D::None));
}

#[test]
fn window_fallback_without_dependencies() {
    let d = tokens_only("Acme did not buy Beta , but sold Gamma .");
    let m = modality(&d);
    let (scope, tag) = m.iter().next().unwrap();
    assert_eq!(scope, tokens(&d, 2, 5), "not buy Beta");
    assert_eq!(tag.confidence, 0.7);
    assert_eq!(at(&m, &d, 3).0, P::Negated);
    assert_eq!(tag_at(&m, &d, 7), None);

    let narrow = ModalityAnnotator::with_config(ModalityConfig { max_window_tokens: 1, ..Default::default() });
    let m = narrow.modality(&d).unwrap();
    assert_eq!(m.iter().next().unwrap().0, tokens(&d, 2, 4), "one token past the cue");
}

#[test]
fn span_lookup_returns_the_innermost_scope() {
    let d = parsed(
        "Acme allegedly said it might not sign .",
        "3:nsubj 3:advmod 0:root 7:nsubj 7:aux 7:advmod 3:ccomp 3:punct",
    );
    let m = modality(&d);
    assert_eq!(m.len(), 2);
    assert_eq!(at(&m, &d, 2), (P::Affirmed, E::Reported, D::None));
    assert_eq!(at(&m, &d, 6), (P::Negated, E::Hedged, D::None));
    assert_eq!(m.tag_for_span(tokens(&d, 0, 8)), None, "no scope covers the period");

    let plain = parsed("Acme bought Beta .", "2:nsubj 0:root 2:obj 2:punct");
    let m = modality(&plain);
    assert!(m.is_empty());
    assert_eq!(tag_at(&m, &plain, 1), None);
}

#[test]
fn config_hash_tracks_the_config() {
    let base = ModalityAnnotator::new().config_hash();
    assert_eq!(base, ModalityAnnotator::new().config_hash());
    for config in [
        ModalityConfig { max_window_tokens: 4, ..Default::default() },
        ModalityConfig { deontic_subjects: vec!["users".into()], ..Default::default() },
    ] {
        assert_ne!(ModalityAnnotator::with_config(config).config_hash(), base);
    }
}

#[test]
fn analyzer_provenance_and_graph_round_trip() {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(ModalityAnnotator::new())
        .build()
        .unwrap();
    let mut batch = vec![bought()];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let d = batch.pop().unwrap();
    let prov = d.provenance(Key::<Modality>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, ModalityAnnotator::ID);
    assert!(prov.reads.contains(&Key::<Dependencies>::DEFAULT.id()));
    let m = d.require(&Key::<Modality>::DEFAULT).unwrap();
    let codecs = codecs();
    let bytes = export(&d, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    let restored = back.require(&Key::<Modality>::DEFAULT).unwrap();
    assert_eq!(restored, m);
    assert_eq!(restored.tag_for_span(token(&back, 3)), m.tag_for_span(token(&d, 3)));
}

#[test]
fn factive_presupposes_its_complement_only() {
    let d =
        parsed("Regulators confirmed that the merger closed .", "2:nsubj 0:root 6:mark 5:det 6:nsubj 2:ccomp 2:punct");
    let m = modality(&d);
    assert_eq!(at(&m, &d, 5), (P::Affirmed, E::Presupposed, D::None));
    assert_eq!(at(&m, &d, 1), A, "the factive verb itself is asserted");
    assert_eq!(tag_at(&m, &d, 5).unwrap().cue, Some(tokens(&d, 1, 3)));
}
