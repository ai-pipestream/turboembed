//! `DiscourseAnnotator` over hand-parsed sentences, with and without
//! dependencies, and the `Edus`/`Discourse` graph round trip.

mod discourse_support;

use discourse_support::{doc, run};
use turbo_analysis::{Analyzer, Annotator, DocumentGraph, Key, Layer, export, import};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    Dependencies, Discourse, DiscourseEdgeLabel, DiscourseRel as R, Edus, Nuclearity as N, Sentences, Tokens,
};
use turbo_chunker::DiscourseAnnotator;

const CONCESSION: (&str, &str) =
    ("Although sales fell , profits rose .", "3:mark 3:nsubj 6:advcl 6:punct 6:nsubj 0:root 6:punct");
const CAUSE: (&str, &str) =
    ("Acme raised prices because costs rose .", "2:nsubj 0:root 2:obj 6:mark 6:nsubj 2:advcl 2:punct");
const ROSE: (&str, &str) = ("Sales rose .", "2:nsubj 0:root 2:punct");
const HOWEVER: (&str, &str) = ("However , margins fell .", "4:advmod 4:punct 4:nsubj 0:root 4:punct");
const CONJ: (&str, &str) = ("Acme bought Beta and sold Gamma .", "2:nsubj 0:root 2:obj 5:cc 2:conj 5:obj 2:punct");

#[test]
fn fronted_concession_is_a_satellite_before_its_nucleus() {
    for parsed in [true, false] {
        let (units, edges) = run(&doc(&[CONCESSION], parsed));
        assert_eq!(units, ["Although sales fell ,", "profits rose ."], "parsed {parsed}");
        assert_eq!(edges, [(0, 1, R::Concession, N::SatelliteToNucleus)]);
    }
}

#[test]
fn trailing_cause_is_a_satellite_after_its_nucleus() {
    for parsed in [true, false] {
        let (units, edges) = run(&doc(&[CAUSE], parsed));
        assert_eq!(units, ["Acme raised prices", "because costs rose ."]);
        assert_eq!(edges, [(0, 1, R::CauseResult, N::NucleusToSatellite)]);
    }
}

#[test]
fn sentence_initial_adverbial_links_across_sentences() {
    for parsed in [true, false] {
        let (units, edges) = run(&doc(&[ROSE, HOWEVER], parsed));
        assert_eq!(units, ["Sales rose .", "However , margins fell ."]);
        assert_eq!(edges, [(0, 1, R::Contrast, N::MultiNuclear)]);
    }
}

#[test]
fn verb_conjunct_splits_with_dependencies_only() {
    let (units, edges) = run(&doc(&[CONJ], true));
    assert_eq!(units, ["Acme bought Beta", "and sold Gamma ."]);
    assert_eq!(edges, [(0, 1, R::Joint, N::MultiNuclear)]);
    let (units, edges) = run(&doc(&[CONJ], false));
    assert_eq!(units, ["Acme bought Beta and sold Gamma ."], "no comma, no cut");
    assert!(edges.is_empty());
}

#[test]
fn unmarked_complement_is_an_elaboration() {
    let said = ("Acme said profits rose .", "2:nsubj 0:root 4:nsubj 2:ccomp 2:punct");
    let (units, edges) = run(&doc(&[said], true));
    assert_eq!(units, ["Acme said", "profits rose ."]);
    assert_eq!(edges, [(0, 1, R::Elaboration, N::NucleusToSatellite)]);
}

#[test]
fn labels_round_trip_through_tags() {
    let tags = DiscourseEdgeLabel::tagset();
    assert_eq!(tags.len(), 24);
    for r in R::ALL {
        for n in N::ALL {
            let l = DiscourseEdgeLabel { relation: r, nuclearity: n };
            assert_eq!(DiscourseEdgeLabel::from_tag(l.tag()), Some(l));
            assert_eq!(tags.label(l.tag()), format!("{}/{}", r.name(), n.name()));
        }
    }
}

#[test]
fn analyzer_provenance_and_graph_round_trip() {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(DiscourseAnnotator::new())
        .build()
        .unwrap();
    let mut batch = vec![doc(&[CONCESSION, ROSE, HOWEVER], true)];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let d = batch.pop().unwrap();
    assert_eq!(d.provenance(Key::<Edus>::DEFAULT.id()).unwrap().annotator, DiscourseAnnotator::ID);
    let edus = d.require(&Key::<Edus>::DEFAULT).unwrap();
    let edges = d.require(&Key::<Discourse>::DEFAULT).unwrap();
    assert_eq!(edus.len(), 4);
    assert_eq!(edges.len(), 2);
    let codecs = codecs();
    let bytes = export(&d, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(back.require(&Key::<Edus>::DEFAULT).unwrap(), edus);
    let restored = back.require(&Key::<Discourse>::DEFAULT).unwrap();
    assert_eq!(restored.iter().map(|(_, e)| e).collect::<Vec<_>>(), edges.iter().map(|(_, e)| e).collect::<Vec<_>>());
}

#[test]
fn edus_check_rejects_bad_rows() {
    use turbo_analysis::{Ix, Span};
    let mut ok = Edus::new(turbo_analysis::ViewId(0));
    ok.push(Span::new(0, 3), Ix::new(0));
    ok.push(Span::new(4, 8), Ix::new(1));
    assert!(ok.check().is_ok());
    let mut backwards = Edus::new(turbo_analysis::ViewId(0));
    backwards.push(Span::new(0, 3), Ix::new(1));
    backwards.push(Span::new(4, 8), Ix::new(0));
    assert!(backwards.check().is_err());
    let mut empty = Edus::new(turbo_analysis::ViewId(0));
    empty.push(Span::new(2, 2), Ix::new(0));
    assert!(empty.check().is_err());
}
