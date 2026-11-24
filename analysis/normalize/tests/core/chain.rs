//! What a chain reports and records, its configuration hash, and edge cases
//! of text and alignment.

use super::documents::{check_document, run_chained};
use super::*;

// 4. Chain bookkeeping

#[test]
fn chain_reports_its_steps() {
    let empty = Chain::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.ids().count(), 0);

    let chain = Chain::new().then(Ligature).then(Spaces).then_dyn(Box::new(Upper));
    assert!(!chain.is_empty());
    assert_eq!(chain.len(), 3);
    assert_eq!(chain.ids().collect::<Vec<_>>(), ["test:ligature", "test:spaces", "test:upper"]);
}

#[test]
fn an_empty_chain_changes_nothing() {
    let (out, a) = normalize(&Chain::new(), "a  b\u{AD}");
    assert_eq!(out, "a  b\u{AD}");
    assert!(a.edits().is_empty());
    assert!(matches!(apply(&Chain::new(), "x  y"), Cow::Borrowed("x  y")));
}

#[test]
fn chain_config_hash_changes_with_the_steps_and_their_order() {
    let ab = Chain::new().then(Ligature).then(Spaces);
    let ab_again = Chain::new().then(Ligature).then(Spaces);
    let ba = Chain::new().then(Spaces).then(Ligature);
    let abc = Chain::new().then(Ligature).then(Spaces).then(SoftHyphen);
    let a = Chain::new().then(Ligature);
    assert_eq!(ab.config_hash(), ab_again.config_hash());
    assert_ne!(ab.config_hash(), ba.config_hash());
    assert_ne!(ab.config_hash(), abc.config_hash());
    assert_ne!(ab.config_hash(), a.config_hash());
    assert_ne!(a.config_hash(), Chain::new().config_hash());
}

#[test]
fn chain_config_hash_follows_a_step_option() {
    let bar = Chain::new().then(AfterPeriod { marker: "|" });
    let star = Chain::new().then(AfterPeriod { marker: "*" });
    assert_ne!(bar.config_hash(), star.config_hash());
    assert_eq!(bar.config_hash(), Chain::new().then(AfterPeriod::default()).config_hash());
    // Nested chains hash by what they hold.
    let nested = Chain::new().then(Chain::new().then(Ligature));
    assert_ne!(nested.config_hash(), Chain::new().then(Ligature).config_hash());
}

#[test]
fn a_step_version_is_part_of_the_chain_hash() {
    // AfterPeriod has version "3" and the others "0": swapping a step for
    // another with the same hash and a different version still differs.
    let a = Chain::new().then(AfterPeriod::default());
    let b = Chain::new().then(Identity);
    assert_ne!(a.config_hash(), b.config_hash());
}

// 5. Provenance

#[test]
fn the_view_of_a_step_records_the_normalizer_id_version_and_config() {
    let normalizer = AfterPeriod { marker: "*" };
    let expected_config = normalizer.config_hash();
    let analyzer = Analyzer::builder().step(Bound::new(NormalizeStep::new(normalizer)).write(0, "v")).build().unwrap();
    let mut docs = vec![Document::new("one. two.")];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    assert_eq!(doc.get_key(&Key::<View>::named("v")).unwrap().text(), "one.* two.*");
    let p = doc.provenance(Key::<View>::named("v").id()).unwrap();
    assert_eq!(p.annotator, "test:after-period");
    assert_eq!(p.version, "3");
    assert_eq!(p.config, expected_config);
    assert_ne!(p.config, 0);
    assert_eq!(p.reads, vec![Key::<View>::DEFAULT.id()]);
}

#[test]
fn a_normalizer_without_options_records_config_zero() {
    let doc = run_chained(&[], "text");
    let p = doc.provenance(Key::<View>::named("v1").id()).unwrap();
    assert_eq!(p.annotator, "turbo:normalize:chain");
    assert_eq!(p.version, "0");
    assert_eq!(p.config, Chain::new().config_hash());

    let analyzer = Analyzer::builder().step(Bound::new(NormalizeStep::new(Upper)).write(0, "v")).build().unwrap();
    let mut docs = vec![Document::new("text")];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let p = docs[0].provenance(Key::<View>::named("v").id()).unwrap();
    assert_eq!((p.annotator, p.config), ("test:upper", 0));
}

#[test]
fn a_chain_step_records_the_hash_of_its_steps() {
    let kinds = [Kind::Ligature, Kind::AfterPeriod];
    let doc = run_chained(&kinds, "\u{FB01}.");
    let p = doc.provenance(Key::<View>::named("v1").id()).unwrap();
    assert_eq!(p.config, chain_of(&kinds).config_hash());
    let reversed = [Kind::AfterPeriod, Kind::Ligature];
    assert_ne!(p.config, chain_of(&reversed).config_hash());
}

#[test]
fn normalize_step_exposes_its_normalizer() {
    let step = NormalizeStep::new(AfterPeriod { marker: "#" });
    assert_eq!(step.normalizer().marker, "#");
}

// 6. Edge cases

#[test]
fn empty_text_stays_empty() {
    for kind in KINDS {
        let (out, a) = normalize(&chain_of(&[kind]), "");
        assert_eq!(out, "");
        assert!(a.edits().is_empty(), "{kind:?}");
        check_chain(&[kind, kind, kind], "");
        check_document(&[kind, kind], "");
    }
    // A step that replaces everything still writes its text for no input.
    let (out, a) = normalize(&ReplaceAll, "");
    assert_eq!(out, "<all>");
    assert_eq!(a.edits().len(), 1);
    check_chain(&[Kind::ReplaceAll, Kind::Upper], "");
}

#[test]
fn text_without_triggers_has_no_edits_through_any_chain() {
    let text = "plain \u{E9}\u{20AC}\u{1F600} text";
    // Upper changes the lowercase ASCII, so leave it out.
    let kinds = [Kind::Ligature, Kind::Spaces, Kind::SoftHyphen, Kind::AbToX, Kind::Identity];
    let (out, a) = normalize(&chain_of(&kinds), text);
    assert_eq!(out, text);
    assert!(a.edits().is_empty());
    assert!(matches!(apply(&chain_of(&kinds), text), Cow::Borrowed(_)));
}

#[test]
fn an_edit_that_covers_the_whole_text() {
    for text in ["ab", "\u{FB01}", "\u{AD}", "   ", "\u{1F600}", "a b"] {
        let (out, a) = normalize(&ReplaceAll, text);
        assert_eq!(out, "<all>");
        assert_eq!(a.edits().len(), 1);
        assert_eq!(a.edits()[0].parent, Span::new(0, text.len() as u32));
        assert_eq!(a.edits()[0].child, Span::new(0, 5));
    }
    // Two stages that each cover everything merge into one edit.
    let chain = Chain::new().then(DeleteAll).then(ReplaceAll);
    let (out, a) = normalize(&chain, "some \u{E9} text");
    assert_eq!(out, "<all>");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(a.span_to_parent(Span::new(1, 2)), Span::new(0, "some \u{E9} text".len() as u32));
}

#[test]
fn a_normalizer_that_deletes_everything() {
    let text = "co\u{AD}op \u{FB01}\u{1F600}";
    let chain = Chain::new().then(Ligature).then(DeleteAll).then(Upper);
    let (out, a) = normalize(&chain, text);
    assert_eq!(out, "");
    assert_eq!(a.edits().len(), 1);
    assert_eq!(a.edits()[0].parent, Span::new(0, text.len() as u32));
    assert_eq!(a.edits()[0].child, Span::new(0, 0));
    assert_eq!(a.span_to_child(Span::new(2, 4)), Span::new(0, 0));
    assert!(matches!(apply(&chain, text), Cow::Owned(s) if s.is_empty()));
    check_chain(&[Kind::Ligature, Kind::DeleteAll, Kind::Upper], text);
    check_document(&[Kind::Ligature, Kind::DeleteAll, Kind::Upper], text);
}

#[test]
fn multi_byte_characters_beside_every_edit_boundary() {
    // Each trigger gets a 2, 3 and 4 byte character on both sides, and is
    // also placed at the very start and end of the text.
    let neighbours = ["\u{E9}", "\u{20AC}", "\u{1F600}", ""];
    let triggers = ["\u{FB01}", "  ", "\u{AD}", ".", "ab", "q", "  \u{AD}  "];
    for left in neighbours {
        for right in neighbours {
            for trigger in triggers {
                let text = format!("{left}{trigger}{right}");
                for a in KINDS {
                    check_chain(&[a], &text);
                    for b in KINDS {
                        check_chain(&[a, b], &text);
                    }
                }
                check_chain(&[Kind::Ligature, Kind::Spaces, Kind::SoftHyphen], &text);
                check_chain(&[Kind::SoftHyphen, Kind::Spaces, Kind::AbToX], &text);
                check_chain(&[Kind::AbToX, Kind::AfterPeriod, Kind::Upper], &text);
                check_document(&[Kind::SoftHyphen, Kind::Spaces], &text);
            }
        }
    }
}

#[test]
fn a_chain_can_be_reused_on_many_texts() {
    let chain = chain_of(&[Kind::Ligature, Kind::Spaces, Kind::SoftHyphen]);
    for &text in INPUTS {
        let first = normalize(&chain, text);
        let second = normalize(&chain, text);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
    }
}

// Known problem in turbo-analysis, outside this crate

#[test]
fn alignment_maps_an_empty_source_span_at_an_insertion() {
    let (_, a) = normalize(&AfterPeriod::default(), "a.b");
    // The empty span at byte 2 sits where "|" was inserted. Its start maps
    // to after the insertion and its end to before it, which is a span
    // that ends before it starts.
    let mapped = a.span_to_child(Span::new(2, 2));
    assert!(mapped.start <= mapped.end);
}

#[test]
fn alignment_maps_an_empty_view_span_at_a_deletion() {
    let (_, a) = normalize(&SoftHyphen, "a\u{AD}b");
    // The empty span at byte 1 of "ab" sits where the soft hyphen was.
    let mapped = a.span_to_parent(Span::new(1, 1));
    assert!(mapped.start <= mapped.end);
}
