//! Pattern and mention validation, mirroring the reference
//! `RelationPatternTest` and the mention half of
//! `RelationAnnotatorTest`.

use turbo_analysis::Ix;
use turbo_relation::{RelationMention, RelationPattern};

#[test]
fn steps_split_on_project_whitespace() {
    for sep in [' ', '\t', '\n', '\u{A0}', '\u{2007}', '\u{202F}', '\u{3000}', '\u{85}'] {
        let p = RelationPattern::new("r", format!("<nsubj{sep}>obj")).unwrap_or_else(|e| panic!("{sep:?}: {e}"));
        assert_eq!(p.step_strings(), ["<nsubj", ">obj"], "separator {sep:?}");
    }
    // File and unit separators are not project whitespace: the
    // whole string is one (invalid) step.
    for sep in ['\u{1C}', '\u{1F}'] {
        assert!(RelationPattern::new("r", format!("<nsubj{sep}>obj")).is_err(), "separator {sep:?} split");
    }
    // Runs of separators produce no empty steps.
    let p = RelationPattern::new("r", "  <nsubj   >obj ").unwrap();
    assert_eq!(p.steps().len(), 2);
    assert!(p.steps()[0].up && !p.steps()[1].up);
    assert_eq!(p.steps()[0].label, "nsubj");
}

#[test]
fn malformed_paths_are_rejected() {
    for path in ["", "   ", "<", ">", "nsubj", "<nsubj>", "<n< subj", ">obj <nsubj", "<a >b <c"] {
        assert!(RelationPattern::new("r", path).is_err(), "accepted {path:?}");
    }
    let err = RelationPattern::new("r", ">obj <nsubj").unwrap_err();
    assert!(err.contains(">obj <nsubj"), "{err}");
    assert!(RelationPattern::new("", "<nsubj").is_err());
    assert!(RelationPattern::new(" \u{A0}", "<nsubj").is_err(), "a no-break-space type is blank");
}

#[test]
fn trigger_must_be_a_single_jdk_lowercased_token() {
    assert!(RelationPattern::with_trigger("r", "<a", "").is_err());
    assert!(RelationPattern::with_trigger("r", "<a", "two words").is_err());
    assert!(RelationPattern::with_trigger("r", "<a", "Founded").is_err());
    assert!(RelationPattern::with_trigger("r", "<a", "founded").is_ok());
    // The project mapping is the simple per-character one: U+0130
    // lowers to a plain i, so the dotted-capital spelling is not
    // lowercased, "istanbul" is, and the JDK string-lowercased
    // spelling (i + combining dot) is accepted as a trigger even
    // though no pivot form ever lowers to it.
    assert!(RelationPattern::with_trigger("r", "<a", "\u{130}stanbul").is_err());
    assert!(RelationPattern::with_trigger("r", "<a", "istanbul").is_ok());
    assert!(RelationPattern::with_trigger("r", "<a", "i\u{307}stanbul").is_ok());
    assert!(RelationPattern::with_trigger("r", "<a", "stra\u{DF}e").is_ok());
    assert!(RelationPattern::with_trigger("r", "<a", "STRASSE").is_err());
    // Supplementary-plane letters follow the same mapping.
    assert!(RelationPattern::with_trigger("r", "<a", "\u{10400}").is_err());
    assert!(RelationPattern::with_trigger("r", "<a", "\u{10428}").is_ok());
}

#[test]
fn extension_constraints_are_validated() {
    let p = RelationPattern::new("r", "<a").unwrap();
    assert!(p.clone().subject_kind(" ").is_err());
    assert!(p.clone().object_kind("").is_err());
    assert!(p.clone().pivot_pos("  ").is_err());
    assert!(p.clone().pivot_lemma("").is_err());
    assert!(p.clone().confidence(-0.1).is_err());
    assert!(p.clone().confidence(f64::NAN).is_err());
    assert!(p.clone().confidence(1.1).is_err());
    let p = p.subject_kind("organization").unwrap().confidence(0.75).unwrap();
    assert_eq!(p.subject_kind.as_deref(), Some("organization"));
    assert_eq!(p.confidence, 0.75);
}

#[test]
fn mention_validation() {
    assert!(RelationMention::new(" ", Ix::new(0), Ix::new(1)).is_err());
    assert!(RelationMention::new("r", Ix::new(1), Ix::new(1)).is_err());
    let m = RelationMention::new("acquisition", Ix::new(0), Ix::new(1)).unwrap();
    assert_eq!((m.subject.get(), m.object.get()), (0, 1));
}
