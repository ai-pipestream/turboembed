//! Relation extraction rules and the mentions they produce: the
//! Rust form of the reference `RelationPattern` and
//! `RelationMention` records, with the same construction
//! validation (messages included where they carry information).
//!
//! A pattern's path is a whitespace-separated sequence of steps,
//! split on the project whitespace predicate (Unicode
//! `White_Space`, as `char::is_whitespace` reports it). A step
//! `<label` walks up from the subject's head token over an arc
//! with that relation label; a step `>label` walks down toward the
//! object's head token. All up steps precede all down steps. The
//! optional trigger constrains the pivot token (the highest token
//! on the path): its form, lowercased with the JDK's
//! `String.toLowerCase(Locale.ROOT)` mapping, must equal the
//! trigger, which must therefore already be in that form.

use turbo_analysis::Ix;
use turbo_analysis_layers::Entities;

/// One path step: a direction and the dependency relation label of
/// the arc it walks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// True for an up step (`<label`, subject toward the pivot),
    /// false for a down step (`>label`, pivot toward the object).
    pub up: bool,
    pub label: String,
}

impl Step {
    /// The step as it appears in a pattern path, marker first.
    pub fn as_str(&self) -> String {
        format!("{}{}", if self.up { '<' } else { '>' }, self.label)
    }
}

fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// The project `StringUtil.toLowerCase`: the simple
/// per-character lowercase mapping (not the JDK's full string
/// mapping, so U+0130 lowers to a plain `i`).
fn java_lowercase(s: &str) -> String {
    s.chars().map(turbo_unicode::java::to_lower_char).collect()
}

fn split_steps(path: &str) -> Vec<&str> {
    path.split(char::is_whitespace).filter(|s| !s.is_empty()).collect()
}

fn valid_step(step: &str) -> bool {
    let mut chars = step.chars();
    match chars.next() {
        Some('<' | '>') => {}
        _ => return false,
    }
    // At least the marker and one label character, and no second
    // marker inside the label.
    let mut label_chars = 0;
    for c in chars {
        if c == '<' || c == '>' {
            return false;
        }
        label_chars += 1;
    }
    label_chars > 0
}

/// An extraction rule: a dependency path shape, the relation type
/// to emit, and optional constraints. The path and trigger follow
/// the reference validation exactly; the entity-kind, pivot
/// part-of-speech, pivot lemma and confidence fields are this
/// crate's extensions and are unrestricted by default.
#[derive(Clone, Debug, PartialEq)]
pub struct RelationPattern {
    /// The relation type to emit, never blank.
    pub relation_type: String,
    /// The path as given; [`RelationPattern::steps`] parses it.
    pub path: String,
    /// The required pivot form (JDK-lowercased), or any pivot.
    pub trigger: Option<String>,
    /// Only subjects whose entity kind label is this, when set.
    pub subject_kind: Option<String>,
    /// Only objects whose entity kind label is this, when set.
    pub object_kind: Option<String>,
    /// Only pivots whose part-of-speech tag is this, when set.
    /// Needs the anchor's POS layer at extraction time.
    pub pivot_pos: Option<String>,
    /// Only pivots whose lemma is exactly this, when set. Needs
    /// the `Lemmas` layer at extraction time.
    pub pivot_lemma: Option<String>,
    /// The confidence written on every match, in `[0, 1]`.
    pub confidence: f64,
    steps: Vec<Step>,
}

impl RelationPattern {
    /// A rule with no trigger and no extension constraints,
    /// validated as the reference record validates.
    pub fn new(relation_type: impl Into<String>, path: impl Into<String>) -> Result<RelationPattern, String> {
        RelationPattern::build(relation_type.into(), path.into(), None)
    }

    /// A rule whose pivot form must be `trigger`.
    pub fn with_trigger(
        relation_type: impl Into<String>,
        path: impl Into<String>,
        trigger: impl Into<String>,
    ) -> Result<RelationPattern, String> {
        RelationPattern::build(relation_type.into(), path.into(), Some(trigger.into()))
    }

    fn build(relation_type: String, path: String, trigger: Option<String>) -> Result<RelationPattern, String> {
        if is_blank(&relation_type) {
            return Err("type must not be null or blank".into());
        }
        if is_blank(&path) {
            return Err("path must not be null or blank".into());
        }
        if let Some(trigger) = &trigger {
            if trigger.is_empty() {
                return Err("trigger must not be blank".into());
            }
            if trigger.chars().any(char::is_whitespace) {
                return Err(format!(
                    "trigger must not contain whitespace, since it is matched against a single token: {trigger}"
                ));
            }
            if java_lowercase(trigger) != *trigger {
                return Err(format!("trigger must be lowercased, but was: {trigger}"));
            }
        }
        let mut steps = Vec::new();
        let mut down = false;
        for step in split_steps(&path) {
            if !valid_step(step) {
                return Err(format!("not a valid path step: {step}"));
            }
            let up = step.starts_with('<');
            if !up {
                down = true;
            } else if down {
                return Err(format!("up steps must come before down steps: {path}"));
            }
            steps.push(Step { up, label: step[1..].to_owned() });
        }
        Ok(RelationPattern {
            relation_type,
            path,
            trigger,
            subject_kind: None,
            object_kind: None,
            pivot_pos: None,
            pivot_lemma: None,
            confidence: 1.0,
            steps,
        })
    }

    /// Restricts the subject entity kind label.
    pub fn subject_kind(mut self, kind: impl Into<String>) -> Result<Self, String> {
        self.subject_kind = Some(non_blank("subject kind", kind.into())?);
        Ok(self)
    }

    /// Restricts the object entity kind label.
    pub fn object_kind(mut self, kind: impl Into<String>) -> Result<Self, String> {
        self.object_kind = Some(non_blank("object kind", kind.into())?);
        Ok(self)
    }

    /// Restricts the pivot token's part-of-speech tag.
    pub fn pivot_pos(mut self, pos: impl Into<String>) -> Result<Self, String> {
        self.pivot_pos = Some(non_blank("pivot part of speech", pos.into())?);
        Ok(self)
    }

    /// Restricts the pivot token's lemma (exact match).
    pub fn pivot_lemma(mut self, lemma: impl Into<String>) -> Result<Self, String> {
        self.pivot_lemma = Some(non_blank("pivot lemma", lemma.into())?);
        Ok(self)
    }

    /// Sets the confidence written on matches.
    pub fn confidence(mut self, confidence: f64) -> Result<Self, String> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(format!("confidence must be finite and in [0, 1]: {confidence}"));
        }
        self.confidence = confidence;
        Ok(self)
    }

    /// The parsed steps of the path, in order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The steps as they appear in the path (`<nsubj`, `>obj`),
    /// as the reference `steps()` returns them.
    pub fn step_strings(&self) -> Vec<String> {
        self.steps.iter().map(Step::as_str).collect()
    }
}

fn non_blank(what: &str, value: String) -> Result<String, String> {
    if is_blank(&value) {
        return Err(format!("{what} must not be blank"));
    }
    Ok(value)
}

/// A typed relation between two entities, by index into the
/// `Entities` layer used for extraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationMention {
    /// The relation type, never blank.
    pub relation_type: String,
    pub subject: Ix<Entities>,
    pub object: Ix<Entities>,
}

impl RelationMention {
    /// Validated as the reference record validates (indexes are
    /// `Ix`, so only the type and the subject-object distinction
    /// can fail).
    pub fn new(relation_type: impl Into<String>, subject: Ix<Entities>, object: Ix<Entities>) -> Result<Self, String> {
        let relation_type = relation_type.into();
        if is_blank(&relation_type) {
            return Err("type must not be null or blank".into());
        }
        if subject == object {
            return Err(format!("subject and object must differ: {}", subject.get()));
        }
        Ok(RelationMention { relation_type, subject, object })
    }
}
