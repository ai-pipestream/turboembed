//! The structured-record layer: every value a record type's fields took,
//! each a span of the text with its record, its field (`record.field`), its
//! probability and logit, and the deterministic layers that corroborate it;
//! and every closed-choice value, which has no span, with its choice.
//! Values of one record may nest and overlap.

use std::sync::Arc;

use turbo_analysis::{Anchor, Key, KeyId, Layer, Overlap, Spans, Tag, TagSet, TextSpans};
use turbo_analysis_layers::{Numbers, Pii};
use turbo_geo::Locations;
use turbo_glossary_layers::GlossaryTerms;

use crate::schema::Source;

/// A source's code in [`Records::support_source`].
pub fn source_code(s: Source) -> u8 {
    match s {
        Source::Glossary => 0,
        Source::Pii => 1,
        Source::Numbers => 2,
        Source::Locations => 3,
    }
}

/// The source a code in [`Records::support_source`] names.
pub fn source_of(code: u8) -> Option<Source> {
    [Source::Glossary, Source::Pii, Source::Numbers, Source::Locations].get(code as usize).copied()
}

/// The default instance of the layer `s` names. The annotator may read
/// another instance; the evidence resolves which through the provenance,
/// and the codec through the instance the record was made on.
pub fn source_key(s: Source) -> KeyId {
    match s {
        Source::Glossary => Key::<GlossaryTerms>::DEFAULT.id(),
        Source::Pii => Key::<Pii>::DEFAULT.id(),
        Source::Numbers => Key::<Numbers>::DEFAULT.id(),
        Source::Locations => Key::<Locations>::DEFAULT.id(),
    }
}

/// `turbo:records`.
pub struct Records {
    /// One span per span value.
    pub spans: Spans<Records>,
    /// Per span value: its record, its field path, probability and logit.
    pub record: Vec<u32>,
    pub path: Vec<Tag>,
    pub score: Vec<f64>,
    pub logit: Vec<f64>,
    /// Per span value, its run of corroborating rows: value `i` holds
    /// `support_source[support_end[i - 1]..support_end[i]]` and the rows in
    /// `support_row` beside them (from the start when `i == 0`).
    pub support_end: Vec<u32>,
    /// A [`source_code`] per supporting row.
    pub support_source: Vec<u8>,
    /// The supporting row's index in that source's default instance.
    pub support_row: Vec<u32>,
    /// Per record: its type and the model's instance index.
    pub record_type: Vec<Tag>,
    pub instance: Vec<u32>,
    /// Per closed-choice value: its record, path, choice, probability and
    /// logit.
    pub choice_record: Vec<u32>,
    pub choice_path: Vec<Tag>,
    pub choice: Vec<Tag>,
    pub choice_score: Vec<f64>,
    pub choice_logit: Vec<f64>,
    /// The record types, every `record.field` path, and every choice.
    pub types: Arc<TagSet>,
    pub paths: Arc<TagSet>,
    pub choices: Arc<TagSet>,
}

impl Records {
    /// The supporting rows of span value `i`.
    pub fn support(&self, i: usize) -> impl Iterator<Item = (Source, u32)> + '_ {
        let start = if i == 0 { 0 } else { self.support_end[i - 1] as usize };
        let end = self.support_end[i] as usize;
        (start..end).filter_map(|k| source_of(self.support_source[k]).map(|s| (s, self.support_row[k])))
    }
}

fn unit(what: &str, scores: &[f64], logits: &[f64]) -> Result<(), String> {
    if let Some(s) = scores.iter().find(|s| !s.is_finite() || **s < 0.0 || **s > 1.0) {
        return Err(format!("a {what} score outside [0, 1]: {s}"));
    }
    if let Some(l) = logits.iter().find(|l| !l.is_finite()) {
        return Err(format!("a {what} logit that is not finite: {l}"));
    }
    Ok(())
}

impl Layer for Records {
    const ID: &'static str = "turbo:records";
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        let (n, records) = (self.spans.len(), self.record_type.len());
        if [self.record.len(), self.path.len(), self.score.len(), self.logit.len(), self.support_end.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err("a record, a path, a score, a logit and a support run for every value".into());
        }
        let c = self.choice_record.len();
        if [self.choice_path.len(), self.choice.len(), self.choice_score.len(), self.choice_logit.len()]
            .iter()
            .any(|&l| l != c)
        {
            return Err("a record, a path, a choice, a score and a logit for every choice value".into());
        }
        if self.instance.len() != records {
            return Err("an instance for every record".into());
        }
        unit("value", &self.score, &self.logit)?;
        unit("choice", &self.choice_score, &self.choice_logit)?;
        if self.record_type.iter().any(|t| t.0 as usize >= self.types.len()) {
            return Err(format!("a record type outside {}", self.types.name));
        }
        if self.choice.iter().any(|t| t.0 as usize >= self.choices.len()) {
            return Err(format!("a choice outside {}", self.choices.name));
        }
        let mut used = vec![false; records];
        for (&r, &p) in self.record.iter().zip(&self.path).chain(self.choice_record.iter().zip(&self.choice_path)) {
            let r = r as usize;
            if r >= records || p.0 as usize >= self.paths.len() {
                return Err("a value of a record or path that is not there".into());
            }
            let prefix = format!("{}.", self.types.label(self.record_type[r]));
            if !self.paths.label(p).starts_with(&prefix) {
                return Err(format!("path {} is not a field of {}", self.paths.label(p), prefix.trim_end_matches('.')));
            }
            used[r] = true;
        }
        if used.iter().any(|u| !u) {
            return Err("a record with no value".into());
        }
        if self.support_end.windows(2).any(|w| w[0] > w[1])
            || self.support_end.last().map_or(0, |&e| e as usize) != self.support_source.len()
            || self.support_row.len() != self.support_source.len()
        {
            return Err("support runs that do not end in order at the support rows".into());
        }
        if self.support_source.iter().any(|&s| source_of(s).is_none()) {
            return Err("a support source outside the four deterministic layers".into());
        }
        Ok(())
    }
}
