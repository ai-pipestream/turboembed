//! A pass's scores decoded as GLiNER2 decodes them, with the schema's
//! deterministic checks applied where GLiNER2 applies its validators:
//! after the spans above the threshold are found and before overlapping
//! ones are suppressed.
//!
//! - A span task's candidates for a field are the spans of whole text words
//!   whose score is at or above the threshold, in start then width order.
//!   They are ranked by score (ties keep that order), and each is kept if
//!   it overlaps no span kept before it; a single-valued field keeps the
//!   first.
//! - A closed-choice field scores each choice at the first prefix token
//!   equal to it without regard to case; a repeated field keeps every
//!   choice at or above the threshold in the schema's order, a single one
//!   the best if it is at or above.
//! - A record instance is kept if any field has a value.
//! - A single-label classification is the argmax of a softmax; a
//!   multi-label one keeps every label whose sigmoid is at or above the
//!   task's threshold (else the best), the best `top_k` first.

use crate::model::{Pass, Scores};
use crate::prompt::Kind;
use crate::schema::{Cardinality, Check, ExtractionSchema, Source};

/// The deterministic layers a candidate can be checked against.
pub trait Corroborate {
    /// The rows of `source` whose spans overlap bytes `start..end` of the
    /// text, in row order.
    fn supporting(&self, source: Source, start: usize, end: usize) -> Vec<u32>;
}

/// No deterministic layers: every `Require` check fails, no boost applies.
pub struct Nothing;

impl Corroborate for Nothing {
    fn supporting(&self, _: Source, _: usize, _: usize) -> Vec<u32> {
        Vec::new()
    }
}

/// One span value: its bytes in the prompt's text, its score and logit,
/// and the rows of the deterministic layers that corroborate it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanValue {
    pub start: usize,
    pub end: usize,
    pub score: f32,
    pub logit: f32,
    pub supported_by: Vec<(Source, u32)>,
}

/// One field value: a span of the text or a closed choice.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue {
    Span(SpanValue),
    /// The index of the choice in the field's choices, its score and logit.
    Choice {
        choice: usize,
        score: f32,
        logit: f32,
    },
}

/// One record instance: per field of the record type, its values.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub structure: usize,
    pub instance: usize,
    pub fields: Vec<Vec<FieldValue>>,
}

/// One classification task's chosen labels: label index, probability and
/// logit, in the order GLiNER2 gives them.
#[derive(Clone, Debug, PartialEq)]
pub struct Classified {
    pub task: usize,
    pub labels: Vec<(usize, f32, f32)>,
}

/// Everything one pass extracted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extraction {
    /// Per entity type of the schema, its spans.
    pub entities: Vec<Vec<SpanValue>>,
    pub classifications: Vec<Classified>,
    pub records: Vec<Record>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The spans of one field's logits (`[tokens][width]`, the words after
/// `skip` prefix tokens) at or above `threshold`, after `checks`, ranked,
/// overlaps suppressed and cut to one for a single-valued field.
fn spans(
    pass: &Pass,
    logits: &[f32],
    skip: usize,
    threshold: f32,
    cardinality: Cardinality,
    checks: &[Check],
    layers: &dyn Corroborate,
) -> Vec<SpanValue> {
    let (w, words) = (pass.width(), &pass.prompt().words);
    let mut found = Vec::new();
    for start in 0..words.len() {
        for width in 0..w.min(words.len() - start) {
            let logit = logits[(skip + start) * w + width];
            let (s, e) = (words[start].start, words[start + width].end);
            let mut supported_by: Vec<(Source, u32)> = Vec::new();
            let mut sources: Vec<Source> = Vec::new();
            for c in checks {
                let (Check::Require(src) | Check::Boost(src, _)) = c;
                if !sources.contains(src) {
                    let rows = layers.supporting(*src, s, e);
                    if !rows.is_empty() {
                        sources.push(*src);
                        supported_by.extend(rows.into_iter().map(|r| (*src, r)));
                    }
                }
            }
            let boost: f32 = checks
                .iter()
                .filter_map(|c| match c {
                    Check::Boost(src, b) if sources.contains(src) => Some(*b),
                    _ => None,
                })
                .sum();
            let score = (sigmoid(logit) + boost).min(1.0);
            let required = checks.iter().all(|c| match c {
                Check::Require(src) => sources.contains(src),
                Check::Boost(..) => true,
            });
            if score >= threshold && required {
                found.push(SpanValue { start: s, end: e, score, logit, supported_by });
            }
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<SpanValue> = Vec::new();
    for c in found {
        if kept.iter().all(|k| !(c.start < k.end && k.start < c.end)) {
            kept.push(c);
        }
    }
    if cardinality == Cardinality::Single {
        kept.truncate(1);
    }
    kept
}

/// The values of a closed-choice field, whose first choice is prefix token
/// `first`, from its logits over the prefix.
fn choices(
    pass: &Pass,
    logits: &[f32],
    choices: &[String],
    first: usize,
    threshold: f32,
    cardinality: Cardinality,
) -> Vec<FieldValue> {
    let w = pass.width();
    let scored: Vec<(usize, f32, f32)> = (0..choices.len())
        .map(|i| {
            let p = first + 2 * i;
            (i, sigmoid(logits[p * w]), logits[p * w])
        })
        .collect();
    let value = |(choice, score, logit): (usize, f32, f32)| FieldValue::Choice { choice, score, logit };
    match cardinality {
        Cardinality::Repeated => scored.into_iter().filter(|c| c.1 >= threshold).map(value).collect(),
        Cardinality::Single => {
            let mut best: Option<(usize, f32, f32)> = None;
            for c in scored {
                if best.is_none_or(|b| c.1 > b.1) {
                    best = Some(c);
                }
            }
            best.filter(|b| b.1 >= threshold).map(value).into_iter().collect()
        }
    }
}

fn classify(task: usize, schema: &ExtractionSchema, logits: &[f32]) -> Classified {
    let t = &schema.classifications[task];
    let probs: Vec<f32> = if t.multi_label {
        logits.iter().map(|&l| sigmoid(l)).collect()
    } else {
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&l| (l - top).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|v| v / sum).collect()
    };
    let best = (0..probs.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
    let pick = |i: usize| (i, probs[i], logits[i]);
    let labels = if t.multi_label {
        let mut chosen: Vec<(usize, f32, f32)> =
            (0..probs.len()).filter(|&i| probs[i] >= t.threshold).map(pick).collect();
        if chosen.is_empty() {
            chosen.push(pick(best));
        }
        if let Some(k) = t.top_k {
            chosen.sort_by(|a, b| b.1.total_cmp(&a.1));
            chosen.truncate(k);
        }
        chosen
    } else {
        vec![pick(best)]
    };
    Classified { task, labels }
}

/// Whether `pass` is the pass of `schema`: its width is at least one and
/// its text tokens are its prefix then its words; its tasks are the schema's, in
/// prompt order, with their children; every score vector has the length
/// its task, instances, text tokens and width give; no logit is NaN or
/// positive infinity (negative infinity marks a span past the text); and
/// the threshold is a probability.
fn check(pass: &Pass, schema: &ExtractionSchema, threshold: f32) -> Result<(), String> {
    let p = pass.prompt();
    if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
        return Err(format!("a threshold outside [0, 1]: {threshold}"));
    }
    let mut kinds: Vec<(Kind, usize)> =
        schema.structures.iter().enumerate().map(|(i, s)| (Kind::Record(i), s.fields.len())).collect();
    if !schema.entities.is_empty() {
        kinds.push((Kind::Entities, schema.entities.len()));
    }
    kinds.extend(schema.classifications.iter().enumerate().map(|(i, c)| (Kind::Classification(i), c.labels.len())));
    // Every text token is a prefix token or a word, and a span is at least
    // one word wide: the bounds every score index below rests on.
    if pass.width() == 0 || p.prefix.len() + p.words.len() != p.token_first.len() {
        return Err("a pass whose text tokens or width do not fit its prompt".into());
    }
    let tasks: Vec<(Kind, usize)> = p.tasks.iter().map(|t| (t.kind, t.markers.len())).collect();
    if tasks != kinds || pass.scores().len() != tasks.len() {
        return Err("a pass whose tasks are not the schema's".into());
    }
    if p.choice_at.len() != schema.structures.len()
        || schema.structures.iter().zip(&p.choice_at).any(|(s, at)| {
            at.len() != s.fields.len()
                || s.fields.iter().zip(at).any(|(f, a)| match (&f.choices, a) {
                    (Some(c), Some(first)) => first + 2 * c.len() > p.prefix.len() + 1,
                    (None, None) => false,
                    _ => true,
                })
        })
    {
        return Err("a pass whose choice prefix is not the schema's".into());
    }
    let size = p.token_first.len() * pass.width();
    let bad = |l: &[f32]| l.iter().any(|x| x.is_nan() || *x == f32::INFINITY);
    for ((kind, fields), scores) in kinds.iter().zip(pass.scores()) {
        match (kind, scores) {
            (Kind::Classification(_), Scores::Labels(l)) if l.len() == *fields && l.iter().all(|x| x.is_finite()) => {}
            (Kind::Record(_) | Kind::Entities, Scores::Spans { count, instances, logits })
                if instances <= count
                    && (*kind != Kind::Entities || *instances <= 1)
                    && Some(logits.len()) == instances.checked_mul(*fields).and_then(|x| x.checked_mul(size))
                    && !bad(logits) => {}
            _ => return Err("a task's scores that do not fit it, or are not numbers".into()),
        }
    }
    Ok(())
}

/// The extraction from `pass` for `schema` at the request's `threshold`,
/// each field's or type's own threshold taking its place where set.
///
/// # Errors
/// A pass that is not `schema`'s, has scores of the wrong size or that are
/// not numbers, or a threshold outside [0, 1].
pub fn decode(
    pass: &Pass,
    schema: &ExtractionSchema,
    threshold: f32,
    layers: &dyn Corroborate,
) -> Result<Extraction, String> {
    check(pass, schema, threshold)?;
    let mut out = Extraction { entities: vec![Vec::new(); schema.entities.len()], ..Extraction::default() };
    let (n, skip, w) = (pass.prompt().token_first.len(), pass.prompt().prefix_len(), pass.width());
    let size = n * w;
    for (task, scores) in pass.prompt().tasks.iter().zip(pass.scores()) {
        match (task.kind, scores) {
            (Kind::Classification(i), Scores::Labels(l)) => out.classifications.push(classify(i, schema, l)),
            (Kind::Entities, Scores::Spans { instances, logits, .. }) if *instances > 0 => {
                for (f, e) in schema.entities.iter().enumerate() {
                    let field = &logits[f * size..(f + 1) * size];
                    let t = e.threshold.unwrap_or(threshold);
                    out.entities[f] = spans(pass, field, skip, t, e.cardinality, &e.checks, layers);
                }
            }
            (Kind::Record(s), Scores::Spans { instances, logits, .. }) => {
                let spec = &schema.structures[s];
                let m = spec.fields.len();
                for inst in 0..*instances {
                    let mut fields = Vec::with_capacity(m);
                    for (f, fs) in spec.fields.iter().enumerate() {
                        let field = &logits[(inst * m + f) * size..(inst * m + f + 1) * size];
                        let t = fs.threshold.unwrap_or(threshold);
                        fields.push(match (&fs.choices, pass.prompt().choice_at[s][f]) {
                            (Some(c), Some(first)) => choices(pass, field, c, first, t, fs.cardinality),
                            _ => spans(pass, field, skip, t, fs.cardinality, &fs.checks, layers)
                                .into_iter()
                                .map(FieldValue::Span)
                                .collect(),
                        });
                    }
                    if fields.iter().any(|v| !v.is_empty()) {
                        out.records.push(Record { structure: s, instance: inst, fields });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}
