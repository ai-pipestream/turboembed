//! Sentence-aligned windows for text longer than one encoder pass, as
//! GLiNER cuts long documents (`turbo_gliner`'s window planner): the text
//! is cut on sentence boundaries (token boundaries when the document has
//! no sentences) into windows that each fit the model's limits, each
//! window after the first starting [`OVERLAP`] sentences back. Every
//! window is extracted on its own and the results merged:
//!
//! - a span value found in more than one window is one value, with the
//!   highest score any window gave it (and that window's logit and
//!   support); an entity type of `Single` cardinality keeps only its
//!   highest-scoring value over all windows;
//! - a record lives in one window: its fields never combine values from
//!   different windows. A record whose values (type, field, span or
//!   choice) all equal an earlier record's is the same record found again
//!   in the overlap, and is dropped; instance numbers count the records
//!   kept, per type, in text order;
//! - every classification label's probability, chosen or not, is the
//!   highest any window gave it; these maxima are not renormalized. The
//!   document's chosen labels are every window's, best first:
//!   one for a single-label task, at most `top_k` for a multi-label one.

use std::ops::Range;

use turbo_analysis::{Layer, View};
use turbo_analysis_layers::{Sentences, Tokens};

use crate::decode::{Corroborate, Extraction, FieldValue, Record, SpanValue};
use crate::model::ExtractError;
use crate::schema::{Cardinality, ExtractionSchema, Source};

/// The sentences each window after the first repeats from the one before.
pub const OVERLAP: usize = 1;

/// Whether `e` is text too long for one pass, which windows answer.
pub fn too_long(e: &ExtractError) -> bool {
    matches!(e, ExtractError::OverLimit { limit: "max_words" | "encoder max_sequence", .. })
}

/// The byte ranges windows are cut between: the document's sentences,
/// or its tokens when it has none, in text order.
pub fn units(view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Vec<Range<usize>>, String> {
    let ranges: Vec<Range<usize>> = match sentences {
        Some(s) => {
            s.check()?;
            if s.spans.view() != view.id() || s.spans.fragment_columns().is_some() {
                return Err("windows need contiguous sentences in the view read".into());
            }
            let (a, b) = s.spans.columns();
            a.iter().zip(b).map(|(&a, &b)| a as usize..b as usize).collect()
        }
        None => {
            tokens.check()?;
            let (a, b) = tokens.spans.columns();
            a.iter().zip(b).map(|(&a, &b)| a as usize..b as usize).collect()
        }
    };
    let text = view.text();
    for (i, r) in ranges.iter().enumerate() {
        if r.start > r.end || text.get(r.clone()).is_none() {
            return Err(format!("window unit {i}: not a range of the text on character boundaries"));
        }
        if i > 0 && r.start < ranges[i - 1].end {
            return Err(format!("window unit {i}: overlaps the one before"));
        }
    }
    Ok(ranges.into_iter().filter(|r| r.start < r.end).collect())
}

/// The windows over `units`: each the longest run of units from its first
/// that `fits`, the next starting [`OVERLAP`] units before the last one
/// ended, always moving forward. A unit that does not fit alone is an
/// error naming it.
pub fn plan(
    text: &str,
    units: &[Range<usize>],
    fits: impl Fn(&str) -> Result<bool, String>,
) -> Result<Vec<Range<usize>>, String> {
    let mut windows = Vec::new();
    let mut i = 0;
    while i < units.len() {
        let start = units[i].start;
        if !fits(&text[start..units[i].end])? {
            return Err(format!("window unit {i} ({} bytes) alone is over the model's limits", units[i].len()));
        }
        let mut j = i;
        while j + 1 < units.len() && fits(&text[start..units[j + 1].end])? {
            j += 1;
        }
        windows.push(start..units[j].end);
        if j + 1 == units.len() {
            break;
        }
        i = if j >= i + OVERLAP { j + 1 - OVERLAP } else { j + 1 };
    }
    Ok(windows)
}

/// Plans from one prompt measurement. Word lengths and prefix sums are
/// built once; extending or advancing a window does no tokenization.
pub fn plan_measured(
    text: &str,
    units: &[Range<usize>],
    measure: &crate::prompt::Measurement,
    max_words: usize,
    max_sequence: usize,
) -> Result<Vec<Range<usize>>, String> {
    let mut first = Vec::with_capacity(units.len());
    let mut last = Vec::with_capacity(units.len());
    let mut at = 0;
    for unit in units {
        while at < measure.words.len() && measure.words[at].0.start < unit.start {
            at += 1;
        }
        first.push(at);
        while at < measure.words.len() && measure.words[at].0.end <= unit.end {
            at += 1;
        }
        last.push(at);
    }
    let mut prefix = vec![0];
    for (_, n) in &measure.words {
        prefix.push(prefix.last().unwrap() + n);
    }
    let fits = |i: usize, j: usize| {
        let (a, b) = (first[i], last[j]);
        let sample = &text[units[i].start..units[j].end];
        let period = usize::from(sample.is_empty() || !sample.ends_with(['.', '!', '?'])) * measure.period;
        b - a <= max_words && measure.overhead + prefix[b] - prefix[a] + period <= max_sequence
    };
    let mut windows = Vec::new();
    let mut i = 0;
    while i < units.len() {
        if !fits(i, i) {
            return Err(format!("window unit {i} ({} bytes) alone is over the model's limits", units[i].len()));
        }
        let mut j = i;
        while j + 1 < units.len() && fits(i, j + 1) {
            j += 1;
        }
        windows.push(units[i].start..units[j].end);
        if j + 1 == units.len() {
            break;
        }
        i = if j >= i + OVERLAP { j + 1 - OVERLAP } else { j + 1 };
    }
    Ok(windows)
}

/// The document's layers, read at a window's offset in the text.
pub struct Shifted<'a> {
    pub layers: &'a dyn Corroborate,
    pub by: usize,
}

impl Corroborate for Shifted<'_> {
    fn supporting(&self, source: Source, start: usize, end: usize) -> Vec<u32> {
        self.layers.supporting(source, start + self.by, end + self.by)
    }
}

/// A window's value in the document's bytes: cut back to the window's
/// text (the model may read a `.` past it), then moved by its offset.
fn shift(v: &SpanValue, by: usize, len: usize) -> Option<SpanValue> {
    let end = v.end.min(len);
    (v.start < end).then(|| SpanValue { start: v.start + by, end: end + by, ..v.clone() })
}

/// Keeps the higher-scored of a value found again.
fn keep(values: &mut Vec<SpanValue>, v: SpanValue) {
    match values.iter_mut().find(|o| (o.start, o.end) == (v.start, v.end)) {
        Some(o) if o.score < v.score => *o = v,
        Some(_) => {}
        None => values.push(v),
    }
}

/// What a record says, scores aside: its type and every field's values.
fn same(a: &Record, b: &Record) -> bool {
    let key = |v: &FieldValue| match v {
        FieldValue::Span(s) => (0, s.start, s.end),
        FieldValue::Choice { choice, .. } => (1, *choice, 0),
    };
    a.structure == b.structure
        && a.fields.len() == b.fields.len()
        && a.fields.iter().zip(&b.fields).all(|(x, y)| {
            let (mut x, mut y): (Vec<_>, Vec<_>) = (x.iter().map(key).collect(), y.iter().map(key).collect());
            x.sort_unstable();
            y.sort_unstable();
            x == y
        })
}

/// Merges one window's extraction, made over `text[at]`, into `out`.
pub fn merge(out: &mut Extraction, window: Extraction, at: Range<usize>) {
    let (by, len) = (at.start, at.len());
    if out.entities.len() < window.entities.len() {
        out.entities.resize(window.entities.len(), Vec::new());
    }
    for (k, values) in window.entities.into_iter().enumerate() {
        for v in values.iter().filter_map(|v| shift(v, by, len)) {
            keep(&mut out.entities[k], v);
        }
    }
    for mut r in window.records {
        for values in &mut r.fields {
            let mut moved = Vec::new();
            for v in values.drain(..) {
                match v {
                    FieldValue::Span(s) => {
                        if let Some(s) = shift(&s, by, len) {
                            moved.push(FieldValue::Span(s));
                        }
                    }
                    c => moved.push(c),
                }
            }
            *values = moved;
        }
        if r.fields.iter().all(Vec::is_empty) || out.records.iter().any(|o| same(o, &r)) {
            continue;
        }
        r.instance = out.records.iter().filter(|o| o.structure == r.structure).count();
        out.records.push(r);
    }
    for c in window.classifications {
        let i = match out.classifications.iter().position(|o| o.task == c.task) {
            Some(i) => i,
            None => {
                out.classifications.push(c);
                continue;
            }
        };
        let merged = &mut out.classifications[i];
        for (best, p) in merged.probs.iter_mut().zip(c.probs) {
            *best = best.max(p);
        }
        let labels = &mut merged.labels;
        for l in c.labels {
            match labels.iter_mut().find(|o| o.0 == l.0) {
                Some(o) if o.1 < l.1 => *o = l,
                Some(_) => {}
                None => labels.push(l),
            }
        }
    }
}

/// The merged extraction held to the schema's cardinalities, in a stable
/// order: values by position, labels by probability, tasks by their order
/// in the schema.
pub fn finish(out: &mut Extraction, schema: &ExtractionSchema) {
    for (values, e) in out.entities.iter_mut().zip(&schema.entities) {
        if e.cardinality == Cardinality::Single
            && let Some(best) = values.iter().cloned().reduce(|b, v| if v.score > b.score { v } else { b })
        {
            *values = vec![best];
        }
        values.sort_by_key(|v| (v.start, v.end));
    }
    out.classifications.sort_by_key(|c| c.task);
    for c in &mut out.classifications {
        c.labels.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let t = &schema.classifications[c.task];
        c.labels.truncate(if t.multi_label { t.top_k.unwrap_or(usize::MAX) } else { 1 });
    }
}
