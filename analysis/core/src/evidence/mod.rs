//! One evidence record per result, whatever made it: a model-backed
//! entity, a rule-backed match, a document category. `Evidence<L>` is
//! aligned to the layer `L` of results, item for item, and points at the
//! layers that hold the result and its support rather than copying them,
//! so the result layers keep their own encoding.
//!
//! A record says:
//!
//! - which component produced the result and, when a model did, the
//!   model's identity (`ModelRef`), from a small table of producers;
//! - the result's spans in the source text (view 0), mapped back through
//!   the views from wherever the result was found;
//! - its raw score, and its calibrated score and abstention reason when
//!   the component gives them. A field a component does not produce is
//!   `None`, never zero;
//! - the units of other layers that support it (the tokens an entity
//!   covers, say), each as a layer instance and a range of its items.
//!
//! [`Evidence::verify`] checks a layer against its document: every model
//! a producer names ran on the document, every span lies inside the source
//! on character boundaries, and every support range lies inside a layer
//! the document holds. [`EvidenceStep`] builds the layer for any result
//! layer that implements [`Evidenced`].

mod codec;
mod step;
#[cfg(test)]
mod tests;

pub use step::{EvidenceStep, Evidenced, read_of};

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::doc::Document;
use crate::export::LayerRef;
use crate::layer::{Anchor, Layer, LayerName, ModelRef};
use crate::span::Span;

/// A component that produced results: the annotator's id and, when it
/// ran a model, the model's identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Producer {
    pub annotator: String,
    pub model: Option<ModelRef>,
}

/// The units of one layer instance that support a result: items
/// `start..end` of `layer`, an index into `Evidence::layers`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Support {
    pub layer: u32,
    pub start: u32,
    pub end: u32,
}

/// The evidence for each item of `L`, by column. Record i's spans are
/// `spans[span_row[i]..span_row[i + 1]]` and its support
/// `support[support_row[i]..support_row[i + 1]]`. The scores and the
/// abstentions are sparse, `(record, value)` in record order, so a record
/// whose producer gave none has none, and a producer that gives none at
/// all costs nothing.
pub struct Evidence<L> {
    /// Shared: the documents of a batch that one step ran on hold one
    /// table.
    pub producers: Arc<Vec<Producer>>,
    /// Each record's producer: an index into `producers`.
    pub producer: Vec<u32>,
    pub span_row: Vec<u32>,
    /// Byte spans of the source text (view 0).
    pub spans: Vec<Span>,
    /// Scores as their producers gave them.
    pub raw: Vec<(u32, f64)>,
    /// Scores in [0, 1] calibrated to the result being right.
    pub calibrated: Vec<(u32, f64)>,
    /// Why a component declined to commit to a result.
    pub abstention: Vec<(u32, String)>,
    /// The layer instances support ranges point into, shared as the
    /// producers are.
    pub layers: Arc<Vec<LayerRef>>,
    pub support_row: Vec<u32>,
    pub support: Vec<Support>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> Clone for Evidence<L> {
    fn clone(&self) -> Self {
        Evidence {
            producers: self.producers.clone(),
            producer: self.producer.clone(),
            span_row: self.span_row.clone(),
            spans: self.spans.clone(),
            raw: self.raw.clone(),
            calibrated: self.calibrated.clone(),
            abstention: self.abstention.clone(),
            layers: self.layers.clone(),
            support_row: self.support_row.clone(),
            support: self.support.clone(),
            _of: PhantomData,
        }
    }
}

impl<L> Default for Evidence<L> {
    fn default() -> Self {
        Evidence {
            producers: Arc::default(),
            producer: Vec::new(),
            span_row: vec![0],
            spans: Vec::new(),
            raw: Vec::new(),
            calibrated: Vec::new(),
            abstention: Vec::new(),
            layers: Arc::default(),
            support_row: vec![0],
            support: Vec::new(),
            _of: PhantomData,
        }
    }
}

/// Record i's entry of a sparse column: a binary search.
fn sparse<T>(column: &[(u32, T)], i: usize) -> Option<&T> {
    let i = u32::try_from(i).ok()?;
    column.binary_search_by_key(&i, |e| e.0).ok().map(|k| &column[k].1)
}

impl<L> Evidence<L> {
    /// How many records.
    pub fn records(&self) -> usize {
        self.producer.len()
    }

    /// Record i's producer.
    pub fn producer_of(&self, i: usize) -> &Producer {
        &self.producers[self.producer[i] as usize]
    }

    /// Record i's source spans.
    pub fn spans_of(&self, i: usize) -> &[Span] {
        &self.spans[self.span_row[i] as usize..self.span_row[i + 1] as usize]
    }

    /// Record i's support.
    pub fn support_of(&self, i: usize) -> &[Support] {
        &self.support[self.support_row[i] as usize..self.support_row[i + 1] as usize]
    }

    /// Record i's raw score, if its producer gave one.
    pub fn raw_of(&self, i: usize) -> Option<f64> {
        sparse(&self.raw, i).copied()
    }

    /// Record i's calibrated score, if its producer gave one.
    pub fn calibrated_of(&self, i: usize) -> Option<f64> {
        sparse(&self.calibrated, i).copied()
    }

    /// Why record i's producer abstained, if it did.
    pub fn abstention_of(&self, i: usize) -> Option<&str> {
        sparse(&self.abstention, i).map(String::as_str)
    }

    /// The columns hold together: one row start per record and one past
    /// the last span and support, rows in order, every producer and
    /// support layer inside its table, support ranges forward, the sparse
    /// columns in record order with each record once, raw scores finite
    /// and calibrated scores in [0, 1].
    pub fn validate(&self) -> Result<(), String> {
        let n = self.producer.len();
        rows(&self.span_row, n, self.spans.len(), "span")?;
        rows(&self.support_row, n, self.support.len(), "support")?;
        if let Some(p) = self.producer.iter().find(|&&p| p as usize >= self.producers.len()) {
            return Err(format!("producer {p} of {}", self.producers.len()));
        }
        if let Some(s) = self.support.iter().find(|s| s.layer as usize >= self.layers.len() || s.start > s.end) {
            return Err(format!("support {s:?} with {} layers", self.layers.len()));
        }
        sparse_rows(self.raw.iter().map(|e| e.0), n, "raw score")?;
        sparse_rows(self.calibrated.iter().map(|e| e.0), n, "calibrated score")?;
        sparse_rows(self.abstention.iter().map(|e| e.0), n, "abstention")?;
        if let Some(r) = self.raw.iter().find(|r| !r.1.is_finite()) {
            return Err(format!("a raw score that is not finite: {}", r.1));
        }
        if let Some(c) = self.calibrated.iter().find(|c| !(0.0..=1.0).contains(&c.1)) {
            return Err(format!("a calibrated score outside [0, 1]: {}", c.1));
        }
        Ok(())
    }

    /// Checks the records against `doc`, the document they describe.
    pub fn verify(&self, doc: &Document) -> Result<(), EvidenceError> {
        self.validate().map_err(EvidenceError::Invalid)?;
        let ran: Vec<&ModelRef> =
            doc.layers().filter_map(|k| doc.provenance(k)).filter_map(|p| p.model.as_ref()).collect();
        for (p, producer) in self.producers.iter().enumerate() {
            if let Some(m) = &producer.model
                && !ran.contains(&m)
            {
                let record = self.producer.iter().position(|&q| q as usize == p);
                return Err(EvidenceError::UnknownModel { record, model: m.clone() });
            }
        }
        let source = doc.source();
        for i in 0..self.records() {
            for &span in self.spans_of(i) {
                let (a, b) = (span.start as usize, span.end as usize);
                if b > source.len() || a > b {
                    return Err(EvidenceError::SpanOutsideSource { record: i, span, len: source.len() });
                }
                if !source.is_char_boundary(a) || !source.is_char_boundary(b) {
                    return Err(EvidenceError::SpanOffCharBoundary { record: i, span });
                }
            }
        }
        let lens: Vec<Option<usize>> =
            self.layers.iter().map(|l| doc.layer_len_by_name(&l.layer, &l.instance)).collect();
        for i in 0..self.records() {
            for s in self.support_of(i) {
                let layer = &self.layers[s.layer as usize];
                match lens[s.layer as usize] {
                    None => return Err(EvidenceError::UnknownSupport { record: i, layer: layer.clone() }),
                    Some(len) if s.end as usize > len => {
                        return Err(EvidenceError::SupportOutOfRange {
                            record: i,
                            layer: layer.clone(),
                            end: s.end,
                            len,
                        });
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(())
    }
}

/// Record indexes of a sparse column: strictly increasing, each below `n`.
fn sparse_rows(records: impl Iterator<Item = u32>, n: usize, what: &str) -> Result<(), String> {
    let mut next = 0u64;
    for r in records {
        if u64::from(r) < next || r as usize >= n {
            return Err(format!("{what}s in record order, each once, for {n} records"));
        }
        next = u64::from(r) + 1;
    }
    Ok(())
}

fn rows(row: &[u32], n: usize, items: usize, what: &str) -> Result<(), String> {
    if row.len() != n + 1 || row[0] != 0 || row[n] as usize != items {
        return Err(format!("one {what} row start per record and one past the last {what}"));
    }
    if !row.windows(2).all(|w| w[0] <= w[1]) {
        return Err(format!("{what} rows in record order"));
    }
    Ok(())
}

/// Why evidence does not describe its document.
#[derive(Clone, Debug, PartialEq)]
pub enum EvidenceError {
    /// The columns do not hold together (`Evidence::validate`).
    Invalid(String),
    /// A producer names a model no layer of the document was made with;
    /// `record` is the first record it produced, if any.
    UnknownModel { record: Option<usize>, model: ModelRef },
    /// A span past the end of the source text.
    SpanOutsideSource { record: usize, span: Span, len: usize },
    /// A span that cuts a character of the source text.
    SpanOffCharBoundary { record: usize, span: Span },
    /// Support in a layer instance the document does not hold.
    UnknownSupport { record: usize, layer: LayerRef },
    /// Support past the end of its layer.
    SupportOutOfRange { record: usize, layer: LayerRef, end: u32, len: usize },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use EvidenceError::*;
        match self {
            Invalid(detail) => write!(f, "evidence columns: {detail}"),
            UnknownModel { record, model } => {
                write!(f, "record {record:?} names model {} {} that did not run here", model.id, model.revision)
            }
            SpanOutsideSource { record, span, len } => {
                write!(f, "record {record}: span {}..{} outside a source of {len} bytes", span.start, span.end)
            }
            SpanOffCharBoundary { record, span } => {
                write!(f, "record {record}: span {}..{} cuts a character", span.start, span.end)
            }
            UnknownSupport { record, layer } => write!(f, "record {record}: no layer {layer} to support it"),
            SupportOutOfRange { record, layer, end, len } => {
                write!(f, "record {record}: support up to {end} in {layer} of {len} items")
            }
        }
    }
}

impl std::error::Error for EvidenceError {}

impl<L: Layer> Layer for Evidence<L> {
    const ID: &'static str = "turbo:evidence";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.producer.len()
    }
    fn check(&self) -> Result<(), String> {
        self.validate()
    }
}
