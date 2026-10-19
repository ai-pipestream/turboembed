//! The data indexers: `OnePassDataIndexer`, `TwoPassDataIndexer` and
//! `OnePassRealValueDataIndexer`, which turn events into the integer form
//! the trainers consume.
//!
//! All three do the same work, in `AbstractDataIndexer`:
//!
//! 1. Count how often each predicate occurs over all events (an event that
//!    names a predicate twice counts it twice). The predicates with a count
//!    at or above the cutoff, sorted by Java's string order, are the
//!    predicate labels; their index in that order is their id.
//! 2. Number the outcomes in the order first seen. Every event's outcome is
//!    numbered, including the outcomes of events dropped in the next step.
//! 3. Map each event's context to predicate ids, keeping order and
//!    duplicates and leaving out predicates under the cutoff. An event with
//!    no predicate left is dropped.
//! 4. When sorting is on: sort the events (stable) by outcome id, then
//!    predicate ids one by one with their values, then length; merge runs
//!    of equal events into the first, counting how often it was seen. With
//!    no event left, `InsufficientTrainingDataException`.
//!
//! They differ in what they keep: `OnePass` and `TwoPass` drop the values
//! after sorting (`getValues()` is null); `OnePassRealValue` keeps them.
//! `TwoPass` spills the events to a file with `DataOutputStream.writeUTF`
//! between step 1 and 3, so a string over 65535 bytes of modified UTF-8
//! is an error there and nowhere else.
//!
//! Two OpenNLP quirks are kept on purpose and marked in the code: the
//! values an event is sorted by (and keeps, in `OnePassRealValue`) are the
//! values of its full context, not of the predicates that survived the
//! cutoff, so after a drop they are read at the wrong positions; and the
//! outcome labels include outcomes only dropped events had.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::{Error, NO_EVENTS, Result};
use crate::event::Event;
use crate::java_string::JavaString;
use crate::params::{DEFAULT_CUTOFF, Key, TrainingParameters};

/// Which of OpenNLP's three indexers to run: the value of the
/// `DataIndexer` parameter.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum IndexerKind {
    /// `OnePassDataIndexer`: everything in memory, values dropped.
    OnePass,
    /// `TwoPassDataIndexer`: events spilled to a file (strings limited to
    /// 65535 bytes), values dropped. The default.
    #[default]
    TwoPass,
    /// `OnePassRealValueDataIndexer`: as `OnePass`, values kept.
    OnePassRealValue,
}

impl IndexerKind {
    /// The parameter value naming this indexer.
    pub fn name(self) -> &'static str {
        match self {
            IndexerKind::OnePass => "OnePass",
            IndexerKind::TwoPass => "TwoPass",
            IndexerKind::OnePassRealValue => "OnePassRealValue",
        }
    }

    /// The indexer a parameter value names, matched exactly as
    /// `DataIndexerFactory` matches it.
    pub fn parse(name: &str) -> Option<IndexerKind> {
        match name {
            "OnePass" => Some(IndexerKind::OnePass),
            "TwoPass" => Some(IndexerKind::TwoPass),
            "OnePassRealValue" => Some(IndexerKind::OnePassRealValue),
            _ => None,
        }
    }

    /// Whether `getValues()` returns the events' values rather than null.
    pub fn keeps_values(self) -> bool {
        self == IndexerKind::OnePassRealValue
    }
}

/// What an indexer run is configured with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexParams {
    pub kind: IndexerKind,
    /// A predicate is kept when its count is at or above this. OpenNLP
    /// reads it as a Java `int`; zero or less keeps every predicate.
    pub cutoff: i32,
    /// Whether to sort and merge the events (the `sort` parameter, which
    /// the event trainers set from `isSortAndMerge()`).
    pub sort: bool,
}

impl Default for IndexParams {
    fn default() -> IndexParams {
        IndexParams::new(IndexerKind::TwoPass, DEFAULT_CUTOFF, true)
    }
}

impl IndexParams {
    pub fn new(kind: IndexerKind, cutoff: i32, sort: bool) -> IndexParams {
        IndexParams { kind, cutoff, sort }
    }

    /// What `DataIndexerFactory.getDataIndexer` and the indexers read from
    /// the training parameters: `DataIndexer` (default `TwoPass`), `Cutoff`
    /// (default 5) and `sort` (default true).
    pub fn from_training_parameters(params: &TrainingParameters) -> Result<IndexParams> {
        let name = params.get_str(Key::DATA_INDEXER)?.unwrap_or(IndexerKind::TwoPass.name());
        let kind = IndexerKind::parse(name).ok_or_else(|| Error::UnknownIndexer(name.to_owned()))?;
        Ok(IndexParams {
            kind,
            cutoff: params.get_int(Key::CUTOFF)?.unwrap_or(DEFAULT_CUTOFF),
            sort: params.get_bool(Key::SORT)?.unwrap_or(true),
        })
    }
}

/// The values of the unique events an `OnePassRealValue` indexer keeps:
/// one range per unique event into `values`, empty where the event had
/// none (OpenNLP's null).
#[derive(Debug, Clone, PartialEq)]
struct EventValues {
    starts: Vec<u32>,
    values: Vec<f32>,
}

/// What an indexer produced: `DataIndexer`'s getters.
#[derive(Debug, Clone, PartialEq)]
pub struct DataIndex {
    kind: IndexerKind,
    outcome_labels: Vec<JavaString>,
    pred_labels: Vec<JavaString>,
    pred_counts: Vec<u32>,
    num_events: u32,
    /// Per unique event.
    outcomes: Vec<u32>,
    seen: Vec<u32>,
    /// `context_starts[i]..context_starts[i + 1]` of `contexts` is event i's.
    context_starts: Vec<u32>,
    contexts: Vec<u32>,
    values: Option<EventValues>,
}

/// One event between indexing and merging: `ComparableEvent`. The
/// predicate ids are a range of a shared buffer; the values are the
/// event's own, full length (see the module documentation).
struct Comparable<'e> {
    outcome: u32,
    preds: std::ops::Range<usize>,
    values: Option<&'e [f32]>,
    seen: u32,
}

/// `Float.compare`: numeric order, then `-0.0` below `0.0`, then NaN
/// above everything and equal to itself.
fn float_compare(a: f32, b: f32) -> Ordering {
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    // Float.floatToIntBits collapses every NaN to one bit pattern.
    let bits = |x: f32| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() as i32 };
    bits(a).cmp(&bits(b))
}

/// `ComparableEvent.compareTo`.
fn compare(a: &Comparable, b: &Comparable, all: &[u32]) -> Ordering {
    if let o @ (Ordering::Less | Ordering::Greater) = a.outcome.cmp(&b.outcome) {
        return o;
    }
    let (pa, pb) = (&all[a.preds.clone()], &all[b.preds.clone()]);
    for i in 0..pa.len().min(pb.len()) {
        if let o @ (Ordering::Less | Ordering::Greater) = pa[i].cmp(&pb[i]) {
            return o;
        }
        // The values are read at the position in the filtered predicate
        // list, out of the array for the full context: OpenNLP's quirk.
        // An event without values counts 1.0 against one with them.
        let o = match (a.values, b.values) {
            (Some(va), Some(vb)) => float_compare(va[i], vb[i]),
            (Some(va), None) => float_compare(va[i], 1.0),
            (None, Some(vb)) => float_compare(1.0, vb[i]),
            (None, None) => Ordering::Equal,
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    pa.len().cmp(&pb.len())
}

impl DataIndex {
    /// Indexes `events` as the indexer `params.kind` would.
    pub fn index(events: &[Event], params: &IndexParams) -> Result<DataIndex> {
        // Step 1: counts, then the predicate labels in Java's string order.
        let mut counter: HashMap<&JavaString, u32> = HashMap::new();
        for e in events {
            if params.kind == IndexerKind::TwoPass {
                check_write_utf(e)?;
            }
            for c in e.context() {
                *counter.entry(c).or_insert(0) += 1;
            }
        }
        let mut kept: Vec<(&JavaString, u32)> =
            counter.iter().filter(|&(_, &n)| i64::from(n) >= i64::from(params.cutoff)).map(|(&s, &n)| (s, n)).collect();
        kept.sort_by(|a, b| a.0.cmp(b.0));
        let pred_index: HashMap<&JavaString, u32> = kept.iter().enumerate().map(|(i, (s, _))| (*s, i as u32)).collect();
        let pred_labels: Vec<JavaString> = kept.iter().map(|(s, _)| (*s).clone()).collect();
        let pred_counts: Vec<u32> = kept.iter().map(|(_, n)| *n).collect();

        // Steps 2 and 3: outcomes in order of first sight, contexts as ids.
        let mut outcome_index: HashMap<&JavaString, u32> = HashMap::new();
        let mut outcome_labels: Vec<JavaString> = Vec::new();
        let mut all_preds: Vec<u32> = Vec::new();
        let mut comparables: Vec<Comparable> = Vec::new();
        for e in events {
            let outcome = *outcome_index.entry(e.outcome()).or_insert_with(|| {
                outcome_labels.push(e.outcome().clone());
                outcome_labels.len() as u32 - 1
            });
            let start = all_preds.len();
            all_preds.extend(e.context().iter().filter_map(|c| pred_index.get(c).copied()));
            if all_preds.len() == start {
                // Dropped: its outcome stays numbered.
                continue;
            }
            // TwoPass writes a values array of length zero as none; one is
            // only possible with an empty context, which was just dropped.
            let values = e.values().filter(|v| !v.is_empty());
            comparables.push(Comparable { outcome, preds: start..all_preds.len(), values, seen: 1 });
        }

        // Step 4: sort and merge.
        let num_events = comparables.len();
        let mut order: Vec<usize> = (0..num_events).collect();
        let mut unique: Vec<usize> = Vec::with_capacity(num_events);
        if params.sort && num_events > 0 {
            order.sort_by(|&a, &b| compare(&comparables[a], &comparables[b], &all_preds));
            let mut champion = order[0];
            unique.push(champion);
            for &i in &order[1..] {
                if compare(&comparables[champion], &comparables[i], &all_preds) == Ordering::Equal {
                    comparables[champion].seen += 1;
                } else {
                    champion = i;
                    unique.push(i);
                }
            }
        } else {
            unique = order;
        }
        if unique.is_empty() {
            return Err(Error::InsufficientTrainingData(NO_EVENTS));
        }

        let mut outcomes = Vec::with_capacity(unique.len());
        let mut seen = Vec::with_capacity(unique.len());
        let mut context_starts = Vec::with_capacity(unique.len() + 1);
        let mut contexts = Vec::with_capacity(all_preds.len());
        let mut values = params.kind.keeps_values().then(|| EventValues { starts: vec![0], values: Vec::new() });
        for &i in &unique {
            let c = &comparables[i];
            outcomes.push(c.outcome);
            seen.push(c.seen);
            context_starts.push(contexts.len() as u32);
            contexts.extend_from_slice(&all_preds[c.preds.clone()]);
            if let Some(v) = &mut values {
                v.values.extend_from_slice(c.values.unwrap_or(&[]));
                v.starts.push(v.values.len() as u32);
            }
        }
        context_starts.push(contexts.len() as u32);
        Ok(DataIndex {
            kind: params.kind,
            outcome_labels,
            pred_labels,
            pred_counts,
            num_events: num_events as u32,
            outcomes,
            seen,
            context_starts,
            contexts,
            values,
        })
    }

    /// Which indexer made this.
    pub fn kind(&self) -> IndexerKind {
        self.kind
    }

    /// `getOutcomeLabels`: every outcome in order of first sight, those of
    /// dropped events included.
    pub fn outcome_labels(&self) -> &[JavaString] {
        &self.outcome_labels
    }

    /// `getPredLabels`: the predicates at or above the cutoff, in Java's
    /// string order; the position is the predicate id.
    pub fn pred_labels(&self) -> &[JavaString] {
        &self.pred_labels
    }

    /// `getPredCounts`: how often each kept predicate occurred, by id.
    pub fn pred_counts(&self) -> &[u32] {
        &self.pred_counts
    }

    /// `getNumEvents`: the events that had a predicate left, before
    /// merging.
    pub fn num_events(&self) -> u32 {
        self.num_events
    }

    /// The events after merging: the rows of the other per event getters.
    pub fn num_unique_events(&self) -> usize {
        self.outcomes.len()
    }

    /// `getOutcomeList()[i]`.
    pub fn outcome(&self, i: usize) -> u32 {
        self.outcomes[i]
    }

    /// `getOutcomeList`.
    pub fn outcomes(&self) -> &[u32] {
        &self.outcomes
    }

    /// `getNumTimesEventsSeen()[i]`: how many equal events were merged into
    /// this one; 1 everywhere when not sorted.
    pub fn seen(&self, i: usize) -> u32 {
        self.seen[i]
    }

    /// `getNumTimesEventsSeen`.
    pub fn times_seen(&self) -> &[u32] {
        &self.seen
    }

    /// `getContexts()[i]`: the predicate ids of unique event `i`, in the
    /// event's order, duplicates kept.
    pub fn context(&self, i: usize) -> &[u32] {
        &self.contexts[self.context_starts[i] as usize..self.context_starts[i + 1] as usize]
    }

    /// Whether `getValues()` is an array rather than null: only for
    /// `OnePassRealValue`.
    pub fn has_values(&self) -> bool {
        self.values.is_some()
    }

    /// The trainers' use of the values: `Ok` when the index drops values
    /// or every unique event has some, else [`Error::MissingValues`] for
    /// the first event without (the JVM's `NullPointerException`).
    pub fn check_values(&self) -> Result<()> {
        if self.values.is_some() {
            for i in 0..self.num_unique_events() {
                if self.values(i).is_none() {
                    return Err(Error::MissingValues { event: i });
                }
            }
        }
        Ok(())
    }

    /// `getValues()[i]`: the values of unique event `i`, of its full
    /// context (see the module documentation), or `None` when the indexer
    /// drops values, the event had none, or `i` is out of range.
    pub fn values(&self, i: usize) -> Option<&[f32]> {
        let v = self.values.as_ref()?;
        let start = *v.starts.get(i)? as usize;
        let end = *v.starts.get(i.checked_add(1)?)? as usize;
        let range = start..end;
        if range.is_empty() { None } else { Some(&v.values[range]) }
    }
}

/// What `TwoPassDataIndexer.computeEventCounts` would throw while writing
/// the event: `UTFDataFormatException` for a string over 65535 bytes.
fn check_write_utf(e: &Event) -> Result<()> {
    if e.outcome().len() > 65535 {
        return Err(Error::StringTooLong { what: "outcome", bytes: e.outcome().len() });
    }
    for c in e.context() {
        if c.len() > 65535 {
            return Err(Error::StringTooLong { what: "context", bytes: c.len() });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
