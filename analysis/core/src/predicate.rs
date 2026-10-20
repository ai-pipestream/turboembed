//! Predicate pipeline combinators: [`ConditionalAnnotator`] (branching or
//! skipping pipeline steps by a document predicate) and [`FilterAnnotator`]
//! (filtering items of a source layer into a distinct target layer instance).

mod filter;

pub use filter::{FilterAnnotator, FilterableLayer, SpanFilter, co_written_aligned_anchors, retain_aligned_anchor};

use std::any::Any;
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::annotate::{AnchorUse, Annotator, Bound, Step};
use crate::doc::{Document, Status};
use crate::error::Error;
use crate::layer::{Key, KeyId, Layer, Provenance};

pub(crate) const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

pub(crate) fn fnv_mix_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

pub(crate) fn fnv_mix_str(h: u64, s: &str) -> u64 {
    let h = fnv_mix_bytes(h, s.as_bytes());
    fnv_mix_bytes(h, &[0xff])
}

pub(crate) fn fnv_mix_u64(h: u64, v: u64) -> u64 {
    fnv_mix_bytes(h, &v.to_le_bytes())
}

pub(crate) fn fnv_mix_key(mut h: u64, k: KeyId) -> u64 {
    h = fnv_mix_str(h, &k.layer.to_string());
    fnv_mix_str(h, k.name)
}

pub(crate) fn panicked(annotator: &'static str, panic: Box<dyn Any + Send>) -> Error {
    let what = panic.downcast_ref::<&str>().copied().or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    Error::Annotator { annotator, detail: format!("panicked: {}", what.unwrap_or("with no message")) }
}

pub(crate) fn check_required_inputs(doc: &Document, reads: &[KeyId], required: &[bool]) -> Result<(), Error> {
    for (&k, &req) in reads.iter().zip(required) {
        if !req {
            continue;
        }
        match doc.status(k) {
            Some(Status::Skipped(why)) => return Err(Error::Skipped(format!("{k} was skipped: {why}"))),
            Some(Status::Failed(e)) => return Err(Error::Skipped(format!("{k} failed: {e}"))),
            _ => {}
        }
    }
    for (&k, &req) in reads.iter().zip(required) {
        if req && !doc.present(k) {
            return Err(Error::Missing(k));
        }
    }
    for &layer in reads {
        if !doc.present(layer) {
            continue;
        }
        for &on in doc.made_on(layer) {
            if !reads.contains(&on)
                && let Some(read) = reads.iter().copied().find(|&r| r.ty == on.ty && doc.present(r))
            {
                return Err(Error::Unaligned { layer, on, read: read.name });
            }
        }
    }
    Ok(())
}

type DocPredicate = Box<dyn Fn(&Document) -> bool + Send + Sync>;

/// Chooses whether to run `when_true` or `when_false` (or skip output layers
/// when `when_false` is `None`) according to a predicate over the [`Document`].
pub struct ConditionalAnnotator {
    predicate_name: &'static str,
    predicate: DocPredicate,
    when_true: Box<dyn Step>,
    when_false: Option<Box<dyn Step>>,
    reads: Vec<KeyId>,
    required: Vec<bool>,
    writes: Vec<KeyId>,
    anchors: Vec<AnchorUse>,
    skip_reason: Option<String>,
}

/// Alias for [`ConditionalAnnotator`].
pub type ConditionalStep = ConditionalAnnotator;

struct ConditionalBatch {
    true_indices: Vec<usize>,
    true_out: Option<Box<dyn Any + Send>>,
    false_indices: Vec<usize>,
    false_out: Option<Box<dyn Any + Send>>,
    non_branch: Vec<(usize, DocNonBranch)>,
}

enum DocNonBranch {
    SkipFalse,
    Skipped(String),
    Failed(Error),
}

impl ConditionalAnnotator {
    pub const ID: &'static str = "turbo:conditional";
    pub const VERSION: &'static str = "0";

    /// Creates a conditional step that runs `when_true` when `predicate` holds,
    /// and marks `when_true`'s output layers as [`Status::Skipped`] when false.
    pub fn new(
        predicate_name: &'static str,
        predicate: impl Fn(&Document) -> bool + Send + Sync + 'static,
        when_true: impl Step + 'static,
    ) -> ConditionalAnnotator {
        let reads = when_true.reads();
        let required = when_true.required();
        let writes = when_true.writes();
        let anchors = when_true.anchors();
        ConditionalAnnotator {
            predicate_name,
            predicate: Box::new(predicate),
            when_true: Box::new(when_true),
            when_false: None,
            reads,
            required,
            writes,
            anchors,
            skip_reason: None,
        }
    }

    /// Creates a conditional step from a default-bound [`Annotator`].
    pub fn from_annotator<A: Annotator>(
        predicate_name: &'static str,
        predicate: impl Fn(&Document) -> bool + Send + Sync + 'static,
        when_true: A,
    ) -> ConditionalAnnotator {
        Self::new(predicate_name, predicate, Bound::new(when_true))
    }

    /// Creates a conditional step with explicit `when_true` and `when_false`
    /// branches. Both branches must write the exact same set of output layers,
    /// and the conditional step requires the union of their required inputs.
    pub fn if_else(
        predicate_name: &'static str,
        predicate: impl Fn(&Document) -> bool + Send + Sync + 'static,
        when_true: impl Step + 'static,
        when_false: impl Step + 'static,
    ) -> Result<ConditionalAnnotator, Error> {
        let true_writes = when_true.writes();
        let false_writes = when_false.writes();
        let true_set: HashSet<KeyId> = true_writes.iter().copied().collect();
        let false_set: HashSet<KeyId> = false_writes.iter().copied().collect();
        if true_writes.len() != false_writes.len() || true_set != false_set {
            return Err(Error::Pipeline("branches must provide the same layers".into()));
        }

        let mut reads = Vec::new();
        let mut required = Vec::new();
        for (k, req) in when_true
            .reads()
            .into_iter()
            .zip(when_true.required())
            .chain(when_false.reads().into_iter().zip(when_false.required()))
        {
            if let Some(pos) = reads.iter().position(|&r| r == k) {
                required[pos] = required[pos] || req;
            } else {
                reads.push(k);
                required.push(req);
            }
        }

        let mut anchors = when_true.anchors();
        let false_anchors = when_false.anchors();
        for (fw, fa) in false_writes.iter().zip(false_anchors) {
            if matches!(fa, AnchorUse::Later(_) | AnchorUse::Unread(_))
                && let Some(pos) = true_writes.iter().position(|tw| tw == fw)
            {
                anchors[pos] = fa;
            }
        }

        Ok(ConditionalAnnotator {
            predicate_name,
            predicate: Box::new(predicate),
            when_true: Box::new(when_true),
            when_false: Some(Box::new(when_false)),
            reads,
            required,
            writes: true_writes,
            anchors,
            skip_reason: None,
        })
    }

    /// Creates an if/else conditional step from two default-bound [`Annotator`]s.
    pub fn if_else_annotators<T: Annotator, F: Annotator>(
        predicate_name: &'static str,
        predicate: impl Fn(&Document) -> bool + Send + Sync + 'static,
        when_true: T,
        when_false: F,
    ) -> Result<ConditionalAnnotator, Error> {
        Self::if_else(predicate_name, predicate, Bound::new(when_true), Bound::new(when_false))
    }

    /// Declares an input layer that the predicate itself requires on the document.
    pub fn require<L: Layer>(mut self, key: Key<L>) -> Self {
        let k = key.id();
        if let Some(pos) = self.reads.iter().position(|&r| r == k) {
            self.required[pos] = true;
        } else {
            self.reads.push(k);
            self.required.push(true);
        }
        self
    }

    /// Sets the skip reason recorded when `when_false` is `None` and the
    /// predicate evaluates to `false`.
    pub fn with_skip_reason(mut self, reason: impl Into<String>) -> Self {
        self.skip_reason = Some(reason.into());
        self
    }

    pub fn predicate_name(&self) -> &'static str {
        self.predicate_name
    }

    /// Deterministic hash combining the predicate identifier and the branch
    /// steps' identifiers, `config_hash` values, and ordered read/write bindings.
    pub fn config_hash(&self) -> u64 {
        fn hash_step(mut h: u64, step: &dyn Step) -> u64 {
            h = fnv_mix_str(h, step.id());
            h = fnv_mix_u64(h, step.config_hash());
            let reads = step.reads();
            let required = step.required();
            h = fnv_mix_u64(h, reads.len() as u64);
            for (slot, (&r, &req)) in reads.iter().zip(&required).enumerate() {
                h = fnv_mix_u64(h, slot as u64);
                h = fnv_mix_key(h, r);
                h = fnv_mix_bytes(h, &[u8::from(req)]);
            }
            let writes = step.writes();
            h = fnv_mix_u64(h, writes.len() as u64);
            for (slot, &w) in writes.iter().enumerate() {
                h = fnv_mix_u64(h, slot as u64);
                h = fnv_mix_key(h, w);
            }
            h
        }

        let mut h = fnv_mix_str(FNV_OFFSET_BASIS, Self::ID);
        h = fnv_mix_str(h, self.predicate_name);
        h = hash_step(h, self.when_true.as_ref());
        match &self.when_false {
            Some(wf) => {
                h = fnv_mix_bytes(h, &[1]);
                h = hash_step(h, wf.as_ref());
            }
            None => {
                h = fnv_mix_bytes(h, &[0]);
            }
        }
        for (&r, &req) in self.reads.iter().zip(&self.required) {
            h = fnv_mix_key(h, r);
            h = fnv_mix_bytes(h, &[u8::from(req)]);
        }
        for &w in &self.writes {
            h = fnv_mix_key(h, w);
        }
        h
    }

    /// Runs this conditional step on a single document.
    pub fn annotate_doc(&self, doc: &mut Document) -> Result<(), Error> {
        let out = self.compute(std::slice::from_ref(doc));
        let mut status = [Ok(())];
        self.store(out, std::slice::from_mut(doc), &mut status);
        status.into_iter().next().unwrap()
    }

    fn present_reads(&self, doc: &Document) -> Vec<KeyId> {
        self.reads
            .iter()
            .copied()
            .zip(self.required.iter().copied())
            .filter(|&(k, req)| req || doc.present(k))
            .map(|(k, _)| k)
            .collect()
    }

    fn base_provenance(&self, doc: &Document) -> Provenance {
        Provenance {
            annotator: Self::ID,
            version: Self::VERSION,
            config: self.config_hash(),
            model: None,
            execution: None,
            reads: self.present_reads(doc),
            on: Vec::new(),
        }
    }
}

impl Step for ConditionalAnnotator {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        self.reads.clone()
    }

    fn required(&self) -> Vec<bool> {
        self.required.clone()
    }

    fn writes(&self) -> Vec<KeyId> {
        self.writes.clone()
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        self.anchors.clone()
    }

    fn config_hash(&self) -> u64 {
        ConditionalAnnotator::config_hash(self)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let mut true_indices = Vec::new();
        let mut true_docs = Vec::new();
        let mut false_indices = Vec::new();
        let mut false_docs = Vec::new();
        let mut non_branch = Vec::new();

        for (i, d) in docs.iter().enumerate() {
            if let Err(e) = check_required_inputs(d, &self.reads, &self.required) {
                let nb = match e {
                    Error::Skipped(why) => DocNonBranch::Skipped(why),
                    other => DocNonBranch::Failed(other),
                };
                non_branch.push((i, nb));
                continue;
            }
            match catch_unwind(AssertUnwindSafe(|| (self.predicate)(d))) {
                Ok(true) => {
                    true_indices.push(i);
                    true_docs.push(d.clone());
                }
                Ok(false) => {
                    if self.when_false.is_some() {
                        false_indices.push(i);
                        false_docs.push(d.clone());
                    } else {
                        non_branch.push((i, DocNonBranch::SkipFalse));
                    }
                }
                Err(panic) => non_branch.push((i, DocNonBranch::Failed(panicked(Self::ID, panic)))),
            }
        }

        let true_out = (!true_docs.is_empty()).then(|| self.when_true.compute(&true_docs));
        let false_out =
            self.when_false.as_ref().and_then(|wf| (!false_docs.is_empty()).then(|| wf.compute(&false_docs)));

        Box::new(ConditionalBatch { true_indices, true_out, false_indices, false_out, non_branch })
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let batch = *out.downcast::<ConditionalBatch>().expect("ConditionalAnnotator's own output");
        let cfg_hash = self.config_hash();
        let default_skip =
            self.skip_reason.clone().unwrap_or_else(|| format!("predicate {} evaluated to false", self.predicate_name));

        let mut store_branch = |step: &dyn Step, indices: &[usize], sub_out: Box<dyn Any + Send>| {
            let mut sub_docs: Vec<Document> = indices.iter().map(|&i| docs[i].clone()).collect();
            let pre_existing: Vec<Vec<bool>> =
                sub_docs.iter().map(|d| self.writes.iter().map(|&k| d.has(k)).collect()).collect();
            let p_reads: Vec<Vec<KeyId>> = sub_docs.iter().map(|d| self.present_reads(d)).collect();
            let p_bases: Vec<Provenance> = sub_docs.iter().map(|d| self.base_provenance(d)).collect();
            let mut sub_status: Vec<Result<(), Error>> = (0..indices.len()).map(|_| Ok(())).collect();
            step.store(sub_out, &mut sub_docs, &mut sub_status);
            for (k_idx, &doc_idx) in indices.iter().enumerate() {
                for (&w, &existed) in self.writes.iter().zip(&pre_existing[k_idx]) {
                    if !existed {
                        sub_docs[k_idx].update_provenance(w, |p| {
                            p.config = cfg_hash;
                            for &r in &p_reads[k_idx] {
                                if !p.reads.contains(&r) {
                                    p.reads.push(r);
                                }
                            }
                        });
                    }
                }
                if let Err(e) = sub_status[k_idx].clone() {
                    for &w in &self.writes {
                        if !sub_docs[k_idx].has(w) {
                            let _ = sub_docs[k_idx].fail(w, e.clone(), p_bases[k_idx].clone());
                        }
                    }
                    if status[doc_idx].is_ok() {
                        status[doc_idx] = Err(e);
                    }
                }
                docs[doc_idx] = sub_docs[k_idx].clone();
            }
        };

        if let Some(true_out) = batch.true_out {
            store_branch(self.when_true.as_ref(), &batch.true_indices, true_out);
        }
        if let Some(false_out) = batch.false_out {
            store_branch(self.when_false.as_deref().expect("when_false present"), &batch.false_indices, false_out);
        }

        for (i, nb) in batch.non_branch {
            let p_base = self.base_provenance(&docs[i]);
            let failure = match nb {
                DocNonBranch::SkipFalse => {
                    self.writes.iter().try_for_each(|&k| docs[i].skip(k, default_skip.clone(), p_base.clone())).err()
                }
                DocNonBranch::Skipped(why) => {
                    self.writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p_base.clone())).err()
                }
                DocNonBranch::Failed(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            for &k in &self.writes {
                if !docs[i].has(k) {
                    let _ = docs[i].fail(k, e.clone(), p_base.clone());
                }
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}

#[cfg(test)]
mod tests;
