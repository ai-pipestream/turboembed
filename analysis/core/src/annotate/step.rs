//! The type-erased step the analyzer runs, and what a step says about
//! where its outputs come from.

use std::any::Any;

use crate::doc::Document;
use crate::error::Error;
use crate::layer::{Execution, KeyId, LayerName, ModelRef, Provenance};
use crate::splice::LayerSplicer;

use super::{Annotator, Bound, Placed, Reads, Writes};

/// What a step's outputs depend on. A step that is a pure function of one
/// item of a partition layer at a time (the tokens of one sentence, the
/// tags of those tokens) says so, and `Analyzer::run_spliced` reuses its
/// outputs on the partitions an edit did not touch. The analyzer trusts
/// the declaration only as far as the pipeline bears it out: a step whose
/// inputs come from a document-wide step is treated as document-wide.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Locality {
    /// The output may depend on the whole document, or on windows that
    /// cross partitions (an encoder's context windows).
    Document,
    /// The output for each item of the partition layer is a pure function
    /// of the inputs inside that item.
    Partition(KeyId),
}

/// What an output is aligned to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AnchorUse {
    /// Not an aligned layer.
    None,
    /// Aligned to this input, or to this output of the same step, written
    /// before the aligned one.
    To(KeyId),
    /// Aligned to an output the step writes after the aligned one: the
    /// document could not measure it yet, so it is refused when the
    /// pipeline is built.
    Later(KeyId),
    /// Aligned to a layer the step does not read: refused when the
    /// pipeline is built.
    Unread(&'static LayerName),
}

/// A bound annotator with its types erased, for the analyzer and for
/// pipelines assembled from configuration. Running is split in two: the
/// computation reads the documents and can run beside other steps; the
/// store adds the results.
pub trait Step: Send + Sync {
    fn id(&self) -> &'static str;
    fn version(&self) -> &'static str {
        "0"
    }
    fn reads(&self) -> Vec<KeyId>;
    /// For each read, whether the step needs it or only uses it when there.
    fn required(&self) -> Vec<bool>;
    fn writes(&self) -> Vec<KeyId>;
    /// For each output, what it is aligned to.
    fn anchors(&self) -> Vec<AnchorUse>;
    fn requires(&self) -> Vec<KeyId> {
        self.reads().into_iter().zip(self.required()).filter_map(|(k, req)| req.then_some(k)).collect()
    }
    fn produces(&self) -> Vec<KeyId> {
        self.writes()
    }
    fn model(&self) -> Option<ModelRef> {
        None
    }
    fn execution(&self) -> Option<Execution> {
        None
    }
    fn config_hash(&self) -> u64 {
        0
    }
    fn is_cacheable(&self) -> bool {
        true
    }
    /// What the outputs depend on, as the step declares it; the analyzer
    /// decides what the pipeline bears out.
    fn locality(&self) -> Locality {
        Locality::Document
    }
    /// For each output, how to splice its items across an edit, when its
    /// layer type supports that (`Layer::splicer`). Empty when the step
    /// cannot say: it is then re-run whole.
    fn splicers(&self) -> Vec<Option<Box<dyn LayerSplicer>>> {
        Vec::new()
    }
    fn expected_provenance(&self, doc: &Document) -> Provenance {
        let reads = self
            .reads()
            .into_iter()
            .zip(self.required())
            .filter(|&(k, required)| required || doc.present(k))
            .map(|(k, _)| k)
            .collect();
        Provenance {
            annotator: self.id(),
            version: self.version(),
            config: self.config_hash(),
            model: self.model(),
            execution: self.execution(),
            reads,
            on: Vec::new(),
        }
    }
    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send>;
    /// Adds what `compute` made. An output the step did not deliver on a
    /// document is recorded as skipped or failed there, and the first
    /// failure is that document's result.
    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]);
}

impl<A: Annotator> Step for Bound<A> {
    fn id(&self) -> &'static str {
        A::ID
    }

    fn version(&self) -> &'static str {
        A::VERSION
    }

    fn reads(&self) -> Vec<KeyId> {
        self.effective_reads().0
    }

    fn required(&self) -> Vec<bool> {
        self.effective_reads().1
    }

    fn writes(&self) -> Vec<KeyId> {
        self.effective_writes()
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        Bound::anchors(self)
    }

    fn model(&self) -> Option<ModelRef> {
        self.annotator.model()
    }

    fn execution(&self) -> Option<Execution> {
        self.annotator.execution()
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn is_cacheable(&self) -> bool {
        self.annotator.is_cacheable()
    }

    /// The declared partition layer, as the instance this step reads: a
    /// tagger bound to the gold sentences partitions by those.
    fn locality(&self) -> Locality {
        match self.annotator.locality() {
            Locality::Document => Locality::Document,
            Locality::Partition(p) => {
                let (reads, _) = self.effective_reads();
                Locality::Partition(reads.into_iter().find(|r| r.ty == p.ty).unwrap_or(p))
            }
        }
    }

    fn splicers(&self) -> Vec<Option<Box<dyn LayerSplicer>>> {
        A::Writes::splicers()
    }

    fn expected_provenance(&self, doc: &Document) -> Provenance {
        self.provenance(doc)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let mut out: Vec<Option<Placed<A::Writes>>> = (0..docs.len()).map(|_| None).collect();
        let mut inputs = Vec::new();
        let mut at = Vec::new();
        let (eff_reads, eff_reqs) = self.effective_reads();
        for (i, d) in docs.iter().enumerate() {
            if let Some(why) = self.skipped_input(d) {
                out[i] = Some(Err(Error::Skipped(why)));
                continue;
            }
            if let Some(e) = self.unaligned(d) {
                out[i] = Some(Err(e));
                continue;
            }
            if let Some((missing_key, _)) =
                eff_reads.iter().copied().zip(eff_reqs.iter().copied()).find(|&(k, req)| req && !d.present(k))
            {
                out[i] = Some(Err(Error::Missing(missing_key)));
                continue;
            }
            match A::Reads::fetch(d, &self.reads) {
                Ok(r) => {
                    inputs.push(r);
                    at.push(i);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        for (&i, r) in at.iter().zip(self.run(docs, &at, inputs)) {
            out[i] = Some(r);
        }
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Option<Placed<A::Writes>>>>().expect("this step's own output");
        let writes = self.effective_writes();
        for (i, r) in out.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let mut p = self.provenance(&docs[i]);
            let failure = match r {
                Ok((w, placed)) => {
                    if placed.is_some() {
                        p.execution = placed;
                    }
                    w.store(&mut docs[i], &self.writes, &p).err()
                }
                Err(Error::Skipped(why)) => {
                    writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p.clone())).err()
                }
                Err(e) => Some(e),
            };
            if let Some(e) = failure {
                for &k in &writes {
                    if !docs[i].has(k) {
                        let _ = docs[i].fail(k, e.clone(), p.clone());
                    }
                }
                if status[i].is_ok() {
                    status[i] = Err(e);
                }
            }
            for &k in &writes {
                docs[i].mark_pipeline_stored(k);
            }
        }
    }
}
