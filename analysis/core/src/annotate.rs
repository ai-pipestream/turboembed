//! Annotators: typed steps that read layers and write layers.

#[path = "annotate/step.rs"]
mod step;
#[path = "annotate/tuples.rs"]
mod tuples;

pub use step::{AnchorUse, Locality, Step};
pub(crate) use tuples::made_on_own;

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::doc::{Document, Status};
use crate::error::Error;
use crate::layer::{Anchor, Execution, KeyId, ModelRef, Provenance};
use crate::splice::LayerSplicer;

/// One input of an annotator: a layer it needs (`Tokens`), or a layer it
/// uses when the document has it (`Option<Sentences>`). An optional input
/// is declared like any other, so the analyzer orders the step after its
/// producer and provenance records whether it was there.
pub trait Slot: 'static {
    type Ref<'d>;
    const REQUIRED: bool;
    const DEFAULT_ID: KeyId;
    fn id(name: &'static str) -> KeyId;
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<Self::Ref<'d>, Error>;
}

/// The layers an annotator reads, as a tuple of slots. The analyzer hands
/// it references of exactly those types: `(&Sentences, Option<&Speakers>)`.
pub trait Reads: 'static {
    type Refs<'d>;
    const N: usize;
    const ALL_IDS: &'static [KeyId] = &[];
    const REQUIRED_FLAGS: &'static [bool] = &[];
    const DEFAULT_REQUIRED_IDS: &'static [KeyId] = &[];
    fn ids(names: &[&'static str]) -> Vec<KeyId>;
    fn required() -> Vec<bool>;
    fn fetch<'d>(doc: &'d Document, names: &[&'static str]) -> Result<Self::Refs<'d>, Error>;
}

/// The layers an annotator writes, as a tuple of owned layers. They are
/// added to the document in tuple order, so an output aligned to another
/// output of the same step (tokens, then a confidence for each) follows
/// it, and is made on the instance the step wrote.
pub trait Writes: Send + 'static {
    const N: usize;
    const DEFAULT_IDS: &'static [KeyId] = &[];
    fn ids(names: &[&'static str]) -> Vec<KeyId>;
    fn anchors() -> Vec<Anchor>;
    /// For each output, its layer type's splicer (`Layer::splicer`).
    fn splicers() -> Vec<Option<Box<dyn LayerSplicer>>>;
    fn store(self, doc: &mut Document, names: &[&'static str], p: &Provenance) -> Result<(), Error>;
}

pub type Refs<'d, A> = <<A as Annotator>::Reads as Reads>::Refs<'d>;

/// A document's outputs and, when the step says, where its model ran for
/// them (`Annotator::annotate_batch_placed`).
pub type Placed<W> = Result<(W, Option<Execution>), Error>;

/// One analysis step. Its inputs and outputs are types, so a tagger that
/// says it reads `(Sentences, Tokens)` and writes `(PosTags,)` is handed
/// `(&Sentences, &Tokens)` and must return a `PosTags`; the compiler holds
/// it to that. It is handed nothing else, not the document, so it cannot
/// read a layer it did not declare, and its provenance is complete. Text
/// is read the same way: a `View` is a layer.
pub trait Annotator: Send + Sync + 'static {
    const ID: &'static str;
    const VERSION: &'static str = "0";
    type Reads: Reads;
    type Writes: Writes;

    /// One document's inputs. Returning `Error::Skipped` declines the
    /// document with a reason.
    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error>;

    /// The inputs of a whole batch of documents at once, a result for
    /// each. The default runs `annotate` on each; a neural annotator packs
    /// every document's tokens into one device run here, which is where its
    /// speed comes from.
    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        inputs.into_iter().map(|i| self.annotate(i)).collect()
    }

    /// `annotate_batch`, with where each document's model ran when that
    /// can differ from call to call: a step that sends a call to one
    /// device or another returns the execution it used, and that
    /// document's layers record it in place of `execution`. One
    /// document's layers come from one execution. The default runs
    /// `annotate_batch` and returns None, which records `execution`.
    fn annotate_batch_placed<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Placed<Self::Writes>> {
        self.annotate_batch(inputs).into_iter().map(|r| r.map(|w| (w, None))).collect()
    }

    /// Upstream layer keys required by this annotator. Defaults to the
    /// required slots of [`Self::Reads`].
    fn requires(&self) -> &[KeyId] {
        <Self::Reads as Reads>::DEFAULT_REQUIRED_IDS
    }

    /// Output layer keys produced by this annotator. Defaults to
    /// [`Self::Writes`].
    fn produces(&self) -> &[KeyId] {
        <Self::Writes as Writes>::DEFAULT_IDS
    }

    /// The model it runs, recorded on every layer it writes.
    ///
    /// [`Analyzer::analyze_incremental`](crate::Analyzer::analyze_incremental) and
    /// [`Analyzer::run_incremental`](crate::Analyzer::run_incremental) include
    /// [`Self::model`] in the provenance digest of every produced layer.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Where that model runs: backend, device, precision.
    fn execution(&self) -> Option<Execution> {
        None
    }

    /// Deterministic configuration fingerprint recorded in [`Provenance::config`].
    ///
    /// [`Analyzer::analyze_incremental`](crate::Analyzer::analyze_incremental) and
    /// [`Analyzer::run_incremental`](crate::Analyzer::run_incremental) key their
    /// provenance-skip cache on `(ID, VERSION, config_hash(), model())`. Any
    /// annotator whose output depends on runtime configuration or weights must
    /// either incorporate all behavior-affecting parameters into [`Self::config_hash`]
    /// / [`Self::model`] or return `false` from [`Self::is_cacheable`].
    fn config_hash(&self) -> u64 {
        0
    }

    /// Whether [`Analyzer::analyze_incremental`](crate::Analyzer::analyze_incremental)
    /// and [`Analyzer::run_incremental`](crate::Analyzer::run_incremental) may skip
    /// this annotator on a provenance cache hit. Defaults to `true`.
    ///
    /// Annotators wrapping opaque closures or external state not captured by
    /// [`Self::config_hash`] and [`Self::model`] should return `false` so they are
    /// always executed during incremental runs.
    fn is_cacheable(&self) -> bool {
        true
    }

    /// What the outputs depend on: the whole document (the default), or
    /// one item of a partition layer at a time. `Locality::Partition` is a
    /// promise that the output for each item of that layer is a pure
    /// function of the inputs inside the item, so
    /// [`Analyzer::run_spliced`](crate::Analyzer::run_spliced) may reuse
    /// it on the items an edit did not touch. The partition layer is named
    /// by its default key; the step's bound instance is used.
    fn locality(&self) -> Locality {
        Locality::Document
    }
}

/// An annotator bound to the layer instances it reads and writes: by
/// default the default instances, or named ones (`gold`, `nfkc`).
pub struct Bound<A: Annotator> {
    annotator: A,
    reads: Vec<&'static str>,
    writes: Vec<&'static str>,
}

impl<A: Annotator> Bound<A> {
    pub fn new(annotator: A) -> Bound<A> {
        Bound { annotator, reads: vec![""; A::Reads::N], writes: vec![""; A::Writes::N] }
    }

    /// Reads input `slot` from the instance called `name`.
    pub fn read(mut self, slot: usize, name: &'static str) -> Bound<A> {
        self.reads[slot] = name;
        self
    }

    /// Writes output `slot` as the instance called `name`.
    pub fn write(mut self, slot: usize, name: &'static str) -> Bound<A> {
        self.writes[slot] = name;
        self
    }

    pub(super) fn effective_reads(&self) -> (Vec<KeyId>, Vec<bool>) {
        let mut ids = A::Reads::ids(&self.reads);
        let mut reqs = A::Reads::required();
        for &r in self.annotator.requires() {
            if !ids.iter().any(|k| k.ty == r.ty && (k.name == r.name || r.name.is_empty())) {
                ids.push(r);
                reqs.push(true);
            }
        }
        (ids, reqs)
    }

    pub(super) fn effective_writes(&self) -> Vec<KeyId> {
        let mut ids = A::Writes::ids(&self.writes);
        for &w in self.annotator.produces() {
            if !ids.iter().any(|k| k.ty == w.ty && (k.name == w.name || w.name.is_empty())) {
                ids.push(w);
            }
        }
        ids
    }

    /// The provenance of this step's layers on one document: its optional
    /// inputs are listed only when the document had them.
    pub(super) fn provenance(&self, doc: &Document, ids: &[KeyId], reqs: &[bool]) -> Provenance {
        let reads = ids
            .iter()
            .copied()
            .zip(reqs.iter().copied())
            .filter(|&(k, required)| required || doc.present(k))
            .map(|(k, _)| k)
            .collect();
        Provenance {
            annotator: A::ID,
            version: A::VERSION,
            config: self.annotator.config_hash(),
            model: self.annotator.model(),
            execution: self.annotator.execution(),
            reads,
            on: Vec::new(),
        }
    }

    /// Each aligned output hangs on an output of its anchor's type that the
    /// step writes before it, or else on the required input of that type.
    /// The document resolves the anchor in the same order.
    pub(super) fn anchors(&self) -> Vec<AnchorUse> {
        let (read_ids, read_reqs) = self.effective_reads();
        let reads: Vec<KeyId> = read_ids.into_iter().zip(read_reqs).filter(|&(_, r)| r).map(|(k, _)| k).collect();
        let writes = A::Writes::ids(&self.writes);
        A::Writes::anchors()
            .into_iter()
            .enumerate()
            .map(|(i, a)| match a {
                Anchor::Aligned { ty, layer } => {
                    let of_type = |keys: &[KeyId]| keys.iter().copied().find(|k| k.ty == ty);
                    if let Some(k) = of_type(&writes[..i]).or_else(|| of_type(&reads)) {
                        AnchorUse::To(k)
                    } else if let Some(k) = of_type(&writes[i + 1..]) {
                        AnchorUse::Later(k)
                    } else {
                        AnchorUse::Unread(layer)
                    }
                }
                _ => AnchorUse::None,
            })
            .collect()
    }

    /// The first required input that was skipped or failed on this
    /// document, and why.
    pub(super) fn skipped_input(doc: &Document, ids: &[KeyId], reqs: &[bool]) -> Option<String> {
        ids.iter().copied().zip(reqs.iter().copied()).filter(|&(_, r)| r).find_map(|(k, _)| match doc.status(k) {
            Some(Status::Skipped(why)) => Some(format!("{k} was skipped: {why}")),
            Some(Status::Failed(e)) => Some(format!("{k} failed: {e}")),
            _ => None,
        })
    }

    /// An input this step would read beside the wrong instance of the
    /// layer it was made on: tags made on the gold tokens, beside the
    /// predicted ones. Reading that instance as well is fine (an
    /// evaluation compares the two).
    pub(super) fn unaligned(doc: &Document, reads: &[KeyId]) -> Option<Error> {
        reads.iter().find_map(|&layer| {
            doc.made_on(layer).iter().filter(|on| !reads.contains(on)).find_map(|&on| {
                let read = reads.iter().copied().find(|&r| r.ty == on.ty && doc.present(r))?;
                Some(Error::Unaligned { layer, on, read: read.name })
            })
        })
    }

    /// Runs the annotator over the inputs of documents `at`, a result for
    /// each. A panic fails only the documents that cause it: a batch that
    /// panics is run again one document at a time to find them. The
    /// annotator is assumed to hold nothing a panic could leave half
    /// changed.
    pub(super) fn run<'d>(
        &self,
        docs: &'d [Document],
        at: &[usize],
        inputs: Vec<Refs<'d, A>>,
    ) -> Vec<Placed<A::Writes>> {
        match catch_unwind(AssertUnwindSafe(|| self.annotator.annotate_batch_placed(inputs))) {
            Ok(out) if out.len() == at.len() => out,
            Ok(out) => {
                let detail = format!("{} results for {} documents", out.len(), at.len());
                at.iter().map(|_| Err(Error::Annotator { annotator: A::ID, detail: detail.clone() })).collect()
            }
            Err(panic) if at.len() == 1 => vec![Err(panicked(A::ID, panic))],
            Err(_) => at
                .iter()
                .flat_map(|&i| match A::Reads::fetch(&docs[i], &self.reads) {
                    Ok(input) => self.run(docs, &[i], vec![input]),
                    Err(e) => vec![Err(e)],
                })
                .collect(),
        }
    }
}

fn panicked(annotator: &'static str, panic: Box<dyn Any + Send>) -> Error {
    let what = panic.downcast_ref::<&str>().copied().or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    Error::Annotator { annotator, detail: format!("panicked: {}", what.unwrap_or("with no message")) }
}
