//! Annotators: typed steps that read layers and write layers.

use std::any::{Any, TypeId};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::doc::{Document, Status};
use crate::error::Error;
use crate::layer::{Anchor, Execution, Key, KeyId, Layer, LayerName, ModelRef, Provenance};

/// One input of an annotator: a layer it needs (`Tokens`), or a layer it
/// uses when the document has it (`Option<Sentences>`). An optional input
/// is declared like any other, so the analyzer orders the step after its
/// producer and provenance records whether it was there.
pub trait Slot: 'static {
    type Ref<'d>;
    const REQUIRED: bool;
    fn id(name: &'static str) -> KeyId;
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<Self::Ref<'d>, Error>;
}

impl<L: Layer> Slot for L {
    type Ref<'d> = &'d L;
    const REQUIRED: bool = true;
    fn id(name: &'static str) -> KeyId {
        Key::<L>::named(name).id()
    }
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<&'d L, Error> {
        doc.require(&Key::<L>::named(name))
    }
}

impl<L: Layer> Slot for Option<L> {
    type Ref<'d> = Option<&'d L>;
    const REQUIRED: bool = false;
    fn id(name: &'static str) -> KeyId {
        Key::<L>::named(name).id()
    }
    fn fetch<'d>(doc: &'d Document, name: &'static str) -> Result<Option<&'d L>, Error> {
        Ok(doc.get_key(&Key::<L>::named(name)))
    }
}

/// The layers an annotator reads, as a tuple of slots. The analyzer hands
/// it references of exactly those types: `(&Sentences, Option<&Speakers>)`.
pub trait Reads: 'static {
    type Refs<'d>;
    const N: usize;
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
    fn ids(names: &[&'static str]) -> Vec<KeyId>;
    fn anchors() -> Vec<Anchor>;
    fn store(self, doc: &mut Document, names: &[&'static str], p: &Provenance) -> Result<(), Error>;
}

impl Reads for () {
    type Refs<'d> = ();
    const N: usize = 0;
    fn ids(_: &[&'static str]) -> Vec<KeyId> {
        Vec::new()
    }
    fn required() -> Vec<bool> {
        Vec::new()
    }
    fn fetch(_: &Document, _: &[&'static str]) -> Result<(), Error> {
        Ok(())
    }
}

macro_rules! tuples {
    ($($L:ident $i:tt),+) => {
        impl<$($L: Slot),+> Reads for ($($L,)+) {
            type Refs<'d> = ($($L::Ref<'d>,)+);
            const N: usize = [$($i),+].len();
            fn ids(names: &[&'static str]) -> Vec<KeyId> {
                vec![$($L::id(names[$i])),+]
            }
            fn required() -> Vec<bool> {
                vec![$($L::REQUIRED),+]
            }
            fn fetch<'d>(doc: &'d Document, names: &[&'static str]) -> Result<Self::Refs<'d>, Error> {
                Ok(($($L::fetch(doc, names[$i])?,)+))
            }
        }
        impl<$($L: Layer),+> Writes for ($($L,)+) {
            const N: usize = [$($i),+].len();
            fn ids(names: &[&'static str]) -> Vec<KeyId> {
                vec![$(Key::<$L>::named(names[$i]).id()),+]
            }
            fn anchors() -> Vec<Anchor> {
                vec![$($L::anchor()),+]
            }
            fn store(self, doc: &mut Document, names: &[&'static str], p: &Provenance) -> Result<(), Error> {
                let ids = <Self as Writes>::ids(names);
                $({
                    let p = made_on_own(&self.$i, &ids[..$i], p);
                    doc.add(&Key::<$L>::named(names[$i]), self.$i, p)?;
                })+
                Ok(())
            }
        }
    };
}
/// The provenance of one output: its step's, with `on` listing the outputs
/// the step wrote before it (`earlier`) that this one is aligned to or
/// indexes, so the document checks it against those instances.
fn made_on_own<L: Layer>(layer: &L, earlier: &[KeyId], p: &Provenance) -> Provenance {
    let mut types: Vec<TypeId> = layer.indexes().into_iter().map(|(k, _)| k.ty).collect();
    if let Anchor::Aligned { ty, .. } = L::anchor() {
        types.push(ty);
    }
    let mut p = p.clone();
    p.on = earlier.iter().copied().filter(|k| types.contains(&k.ty)).collect();
    p
}

tuples!(A 0);
tuples!(A 0, B 1);
tuples!(A 0, B 1, C 2);
tuples!(A 0, B 1, C 2, D 3);
tuples!(A 0, B 1, C 2, D 3, E 4);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

pub type Refs<'d, A> = <<A as Annotator>::Reads as Reads>::Refs<'d>;

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

    /// The model it runs, recorded on every layer it writes.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Where that model runs: backend, device, precision.
    fn execution(&self) -> Option<Execution> {
        None
    }

    fn config_hash(&self) -> u64 {
        0
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

    /// The provenance of this step's layers on one document: its optional
    /// inputs are listed only when the document had them.
    fn provenance(&self, doc: &Document) -> Provenance {
        let reads = A::Reads::ids(&self.reads)
            .into_iter()
            .zip(A::Reads::required())
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
    fn anchors(&self) -> Vec<AnchorUse> {
        let reads: Vec<KeyId> = A::Reads::ids(&self.reads)
            .into_iter()
            .zip(A::Reads::required())
            .filter(|&(_, r)| r)
            .map(|(k, _)| k)
            .collect();
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
    fn skipped_input(&self, doc: &Document) -> Option<String> {
        A::Reads::ids(&self.reads).into_iter().zip(A::Reads::required()).filter(|&(_, r)| r).find_map(
            |(k, _)| match doc.status(k) {
                Some(Status::Skipped(why)) => Some(format!("{k} was skipped: {why}")),
                Some(Status::Failed(e)) => Some(format!("{k} failed: {e}")),
                _ => None,
            },
        )
    }

    /// An input this step would read beside the wrong instance of the
    /// layer it was made on: tags made on the gold tokens, beside the
    /// predicted ones. Reading that instance as well is fine (an
    /// evaluation compares the two).
    fn unaligned(&self, doc: &Document) -> Option<Error> {
        let reads = A::Reads::ids(&self.reads);
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
    fn run<'d>(&self, docs: &'d [Document], at: &[usize], inputs: Vec<Refs<'d, A>>) -> Vec<Result<A::Writes, Error>> {
        match catch_unwind(AssertUnwindSafe(|| self.annotator.annotate_batch(inputs))) {
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
    fn reads(&self) -> Vec<KeyId>;
    /// For each read, whether the step needs it or only uses it when there.
    fn required(&self) -> Vec<bool>;
    fn writes(&self) -> Vec<KeyId>;
    /// For each output, what it is aligned to.
    fn anchors(&self) -> Vec<AnchorUse>;
    fn config_hash(&self) -> u64 {
        0
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

    fn reads(&self) -> Vec<KeyId> {
        A::Reads::ids(&self.reads)
    }

    fn required(&self) -> Vec<bool> {
        A::Reads::required()
    }

    fn writes(&self) -> Vec<KeyId> {
        A::Writes::ids(&self.writes)
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        Bound::anchors(self)
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let mut out: Vec<Option<Result<A::Writes, Error>>> = (0..docs.len()).map(|_| None).collect();
        let mut inputs = Vec::new();
        let mut at = Vec::new();
        for (i, d) in docs.iter().enumerate() {
            if let Some(why) = self.skipped_input(d) {
                out[i] = Some(Err(Error::Skipped(why)));
                continue;
            }
            if let Some(e) = self.unaligned(d) {
                out[i] = Some(Err(e));
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
        let out = *out.downcast::<Vec<Option<Result<A::Writes, Error>>>>().expect("this step's own output");
        let writes = A::Writes::ids(&self.writes);
        for (i, r) in out.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let p = self.provenance(&docs[i]);
            let failure = match r {
                Ok(w) => w.store(&mut docs[i], &self.writes, &p).err(),
                Err(Error::Skipped(why)) => {
                    writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p.clone())).err()
                }
                Err(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            // Every output not stored is failed with the error, so the
            // document says why it is missing and what needs it is skipped.
            // Recording is refused only when another layer type holds the
            // id, and the document's result still says what failed.
            for &k in &writes {
                if !docs[i].has(k) {
                    let _ = docs[i].fail(k, e.clone(), p.clone());
                }
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}
