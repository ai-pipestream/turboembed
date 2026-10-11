//! Annotators: typed steps that read layers and write layers.

use std::any::Any;

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

/// The layers an annotator writes, as a tuple of owned layers.
pub trait Writes: Send + 'static {
    const N: usize;
    fn ids(names: &[&'static str]) -> Vec<KeyId>;
    fn anchors() -> Vec<Anchor>;
    fn store(
        self,
        doc: &mut Document,
        names: &[&'static str],
        anchors: &[Option<KeyId>],
        p: &Provenance,
    ) -> Result<(), Error>;
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
            fn store(self, doc: &mut Document, names: &[&'static str], anchors: &[Option<KeyId>], p: &Provenance) -> Result<(), Error> {
                $(doc.add_anchored(&Key::<$L>::named(names[$i]), self.$i, p.clone(), anchors[$i])?;)+
                Ok(())
            }
        }
    };
}
tuples!(A 0);
tuples!(A 0, B 1);
tuples!(A 0, B 1, C 2);
tuples!(A 0, B 1, C 2, D 3);
tuples!(A 0, B 1, C 2, D 3, E 4);
tuples!(A 0, B 1, C 2, D 3, E 4, F 5);

pub type Refs<'d, A> = <<A as Annotator>::Reads as Reads>::Refs<'d>;

/// One analysis step. Its inputs and outputs are types, so a tagger that
/// says it reads `(Sentences, Tokens)` and writes `(PosTags,)` is handed
/// `(&Sentences, &Tokens)` and must return a `PosTags`; the compiler holds
/// it to that.
pub trait Annotator: Send + Sync + 'static {
    const ID: &'static str;
    const VERSION: &'static str = "0";
    type Reads: Reads;
    type Writes: Writes;

    /// One document. Returning `Error::Skipped` declines it with a reason.
    fn annotate(&self, doc: &Document, input: Refs<'_, Self>) -> Result<Self::Writes, Error>;

    /// A whole batch of documents at once. The default runs `annotate` on
    /// each; a neural annotator packs every document's tokens into one
    /// device run here, which is where its speed comes from.
    fn annotate_batch<'d>(
        &self,
        docs: &[&'d Document],
        inputs: Vec<Refs<'d, Self>>,
    ) -> Vec<Result<Self::Writes, Error>> {
        docs.iter().zip(inputs).map(|(d, i)| self.annotate(d, i)).collect()
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
        }
    }

    /// Each aligned output hangs on the required input of its anchor's type.
    fn anchors(&self) -> Vec<AnchorUse> {
        let reads: Vec<KeyId> = A::Reads::ids(&self.reads)
            .into_iter()
            .zip(A::Reads::required())
            .filter(|&(_, r)| r)
            .map(|(k, _)| k)
            .collect();
        A::Writes::anchors()
            .into_iter()
            .map(|a| match a {
                Anchor::Aligned { ty, layer } => {
                    reads.iter().copied().find(|k| k.ty == ty).map_or(AnchorUse::Unread(layer), AnchorUse::To)
                }
                _ => AnchorUse::None,
            })
            .collect()
    }

    /// The first required input that was skipped on this document, and why.
    fn skipped_input(&self, doc: &Document) -> Option<String> {
        A::Reads::ids(&self.reads).into_iter().zip(A::Reads::required()).filter(|&(_, r)| r).find_map(
            |(k, _)| match doc.status(k) {
                Some(Status::Skipped(why)) => Some(format!("{k} was skipped: {why}")),
                _ => None,
            },
        )
    }
}

/// What an output is aligned to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AnchorUse {
    /// Not an aligned layer.
    None,
    /// Aligned to this input.
    To(KeyId),
    /// Aligned to a layer the step does not read: refused when the
    /// pipeline is built.
    Unread(&'static LayerName),
}

impl AnchorUse {
    fn key(self) -> Option<KeyId> {
        match self {
            AnchorUse::To(k) => Some(k),
            _ => None,
        }
    }
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
    fn compute(&self, docs: &[Document], live: &[bool]) -> Box<dyn Any + Send>;
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

    fn compute(&self, docs: &[Document], live: &[bool]) -> Box<dyn Any + Send> {
        let mut out: Vec<Option<Result<A::Writes, Error>>> = (0..docs.len()).map(|_| None).collect();
        let mut batch = Vec::new();
        let mut inputs = Vec::new();
        let mut at = Vec::new();
        for (i, d) in docs.iter().enumerate().filter(|(i, _)| live[*i]) {
            if let Some(why) = self.skipped_input(d) {
                out[i] = Some(Err(Error::Skipped(why)));
                continue;
            }
            match A::Reads::fetch(d, &self.reads) {
                Ok(r) => {
                    batch.push(d);
                    inputs.push(r);
                    at.push(i);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        for (i, r) in at.into_iter().zip(self.annotator.annotate_batch(&batch, inputs)) {
            out[i] = Some(r);
        }
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Option<Result<A::Writes, Error>>>>().expect("this step's own output");
        let anchors: Vec<Option<KeyId>> = Bound::anchors(self).into_iter().map(AnchorUse::key).collect();
        let writes = A::Writes::ids(&self.writes);
        for (i, r) in out.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let p = self.provenance(&docs[i]);
            let stored = match r {
                Ok(w) => w.store(&mut docs[i], &self.writes, &anchors, &p),
                Err(Error::Skipped(why)) => writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p.clone())),
                Err(e) => Err(e),
            };
            if let Err(e) = stored {
                status[i] = Err(e);
            }
        }
    }
}
