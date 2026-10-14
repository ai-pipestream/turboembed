//! What a layer is, how a document names one, and where it came from.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::span::Overlap;
use crate::text::ViewId;

/// A closed set of labels (a POS tag set, entity kinds, dependency
/// relations, edge kinds), interned: a layer stores a u16 per item,
/// never a string.
#[derive(Debug, Default)]
pub struct TagSet {
    pub name: String,
    names: Vec<Box<str>>,
    index: HashMap<Box<str>, u16>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub u16);

impl TagSet {
    pub fn new(name: &str, labels: &[&str]) -> Arc<TagSet> {
        let names: Vec<Box<str>> = labels.iter().map(|&l| l.into()).collect();
        let index = names.iter().enumerate().map(|(i, n)| (n.clone(), i as u16)).collect();
        Arc::new(TagSet { name: name.into(), names, index })
    }

    pub fn tag(&self, label: &str) -> Option<Tag> {
        self.index.get(label).map(|&i| Tag(i))
    }

    pub fn label(&self, t: Tag) -> &str {
        &self.names[t.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// A layer: one typed, columnar result over a document. Each layer is its
/// own Rust type (Sentences, Tokens, PosTags, Dependencies, Chunks...),
/// so reading one gives that type, with its own fields, not a list of
/// generic annotations. Add-ons define new layer types in their own
/// crates; the container never enumerates them.
pub trait Layer: Any + Send + Sync + Sized {
    /// The stable, namespaced id the wire and provenance use. A shared
    /// layer the OpenNLP gRPC standard defines has the standard's id
    /// ("opennlp:tokens", "opennlp:pos"); an add-on's own layers are in
    /// its own namespace ("acme:invoice-fields").
    const ID: &'static str;
    /// Bumped when what the layer's fields mean changes.
    const SCHEMA: u16 = 1;
    /// For a layer generic over another (`Vectors<Sentences>`,
    /// `Groups<Tokens>`), that layer's name, or for a layer generic over
    /// two (`Edges<From, To>`), the first one's, so distinct instantiations
    /// have distinct names. It is the inner layer's full name, so nested
    /// generic layers stay distinct too: `Groups<Edges<Entities, Tokens>>`
    /// is `turbo:groups<turbo:edges<opennlp:entities, opennlp:tokens>>`.
    const OF: Option<&'static LayerName> = None;
    /// For a layer generic over two layers (`Edges<From, To>`), the second
    /// one's name, so `Edges<Entities, Tokens>` is
    /// `turbo:edges<opennlp:entities, opennlp:tokens>`. Set only with `OF`.
    const TO: Option<&'static LayerName> = None;
    /// The static name a `KeyId` points at.
    const NAME: &'static LayerName = &LayerName { id: Self::ID, of: Self::OF, to: Self::TO };
    /// What overlap the layer's spans may have when anchored to text.
    const OVERLAP: Overlap = Overlap::None;

    /// What the layer's items are anchored to.
    fn anchor() -> Anchor {
        Anchor::Text
    }

    /// How many items it holds.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// For a layer anchored to text: the view its spans are in and the
    /// span columns, `[start]` and `[end]`, so the container can check
    /// every offset before it takes the layer.
    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        None
    }

    /// For a layer anchored to text that holds discontinuous items: the
    /// view and the fragment columns, `[frag_start]` and `[frag_end]`, so
    /// the container and offset-invariant checker check every fragment's
    /// bounds and character boundaries.
    fn fragment_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        None
    }

    /// The other layers its items point into by index, besides an anchor:
    /// a token's sentence, an entity's tokens, an edge's endpoints. Each is
    /// named by its key (`Key::DEFAULT` unless the layer names a specific
    /// instance), with one past the largest index held. The document checks
    /// that every index is inside that layer, and records which instance of
    /// it the layer was made on, as it does an anchor.
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        Vec::new()
    }

    /// Invariants only the layer knows: tags inside their tag set, heads
    /// inside their sentence, columns of one length.
    fn check(&self) -> Result<(), String> {
        Ok(())
    }
}

/// What a layer's items hang on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// Each item carries a span of one text view.
    Text,
    /// Item i describes item i of another layer: a tag per token, a vector
    /// per sentence. The container checks the lengths match.
    Aligned { ty: TypeId, layer: &'static LayerName },
    /// The layer describes the document as a whole: its language, its
    /// category. No spans.
    Document,
}

impl Anchor {
    pub fn aligned<L: Layer>() -> Anchor {
        Anchor::Aligned { ty: TypeId::of::<L>(), layer: L::NAME }
    }
}

/// Names one instance of layer `L` on a document. Most layers have one
/// instance, `Key::DEFAULT`; a gold layer beside a predicted one, or two
/// tokenizations, are two names.
pub struct Key<L> {
    pub name: &'static str,
    _l: PhantomData<fn() -> L>,
}

impl<L: Layer> Key<L> {
    pub const DEFAULT: Key<L> = Key { name: "", _l: PhantomData };

    pub const fn named(name: &'static str) -> Key<L> {
        Key { name, _l: PhantomData }
    }

    pub fn id(&self) -> KeyId {
        KeyId { ty: TypeId::of::<L>(), layer: L::NAME, name: self.name }
    }
}

impl<L> Clone for Key<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Key<L> {}
impl<L> fmt::Debug for Key<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = std::any::type_name::<L>();
        write!(f, "Key<{}>", name.rsplit("::").next().unwrap_or(name))?;
        if !self.name.is_empty() {
            write!(f, "@{}", self.name)?;
        }
        Ok(())
    }
}

/// A layer type's full name: its id, and for a generic layer the full
/// names of the layers it is generic over.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LayerName {
    pub id: &'static str,
    pub of: Option<&'static LayerName>,
    pub to: Option<&'static LayerName>,
}

impl fmt::Display for LayerName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)?;
        match (self.of, self.to) {
            (Some(of), Some(to)) => write!(f, "<{of}, {to}>"),
            (Some(of), None) => write!(f, "<{of}>"),
            (None, Some(to)) => write!(f, "<, {to}>"),
            (None, None) => Ok(()),
        }
    }
}

/// A key with its type erased, for pipelines built at run time. It is the
/// layer's type and the instance's name; the type's own name rides along
/// for messages, so a key stays small enough to pass by value.
#[derive(Copy, Clone)]
pub struct KeyId {
    pub ty: TypeId,
    pub layer: &'static LayerName,
    pub name: &'static str,
}

impl PartialEq for KeyId {
    fn eq(&self, other: &KeyId) -> bool {
        self.ty == other.ty && self.name == other.name
    }
}

impl Eq for KeyId {}

impl std::hash::Hash for KeyId {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.ty.hash(h);
        self.name.hash(h);
    }
}

impl fmt::Debug for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.layer)?;
        if !self.name.is_empty() {
            write!(f, "@{}", self.name)?;
        }
        Ok(())
    }
}

impl fmt::Display for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

const _: () = assert!(std::mem::size_of::<KeyId>() <= 40);

/// Which model produced a layer, by the bundle's own identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRef {
    pub id: String,
    pub revision: String,
    pub sha256: String,
}

/// Where a model ran. Backends give measurably different numbers (an FP16
/// run is not an FP32 run), so a vector says which it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Execution {
    pub backend: String,
    pub device: String,
    pub precision: String,
}

/// Where a layer came from. Every layer has one, so a chunk, an entity or
/// a vector can say exactly what made it and from what.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub annotator: &'static str,
    pub version: &'static str,
    /// A hash of the annotator's configuration.
    pub config: u64,
    pub model: Option<ModelRef>,
    pub execution: Option<Execution>,
    /// The layers it read, optional ones only when they were there.
    pub reads: Vec<KeyId>,
    /// The layers it was made on that its step wrote itself rather than
    /// read: the tokens a tokenizer's own confidence column is aligned to,
    /// or the sentences a sentence and token splitter's tokens index. Only
    /// outputs the step wrote before this one are listed. Empty for a layer
    /// made only on what its step read.
    pub on: Vec<KeyId>,
}

impl Provenance {
    pub fn by(annotator: &'static str) -> Provenance {
        Provenance {
            annotator,
            version: "0",
            config: 0,
            model: None,
            execution: None,
            reads: Vec::new(),
            on: Vec::new(),
        }
    }
}

pub(crate) fn as_any_mut<L: Layer>(l: &mut L) -> &mut dyn Any {
    l
}
