//! Golden layers as test inputs: a `Document` built from the layers the
//! JVM produced, so one Rust annotator can run on the JVM's own
//! sentences, tokens or tags and be compared with its golden alone. A
//! POS tagger port is tested on the JVM's tokens this way, whatever the
//! Rust tokenizer does.
//!
//! Every loaded layer goes under a named key, `golden` unless the caller
//! picks another name, so the JVM's instance of a layer never takes the
//! default key a Rust annotator writes: `golden_key::<Tokens>()` is the
//! JVM's tokens and `Key::<Tokens>::DEFAULT` stays free for the Rust
//! tokenizer's. A layer aligned to another (tags per token, confidences
//! per sentence) and the entities, which index tokens, are loaded onto the
//! instance of that layer loaded under the same name, and their
//! provenance says so.

mod error;
mod layers;

use std::sync::Mutex;

use turbo_analysis::{Anchor, Document, Key, KeyId, Layer, LayerName, Provenance};
use turbo_analysis_layers::{Confidence, Entities, Language, Lemmas, PosTags, Sentences, Tokens};

pub use error::LoadError;

use crate::compare::{GoldenCompare, check_input, check_shape, shape_of};
use crate::format::{Golden, GoldenLayer};
use crate::sha::Sha256;

/// The key name a golden layer is loaded under unless the caller picks
/// another.
pub const GOLDEN_KEY: &str = "golden";

/// The key of the golden instance of `L`: `Key::<L>::named(GOLDEN_KEY)`.
pub fn golden_key<L: Layer>() -> Key<L> {
    Key::named(GOLDEN_KEY)
}

/// A layer type that can be built from its golden layer. The golden
/// layer's anchor and columns are checked against the type (`check_shape`)
/// before `load` runs, so a column the type's shape names is there, with
/// its type and as many rows as the others.
pub trait GoldenLoad: GoldenCompare {
    /// Builds the layer from its golden rows. Spans are source bytes, on
    /// `ViewId::SOURCE`; values are the f64 values as read.
    fn load(golden: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Self, LoadError>;

    /// The instances under the loaded layer's name that it is made on,
    /// which its provenance lists as read so the document anchors it
    /// there: by default an aligned layer's anchor, and nothing for any
    /// other.
    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        match Self::anchor() {
            Anchor::Aligned { ty, layer } => vec![KeyId { ty, layer, name: cx.name }],
            Anchor::Text | Anchor::Document => Vec::new(),
        }
    }
}

/// What a loader works in: the document so far, the golden file and the
/// name the layer is loaded under.
pub struct LoadCx<'a> {
    pub doc: &'a Document,
    pub golden: &'a Golden,
    /// The key name the layer is loaded under. The layers it is made on
    /// are the ones loaded under the same name.
    pub name: &'static str,
    /// The key the layer is loaded under, for errors.
    pub key: KeyId,
}

impl<'a> LoadCx<'a> {
    /// The document's source text, which is the golden's input.
    pub fn text(&self) -> &'a str {
        self.doc.source()
    }

    /// The instance of `A` loaded under the same name: an aligned layer's
    /// anchor, or the tokens an entity's token range indexes.
    pub fn anchor<A: Layer>(&self) -> Result<&'a A, LoadError> {
        let key = Key::<A>::named(self.name);
        let doc: &'a Document = self.doc;
        doc.get_key(&key).ok_or(LoadError::MissingAnchor { layer: self.key, anchor: key.id() })
    }
}

/// A document over `source` with every layer block of `golden` that has a
/// loader, in file order, each under `GOLDEN_KEY`. The source must be the
/// golden's input, by its length in UTF-8 bytes and its SHA-256.
///
/// A `turbo:view` block (what a normalizer produced) is not loaded: a view
/// golden is only compared with the view a Rust normalizer makes, so the
/// document from a view golden has the source text alone, for that
/// normalizer to run on. Any other block without a loader is refused with
/// `LoadError::NoLoader`.
pub fn load_document(golden: &Golden, source: &str) -> Result<Document, LoadError> {
    check_source(golden, source)?;
    let mut doc = Document::new(source);
    for g in &golden.layers {
        match LOADERS.iter().find(|(name, _)| is_named(&g.id, name)) {
            Some((_, load)) => load(&mut doc, golden, g, GOLDEN_KEY)?,
            None if is_named(&g.id, turbo_analysis::View::NAME) => {}
            None => return Err(LoadError::NoLoader { layer: g.id.clone() }),
        }
    }
    Ok(doc)
}

/// Loads `golden`'s layer of type `L` onto `doc` under `key`. The
/// document's source must be the golden's input, and a layer `L` is
/// aligned to, or indexes, must be on the document under the same key
/// name already.
pub fn load_layer<L: GoldenLoad>(doc: &mut Document, golden: &Golden, key: &Key<L>) -> Result<(), LoadError> {
    check_source(golden, doc.source())?;
    let Some(g) = golden.layers.iter().find(|g| is_named(&g.id, L::NAME)) else {
        return Err(LoadError::MissingGolden {
            layer: key.id(),
            available: golden.layers.iter().map(|l| l.id.clone()).collect(),
        });
    };
    add::<L>(doc, golden, g, key.name)
}

type LoadFn = fn(&mut Document, &Golden, &GoldenLayer, &'static str) -> Result<(), LoadError>;

/// A loader per layer type `load_document` loads, by the layer's name.
const LOADERS: &[(&LayerName, LoadFn)] = &[
    (Sentences::NAME, add::<Sentences>),
    (Confidence::<Sentences>::NAME, add::<Confidence<Sentences>>),
    (Tokens::NAME, add::<Tokens>),
    (Confidence::<Tokens>::NAME, add::<Confidence<Tokens>>),
    (PosTags::NAME, add::<PosTags>),
    (Lemmas::NAME, add::<Lemmas>),
    (Entities::NAME, add::<Entities>),
    (Language::NAME, add::<Language>),
];

/// Whether a golden block's id is this layer name, as `LayerName` prints
/// it (`turbo:confidence<opennlp:tokens>` for a generic layer).
fn is_named(id: &str, name: &LayerName) -> bool {
    if name.of.is_empty() {
        return id == name.id;
    }
    id.strip_prefix(name.id).and_then(|r| r.strip_prefix('<')).and_then(|r| r.strip_suffix('>')) == Some(name.of)
}

fn check_source(golden: &Golden, source: &str) -> Result<(), LoadError> {
    check_input(golden, source).map_err(|found| {
        let input = &golden.header.input;
        LoadError::Input {
            expected_len: input.utf8_len,
            found_len: found.len,
            expected_sha256: input.sha256,
            found_sha256: found.sha256,
        }
    })
}

/// Checks the block's shape, builds the layer and adds it under `name`.
fn add<L: GoldenLoad>(
    doc: &mut Document,
    golden: &Golden,
    g: &GoldenLayer,
    name: &'static str,
) -> Result<(), LoadError> {
    let key = Key::<L>::named(name);
    check_shape(&key.id().to_string(), &shape_of::<L>(), g).map_err(LoadError::Shape)?;
    let cx = LoadCx { doc, golden, name, key: key.id() };
    let layer = L::load(g, &cx)?;
    let reads = L::made_on(&cx);
    doc.add(&key, layer, provenance(golden, reads)).map_err(LoadError::Document)
}

/// A loaded layer's provenance. The annotator is `golden:<class>`, the
/// Java class that produced the layer, and the version is the OpenNLP
/// version that ran. The configuration hash is the first eight bytes, big
/// endian, of the SHA-256 of what set the output besides the input: the
/// annotator's class and method, the OpenNLP commit, each model's file
/// name and SHA-512, and each option, every field preceded by its length.
/// No model is recorded: `ModelRef` holds one model with a SHA-256, and a
/// golden names any number of models by SHA-512; the hash covers them.
fn provenance(golden: &Golden, reads: Vec<KeyId>) -> Provenance {
    let h = &golden.header;
    let mut sha = Sha256::new();
    let mut field = |bytes: &[u8]| {
        sha.update(&(bytes.len() as u64).to_be_bytes());
        sha.update(bytes);
    };
    field(h.annotator.class.as_bytes());
    field(h.annotator.method.as_bytes());
    field(&h.opennlp.commit);
    for m in &h.models {
        field(m.file.as_bytes());
        field(&m.sha512);
    }
    for c in &h.config {
        field(c.key.as_bytes());
        field(c.value.as_bytes());
    }
    let digest = sha.finish();
    let config = u64::from_be_bytes(digest[..8].try_into().expect("eight bytes"));
    Provenance {
        version: intern(&h.opennlp.version),
        config,
        reads,
        ..Provenance::by(intern(&format!("golden:{}", h.annotator.class)))
    }
}

/// A `'static` copy of `s`, as `Provenance` holds its annotator and
/// version. Each distinct string is leaked once and reused after that, so
/// the memory is bounded by the number of distinct annotator classes and
/// OpenNLP versions the loaded goldens name.
fn intern(s: &str) -> &'static str {
    static TABLE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut table = TABLE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(&found) = table.iter().find(|&&t| t == s) {
        return found;
    }
    let leaked: &'static str = Box::leak(s.into());
    table.push(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_ids_match_layer_names() {
        assert!(is_named("opennlp:tokens", Tokens::NAME));
        assert!(!is_named("opennlp:tokens", Confidence::<Tokens>::NAME));
        assert!(is_named("turbo:confidence<opennlp:tokens>", Confidence::<Tokens>::NAME));
        assert!(!is_named("turbo:confidence<opennlp:sentences>", Confidence::<Tokens>::NAME));
        assert!(!is_named("turbo:confidence<opennlp:tokens", Confidence::<Tokens>::NAME));
        assert!(!is_named("turbo:confidence", Confidence::<Tokens>::NAME));
    }

    #[test]
    fn interned_strings_are_reused() {
        let a = intern("golden:a.B");
        let b = intern(&String::from("golden:a.B"));
        assert!(std::ptr::eq(a, b));
        assert_eq!(a, "golden:a.B");
    }
}
