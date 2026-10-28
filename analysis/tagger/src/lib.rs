//! Tagging with a bundle's own token tagger heads (`manifest.heads`,
//! docs/bundle.md "Heads"): one encoder, any backend, and heads trained
//! by `turbo-trainer`. [`PosTagger`] writes the part-of-speech layer,
//! [`EntityTagger`] the entity layer and [`DependencyParser`] the
//! dependency layer, each from the head that names that layer, or from a
//! head named outright.
//!
//! The encoder pools each word's pieces on the device (words.rs); the
//! head scores and decodes on the host, the same on every backend.

pub mod annotate;
pub mod parse;
pub mod words;

use std::path::Path;
use std::sync::Arc;

use turbo::manifest::Scheme;
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Layer, ModelRef, Refs, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};

pub use annotate::HeadRunner;
pub use parse::DependencyParser;
pub use words::WordEncoder;

/// Writes the part-of-speech layer from a SCHEME_TAGS head.
pub struct PosTagger {
    run: HeadRunner,
    tagset: Arc<TagSet>,
}

impl PosTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is this layer's id.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<PosTagger, CoreError> {
        let run = HeadRunner::open(dir, name, PosTags::ID, o)?;
        if run.scheme() != Scheme::Tags {
            return Err(CoreError::new(INVALID_ARGUMENT, "a part-of-speech head reads SCHEME_TAGS labels"));
        }
        let spec = run.tagger().spec();
        let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
        let tagset = TagSet::new(&spec.name, &labels);
        Ok(PosTagger { run, tagset })
    }

    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }
}

impl Annotator for PosTagger {
    const ID: &'static str = "turbo-tagger-pos";
    type Reads = (View, Tokens, Option<Sentences>);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.run
            .pos_tags(view, tokens, sentences, &self.tagset)
            .map(|t| (t,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.run.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.run.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        config(&self.run)
    }
}

/// Writes the entity layer from a SCHEME_BIO or SCHEME_BILOU head.
pub struct EntityTagger {
    run: HeadRunner,
    kinds: Arc<TagSet>,
}

impl EntityTagger {
    /// As [`PosTagger::open`], for the entity layer.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<EntityTagger, CoreError> {
        let run = HeadRunner::open(dir, name, Entities::ID, o)?;
        if run.scheme() == Scheme::Tags {
            return Err(CoreError::new(INVALID_ARGUMENT, "an entity head reads SCHEME_BIO or SCHEME_BILOU labels"));
        }
        let types: Vec<&str> = run.tagger().types().iter().map(String::as_str).collect();
        let kinds = TagSet::new(&run.tagger().spec().name, &types);
        Ok(EntityTagger { run, kinds })
    }

    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }
}

impl Annotator for EntityTagger {
    const ID: &'static str = "turbo-tagger-entities";
    type Reads = (View, Tokens, Option<Sentences>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.run
            .entities(view, tokens, sentences, &self.kinds)
            .map(|e| (e,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.run.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.run.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        config(&self.run)
    }
}

/// What changes the output beside the model: the head, and the pieces a
/// row holds (where long sentences are cut).
fn config(run: &HeadRunner) -> u64 {
    let width = run.width() as u64;
    Fnv1a::new().str(&run.tagger().spec().name).u64(width).finish()
}
