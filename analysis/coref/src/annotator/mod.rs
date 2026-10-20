//! The coreference annotator (`CorefAnnotator`): mention detection, then
//! the sieves or a model, written as the `Mentions` layer, its chains as
//! `Groups<Mentions>`, and each resolved link as an edge of
//! `Edges<Mentions, Mentions>` from anaphor to antecedent.

use std::sync::Arc;

use turbo_analysis::{Annotator, Edges, Error, Groups, ModelRef, Refs, View};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, PosTags, Sentences, Tokens};

use crate::detect::Detector;
use crate::input::{CorefInput, Prepared, lowercase};
use crate::layer::{Mentions, Speakers};
use crate::model::CorefModel;
use crate::sieve::Resolver;
use crate::vectors::{TokenVectors, WordVectors, encode};

mod config;
mod output;

pub use output::sieve_tags;

/// The output of one resolution: the mentions, every chain (singletons
/// included) in first-mention order, and the antecedent links.
pub type Resolution = (Mentions, Groups<Mentions>, Edges<Mentions, Mentions>);

/// The least link probability at which a pair classifier links, the
/// default of the model constructors.
pub const DEFAULT_THRESHOLD: f64 = 0.1;

/// Resolves entity, noun phrase and pronoun mentions into chains.
///
/// Without a model, the annotator applies the entity-centric sieves of
/// Lee et al. (2013). A [`CorefModel`] replaces the head and pronoun
/// sieves with antecedent ranking or pair classification. First and
/// second person pronouns use the optional [`Speakers`] layer, or quoted
/// speech attribution without it.
pub struct CorefAnnotator {
    person_types: Vec<String>,
    neutral_types: Vec<String>,
    model: Option<Arc<CorefModel>>,
    threshold: f64,
    vectors: Option<Arc<dyn WordVectors>>,
    encoder: Option<Arc<dyn TokenVectors>>,
}

impl std::fmt::Debug for CorefAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CorefAnnotator")
            .field("person_types", &self.person_types)
            .field("neutral_types", &self.neutral_types)
            .field("model", &self.model.is_some())
            .field("threshold", &self.threshold)
            .field("vectors", &self.vectors.is_some())
            .field("encoder", &self.encoder.is_some())
            .finish()
    }
}

impl Default for CorefAnnotator {
    /// The rule-based annotator for the conventional entity type names:
    /// `person` for gendered pronouns, `organization` and `location` for
    /// neutral ones.
    fn default() -> CorefAnnotator {
        CorefAnnotator::rules(&["person"], &["organization", "location"]).expect("valid default types")
    }
}

fn lowered(types: &[&str], name: &str) -> Result<Vec<String>, String> {
    if types.is_empty() {
        return Err(format!("{name} must not be null or empty"));
    }
    let mut out: Vec<String> = Vec::with_capacity(types.len());
    for t in types {
        if t.chars().all(turbo_unicode::java::is_whitespace) {
            return Err(format!("{name} must not contain blank entries"));
        }
        let l = lowercase(t);
        if !out.contains(&l) {
            out.push(l);
        }
    }
    Ok(out)
}

impl CorefAnnotator {
    /// The rule-based annotator for the given entity types, matched
    /// case-insensitively.
    pub fn rules(person_types: &[&str], neutral_types: &[&str]) -> Result<CorefAnnotator, String> {
        CorefAnnotator::new(person_types, neutral_types, None, DEFAULT_THRESHOLD, None, None)
    }

    /// A model annotator for the conventional entity type names.
    pub fn with_model(model: impl Into<Arc<CorefModel>>) -> Result<CorefAnnotator, String> {
        CorefAnnotator::new(
            &["person"],
            &["organization", "location"],
            Some(model.into()),
            DEFAULT_THRESHOLD,
            None,
            None,
        )
    }

    /// The fully configured annotator. `threshold` lies in `[0, 1]`; the
    /// encoder's dimension must match the model's token-vector dimension
    /// (zero meaning no encoder).
    pub fn new(
        person_types: &[&str],
        neutral_types: &[&str],
        model: Option<Arc<CorefModel>>,
        threshold: f64,
        vectors: Option<Arc<dyn WordVectors>>,
        encoder: Option<Arc<dyn TokenVectors>>,
    ) -> Result<CorefAnnotator, String> {
        let person_types = lowered(person_types, "personTypes")?;
        let neutral_types = lowered(neutral_types, "neutralTypes")?;
        if !(0.0..=1.0).contains(&threshold) {
            return Err(format!("threshold must lie in [0, 1]: {threshold}"));
        }
        if let Some(v) = &vectors
            && v.dimension() == 0
        {
            return Err("vectors dimension must be positive: 0".into());
        }
        if let Some(e) = &encoder
            && e.dimension() == 0
        {
            return Err("encoder dimension must be positive: 0".into());
        }
        if let Some(m) = &model {
            let expected = m.token_vector_dimension();
            match &encoder {
                Some(_) if expected == 0 => return Err("model was trained without a token encoder".into()),
                None if expected > 0 => return Err(format!("model requires a token encoder of dimension {expected}")),
                Some(e) if e.dimension() != expected => {
                    return Err(format!(
                        "encoder dimension {} does not match model dimension {expected}",
                        e.dimension()
                    ));
                }
                _ => {}
            }
        }
        Ok(CorefAnnotator { person_types, neutral_types, model, threshold, vectors, encoder })
    }

    /// The token-vector dimension of this configuration, zero without an
    /// encoder.
    pub fn token_vector_dimension(&self) -> usize {
        self.encoder.as_ref().map_or(0, |e| e.dimension())
    }

    /// Validates the input, detects its mentions and builds the resolver
    /// over them; `None` for a document without tokens.
    pub(crate) fn resolver<'p>(&'p self, prepared: &'p Prepared<'p>) -> Result<Resolver<'p>, String> {
        let mentions = Detector::new(prepared, &self.person_types).detect();
        let token_vectors = match &self.encoder {
            Some(e) => Some(encode(e.as_ref(), &prepared.forms, &prepared.sentence_of_token)?),
            None => None,
        };
        Ok(Resolver::new(
            mentions,
            &prepared.forms,
            &prepared.sentence_of_token,
            prepared.speakers.as_deref(),
            &self.person_types,
            &self.neutral_types,
            self.vectors.as_deref(),
            token_vectors,
        ))
    }

    /// Resolves coreference over `input`.
    pub fn resolve(&self, input: &CorefInput<'_>) -> Result<Resolution, String> {
        let Some(prepared) = input.prepare()? else {
            return Ok(output::empty(input.view));
        };
        let mut resolver = self.resolver(&prepared)?;
        match &self.model {
            None => resolver.resolve(),
            Some(model) => resolver.resolve_with_model(model, self.threshold)?,
        }
        output::layers(input.view, &mut resolver)
    }
}

impl Annotator for CorefAnnotator {
    const ID: &'static str = "turbo-coref:coref";
    type Reads = (View, Sentences, Tokens, PosTags, Entities, Option<PhraseChunks>, Option<Phrases>, Option<Speakers>);
    type Writes = (Mentions, Groups<Mentions>, Edges<Mentions, Mentions>);

    fn annotate(
        &self,
        (view, sentences, tokens, tags, entities, chunks, phrases, speakers): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let input = CorefInput {
            text: view.text(),
            view: view.id(),
            sentences,
            tokens,
            tags,
            entities,
            chunks,
            phrases,
            speakers,
        };
        self.resolve(&input).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        self.model.as_ref().map(|m| m.model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        config::hash(self)
    }
}
