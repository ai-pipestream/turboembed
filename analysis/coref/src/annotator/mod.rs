//! The coreference annotator (`CorefAnnotator`): mention detection, then
//! the sieves, written as the `Mentions` layer, its chains as
//! `Groups<Mentions>`, and each resolved link as an edge of
//! `Edges<Mentions, Mentions>` from anaphor to antecedent.

use turbo_analysis::{Annotator, Edges, Error, Groups, Refs, View};
use turbo_analysis_layers::{Entities, PhraseChunks, Phrases, PosTags, Sentences, Tokens};

use crate::detect::Detector;
use crate::input::{CorefInput, lowercase};
use crate::layer::{Mentions, Speakers};
use crate::sieve::Resolver;

mod config;
mod output;

pub use output::sieve_tags;

/// The output of one resolution: the mentions, every chain (singletons
/// included) in first-mention order, and the antecedent links.
pub type Resolution = (Mentions, Groups<Mentions>, Edges<Mentions, Mentions>);

/// Resolves entity, noun phrase and pronoun mentions into chains with
/// the entity-centric sieves of Lee et al. (2013). First and second
/// person pronouns use the optional [`Speakers`] layer, or quoted speech
/// attribution without it.
#[derive(Debug)]
pub struct CorefAnnotator {
    person_types: Vec<String>,
    neutral_types: Vec<String>,
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
        if turbo_unicode::whitespace::is_blank(t) {
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
        let person_types = lowered(person_types, "personTypes")?;
        let neutral_types = lowered(neutral_types, "neutralTypes")?;
        Ok(CorefAnnotator { person_types, neutral_types })
    }

    /// Resolves coreference over `input`.
    pub fn resolve(&self, input: &CorefInput<'_>) -> Result<Resolution, String> {
        let Some(prepared) = input.prepare()? else {
            return Ok(output::empty(input.view));
        };
        let mentions = Detector::new(&prepared, &self.person_types).detect();
        let mut resolver = Resolver::new(
            mentions,
            &prepared.forms,
            &prepared.sentence_of_token,
            prepared.speakers.as_deref(),
            &self.person_types,
            &self.neutral_types,
        );
        resolver.resolve();
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

    fn config_hash(&self) -> u64 {
        config::hash(self)
    }
}
