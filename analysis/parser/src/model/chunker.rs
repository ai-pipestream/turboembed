//! The chunker sub-model inside a [`ParserModel`] and the factory it was written with
//! (`opennlp.tools.parser.ParserChunkerFactory` or `opennlp.tools.chunker.ChunkerFactory`).

use turbo_maxent::{MaxentModel, ModelFile};
use turbo_sequence::BeamSearch;
use turbo_sequence::chunker::{
    ChunkerValidator, DEFAULT_BEAM_SIZE as CHUNKER_DEFAULT_BEAM_SIZE, DefaultChunkerContext,
    MODEL_ENTRY_NAME as CHUNKER_MODEL_ENTRY_NAME, TokenTag,
};
use turbo_sequence::manifest_beam_size;

use crate::context::{ChunkContextGenerator, ParserChunkerSequenceValidator};

use super::*;

/// Which factory the chunker sub-model inside a [`ParserModel`] uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParserChunkerFactoryKind {
    /// `opennlp.tools.parser.ParserChunkerFactory` (`ChunkContextGenerator` + `ParserChunkerSequenceValidator`).
    ParserChunker,
    /// `opennlp.tools.chunker.ChunkerFactory` (`DefaultChunkerContext` + `ChunkerValidator`).
    DefaultChunker,
}

/// The chunker sub-model embedded in a [`ParserModel`] (`parserchunker.chunker`).
#[derive(Debug)]
pub struct ParserChunkerModel {
    file: ModelFile,
    maxent: MaxentModel,
    beam_size: usize,
    factory_kind: ParserChunkerFactoryKind,
    outcomes: Vec<String>,
}

impl ParserChunkerModel {
    /// Opens a chunker sub-model from a [`ModelFile`].
    pub fn open(file: ModelFile) -> Result<Self, ParserModelError> {
        if file.component_name() != "ChunkerME" {
            return Err(ParserModelError::ChunkerModel(format!(
                "component {:?} is not ChunkerME",
                file.component_name()
            )));
        }
        let factory_kind = match file.manifest().get("factory") {
            None | Some(DEFAULT_CHUNKER_FACTORY) => ParserChunkerFactoryKind::DefaultChunker,
            Some(PARSER_CHUNKER_FACTORY) => ParserChunkerFactoryKind::ParserChunker,
            Some(other) => {
                return Err(ParserModelError::ChunkerModel(format!("unsupported chunker factory {other:?}")));
            }
        };
        let maxent = file.maxent(CHUNKER_MODEL_ENTRY_NAME)?;
        if maxent.num_outcomes() == 0 {
            return Err(ParserModelError::ChunkerModel("classifier has no outcomes".into()));
        }
        let outcomes: Vec<String> = maxent
            .outcome_texts()
            .map_err(|o| ParserModelError::ChunkerModel(format!("outcome {o:?} is not valid UTF-8")))?
            .into_iter()
            .map(str::to_owned)
            .collect();
        let beam_size = manifest_beam_size(&file, CHUNKER_DEFAULT_BEAM_SIZE)?;
        if beam_size == 0 {
            return Err(ParserModelError::ChunkerModel("beam size must not be 0".into()));
        }
        Ok(Self { file, maxent, beam_size, factory_kind, outcomes })
    }

    /// Returns the underlying [`ModelFile`].
    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// Returns the classifier (`chunker.model`).
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// Returns the beam size.
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }

    /// Returns the factory kind.
    pub fn factory_kind(&self) -> ParserChunkerFactoryKind {
        self.factory_kind
    }

    /// Returns the outcome labels.
    pub fn outcomes(&self) -> &[String] {
        &self.outcomes
    }

    /// Runs `ChunkerME.topKSequences(words, tags, minSequenceScore)` over this chunker model.
    pub fn top_k_sequences<'m>(
        &'m self,
        words: &[&str],
        tags: &[&str],
        min_sequence_score: f64,
    ) -> Vec<turbo_sequence::Sequence<'m>> {
        let tuples: Vec<TokenTag<'_>> =
            words.iter().copied().zip(tags.iter().copied()).map(|(token, tag)| TokenTag { token, tag }).collect();
        let search = BeamSearch::new(self.beam_size, &self.maxent).expect("outcomes validated at open");
        match self.factory_kind {
            ParserChunkerFactoryKind::ParserChunker => {
                let mut cg = ChunkContextGenerator::with_cache(CHUNKER_DEFAULT_BEAM_SIZE);
                let validator = ParserChunkerSequenceValidator::new(&self.outcomes);
                search.best_sequences(CHUNKER_DEFAULT_BEAM_SIZE, &tuples, min_sequence_score, &mut cg, &validator)
            }
            ParserChunkerFactoryKind::DefaultChunker => {
                let mut cg = DefaultChunkerContext;
                let validator = ChunkerValidator;
                search.best_sequences(CHUNKER_DEFAULT_BEAM_SIZE, &tuples, min_sequence_score, &mut cg, &validator)
            }
        }
    }
}
