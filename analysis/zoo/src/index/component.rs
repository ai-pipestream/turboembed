//! The component a model serves, and the index's lookups by language and
//! component.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Artifact, Index};
use crate::licence::LicenceClass;

constant_enum! {
    /// The analysis capability a model or learned head serves.
    Component, "a component" {
        Lemmatizer => "COMPONENT_LEMMATIZER",
        Tokenizer => "COMPONENT_TOKENIZER",
        SentenceDetector => "COMPONENT_SENTENCE_DETECTOR",
        Pos => "COMPONENT_POS",
        NameFinder => "COMPONENT_NAME_FINDER",
        Chunker => "COMPONENT_CHUNKER",
        Parser => "COMPONENT_PARSER",
        LanguageDetector => "COMPONENT_LANGUAGE_DETECTOR",
        DocumentCategorizer => "COMPONENT_DOCUMENT_CATEGORIZER",
        /// Spans of the entity types a caller names (GLiNER).
        EntityExtractor => "COMPONENT_ENTITY_EXTRACTOR",
        /// Entities, classes and structured fields of a schema (GLiNER2).
        SchemaExtractor => "COMPONENT_SCHEMA_EXTRACTOR",
        /// Relations between entity spans (GLiREL).
        RelationExtractor => "COMPONENT_RELATION_EXTRACTOR",
        /// An answer span in a context, or none (SQuAD 2.0).
        ExtractiveQa => "COMPONENT_EXTRACTIVE_QA",
        /// Labels a caller names, scored against a text by an NLI
        /// cross-encoder (zero-shot categorization, docs/doccat.md).
        ZeroShotCategorizer => "COMPONENT_ZERO_SHOT_CATEGORIZER",
        /// Typed dependency arcs between tokens.
        DependencyParser => "COMPONENT_DEPENDENCY_PARSER",
        /// A vocabulary-sized sparse term-weight vector per text (SPLADE).
        SparseEncoder => "COMPONENT_SPARSE_ENCODER",
        /// One vector per token, scored by late interaction (ColBERT).
        LateInteractionEncoder => "COMPONENT_LATE_INTERACTION_ENCODER",
        /// Aspect spans and the polarity expressed towards each.
        AspectSentiment => "COMPONENT_ASPECT_SENTIMENT",
    }
}

impl Index {
    /// The listed artifacts for `component` in `language`, the least
    /// restrictive licence class first (`CLASS_PERMISSIVE` before any class
    /// needing acceptance or kept local) and in index order within a class.
    pub fn find(&self, language: &str, component: Component) -> impl Iterator<Item = &Artifact> + '_ {
        let mut found: Vec<&Artifact> = self
            .artifacts
            .iter()
            .filter(|a| a.component == Some(component) && a.language.as_ref().is_some_and(|l| l.as_str() == language))
            .collect();
        found.sort_by_key(|a| a.class);
        found.into_iter()
    }

    /// [`Index::find`]'s models whose class is `at_most` or less
    /// restrictive, the least restrictive first and in index order within
    /// a class: with `LicenceClass::Permissive`, only the models that need
    /// no acceptance.
    pub fn find_within(&self, language: &str, component: Component, at_most: LicenceClass) -> Vec<&Artifact> {
        let mut found: Vec<&Artifact> = self.find(language, component).filter(|a| a.class <= at_most).collect();
        found.sort_by_key(|a| a.class);
        found
    }
}
