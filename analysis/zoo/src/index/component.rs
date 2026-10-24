//! The component a model serves, and the index's lookups by language and
//! component.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Artifact, Index};
use crate::licence::LicenceClass;

constant_enum! {
    /// The analysis component a model serves. The first eight are
    /// OpenNLP's `opennlp.tools.models.ModelType` constants, prefixed; the
    /// others have no `ModelType`, so OpenNLP's lookup never returns them.
    Component, "a component" {
        Lemmatizer => "COMPONENT_LEMMATIZER",
        Tokenizer => "COMPONENT_TOKENIZER",
        SentenceDetector => "COMPONENT_SENTENCE_DETECTOR",
        Pos => "COMPONENT_POS",
        PosGeneric => "COMPONENT_POS_GENERIC",
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
    }
}

impl Index {
    /// The listed models for `component` in `language`, in index order, as
    /// OpenNLP's `DownloadUtil` looks a model up by language and
    /// `ModelType`.
    pub fn find(&self, language: &str, component: Component) -> impl Iterator<Item = &Artifact> + '_ {
        let language = language.to_owned();
        self.artifacts.iter().filter(move |a| {
            a.component == Some(component) && a.language.as_ref().is_some_and(|l| l.as_str() == language)
        })
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
