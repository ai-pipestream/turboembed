//! An evaluation set once it is read: the documents, the queries, and the
//! judgements at document level and, where the set has them, at span
//! level.
//!
//! Every reader produces these values. The pipeline reads nothing else
//! from a set, so a set's file format stays inside its reader.

use std::collections::BTreeSet;

use crate::Result;
use crate::metrics::Qrels;
use crate::spans::SpanRels;

/// One document of a corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doc {
    pub id: String,
    /// The title, when the set gives one apart from the text. BEIR
    /// prepends it to the text it embeds; a pipeline decides for itself.
    pub title: Option<String>,
    /// The text as loaded. Span judgements are UTF-8 byte offsets into
    /// this text.
    pub text: String,
}

/// One query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub id: String,
    pub text: String,
}

/// A set as read.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSet {
    /// The set's name, as records name it.
    pub name: String,
    /// SHA-256 of the files read, 64 lower case hex, as records carry it.
    pub sha256: String,
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    pub qrels: Qrels,
    /// `None` for sets without span judgements.
    pub spans: Option<SpanRels>,
}

impl LoadedSet {
    /// Refuses a set with a document or query id twice, or with no
    /// documents or no queries.
    pub fn check(&self) -> Result<()> {
        if self.docs.is_empty() {
            return Err(format!("set {} has no documents", self.name));
        }
        if self.queries.is_empty() {
            return Err(format!("set {} has no queries", self.name));
        }
        let mut seen = BTreeSet::new();
        for d in &self.docs {
            if !seen.insert(d.id.as_str()) {
                return Err(format!("set {} has document {} twice", self.name, d.id));
            }
        }
        seen.clear();
        for q in &self.queries {
            if !seen.insert(q.id.as_str()) {
                return Err(format!("set {} has query {} twice", self.name, q.id));
            }
        }
        Ok(())
    }
}
