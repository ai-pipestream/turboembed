use crate::annotate::Step;
use crate::{Document, KeyId};

use super::ConditionalAnnotator;

/// A document predicate with explicit input dependencies and configuration
/// identity. Ordinary closure-based ConditionalStep constructors remain valid.
pub trait DocPredicate: Send + Sync + 'static {
    fn test(&self, doc: &Document) -> bool;
    /// Every input the predicate reads, including named instances.
    fn reads(&self) -> Vec<KeyId>;
    /// Include every parameter that changes the predicate's meaning.
    fn config_hash(&self) -> u64;
}

impl ConditionalAnnotator {
    /// Creates a conditional step whose typed predicate automatically declares
    /// its required inputs and contributes its configuration to provenance.
    pub fn from_predicate(
        predicate_name: &'static str,
        predicate: impl DocPredicate,
        when_true: impl Step + 'static,
    ) -> Self {
        let config = predicate.config_hash();
        let reads = predicate.reads();
        let mut out = Self::new(predicate_name, move |doc| predicate.test(doc), when_true);
        out.predicate_config = Some(config);
        for key in reads {
            if let Some(at) = out.reads.iter().position(|&r| r == key) {
                out.required[at] = true;
            } else {
                out.reads.push(key);
                out.required.push(true);
            }
        }
        out
    }
}
