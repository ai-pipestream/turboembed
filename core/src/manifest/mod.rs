//! `manifest.json` in the one canonical JSON form docs/bundle.md
//! describes, parsed strictly and written by the bundle tool. An unknown field or enum value, a
//! missing required field, an over-long string or a bad path is
//! BUNDLE_INVALID naming the field.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::status::{Error, Result, UNSUPPORTED_TASK, invalid};

pub const BUNDLE_VERSION: u32 = 1;

fn is_zero(v: &u32) -> bool {
    *v == 0
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub bundle_version: u32,
    pub model: Model,
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<Embed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<Rerank>,
    pub qa: Option<Qa>,
    pub tokenizer: Tokenizer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<Architecture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_embedding: Option<StaticEmbedding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classifier: Option<Classifier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<Projection>,
    pub artifacts: Vec<Artifact>,
    pub reference: Reference,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub revision: String,
    pub source: Source,
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_file: Option<String>,
    /// The class the model zoo puts `license` in (analysis/zoo), which the
    /// bundle tool writes from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_class: Option<LicenseClass>,
}

/// A model licence's class, as the zoo names it: what may be shipped,
/// and what is fetched only after the user accepts the licence. A bundle
/// is never made under a refused licence, so there is no refused class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseClass {
    #[serde(rename = "CLASS_PERMISSIVE")]
    Permissive,
    #[serde(rename = "CLASS_LGPL")]
    Lgpl,
    #[serde(rename = "CLASS_SHARE_ALIKE")]
    ShareAlike,
    #[serde(rename = "CLASS_GPL")]
    Gpl,
    #[serde(rename = "CLASS_LOCAL_ONLY")]
    LocalOnly,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repository: String,
    pub commit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Embed {
    pub dim: u32,
    pub pooling: Pooling,
    pub normalize: Normalize,
    pub max_seq: u32,
    pub max_batch: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix_query: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix_document: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_dims: Vec<u32>,
    /// A learned sparse output beside the dense vectors: the masked
    /// language model head named in `artifacts[].tensor_names` is pooled
    /// over the text's tokens into weighted vocabulary terms (mlm.rs,
    /// sparse.rs; SPLADE, arXiv 2403.06789).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<Sparse>,
}

/// The sparse output kind an embed bundle serves: which pooling turns the
/// MLM head's per-token logits into term weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sparse {
    pub pooling: SparsePooling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparsePooling {
    /// `max_t log(1 + relu(logit_t))`, as SPLADE pools (arXiv 2403.06789).
    #[serde(rename = "SPARSE_POOLING_MAXLOG")]
    MaxLog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pooling {
    #[serde(rename = "POOLING_MEAN")]
    Mean,
    #[serde(rename = "POOLING_CLS")]
    Cls,
    #[serde(rename = "POOLING_LAST")]
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Normalize {
    #[serde(rename = "NORMALIZE_NONE")]
    None,
    #[serde(rename = "NORMALIZE_L2")]
    L2,
}

mod architecture;
mod artifact;
mod helpers;
mod provenance;
mod qa;
pub use qa::Qa;
mod tokenizer;
mod validate;
mod validate_architecture;
mod validate_artifacts;
mod validate_tokenizer;
pub use architecture::*;
pub use artifact::*;
pub use helpers::role_name;
use helpers::*;
pub use provenance::*;
pub use tokenizer::*;
impl Manifest {
    /// Parse and check everything that can be checked without opening
    /// another file.
    pub fn parse(bytes: &[u8]) -> Result<Manifest> {
        let de = &mut serde_json::Deserializer::from_slice(bytes);
        let task = peek_task(bytes);
        let m: Manifest = match serde_path_to_error::deserialize(&mut *de) {
            Ok(m) => m,
            Err(e) => {
                // A task this build does not have is reported as such, not
                // as a malformed manifest.
                if let Some(t) = task
                    .filter(|t| t != "TASK_EMBED" && t != "TASK_RERANK" && t != "TASK_EXTRACTIVE_QA" && is_task_name(t))
                {
                    return Err(unsupported_task(&t));
                }
                let path = e.path().to_string();
                return Err(invalid(format!("manifest.json: {path}: {}", e.inner())));
            }
        };
        de.end().map_err(|e| invalid(format!("manifest.json: {e}")))?;
        m.validate()?;
        Ok(m)
    }

    /// The manifest in the JSON form bundle writers and tests persist.
    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("a validated manifest serializes")
    }

    pub fn embed(&self) -> &Embed {
        self.embed.as_ref().expect("validate() requires embed for TASK_EMBED")
    }

    pub fn file(&self, path: &str) -> &FileEntry {
        self.files.iter().find(|f| f.path == path).expect("validate() requires every path in files")
    }
}

mod rerank;
pub use rerank::*;
