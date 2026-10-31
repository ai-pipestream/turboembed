//! Native MASC NER preparation (#1524). No reference models or downloads.
//!
//! This is a four-type MASC head, not a combined MASC/WNUT head. A type
//! absent from a dataset's annotation scheme is not a verified negative.
//! Production bundle output and real-data accuracy gates remain separate.
//!
//! # Production block
//!
//! The MASC reader currently fragments cross-sentence entities without
//! retaining identity or a fragmentation marker. Sentence-local checks
//! cannot detect this loss. Do not run production MASC training through
//! this module until the formats correction requested on #1509 is merged
//! and a regression verifies refusal or explicit handling of fragments.
//! Only nonfragmented, project-authored correctness fixtures have been run.

mod data;
mod fit;

pub use data::{DocumentId, Partition, Prepared, Sentence, prepare, sentences};
pub use fit::{Run, RunIdentity, fit_cpu};

use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo::status::{Error, INVALID_ARGUMENT};

/// MASC annotation names, with `org` mapped explicitly to organization.
pub const TYPES: [&str; 4] = ["person", "location", "organization", "date"];

/// A linear token head and constrained BIO CRF over frozen encoder words.
pub fn head() -> HeadSpec {
    let mut labels = vec!["O".to_owned()];
    for kind in TYPES {
        labels.extend([format!("B-{kind}"), format!("I-{kind}")]);
    }
    HeadSpec {
        name: "masc-ner".into(),
        layer: "turboembed:entities".into(),
        kind: HeadKind::TokenTagger,
        labels,
        scheme: Scheme::Bio,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Crf,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        relation: None,
        coref: None,
        license_class: Some(turbo::manifest::LicenseClass::Permissive),
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.masc-ner".into(),
    }
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::new(INVALID_ARGUMENT, format!("native MASC NER: {message}"))
}

fn digest(bytes: &[u8]) -> String {
    turbo_sha256::sha256(bytes).to_hex()
}

fn field(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}
