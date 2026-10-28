//! `heads`: the task heads a bundle carries beside its encoder, each read
//! on the host from the encoder's final-layer states (head/tagger.rs).
//! A list, so one bundle can carry several heads over one encoder pass.
//! Each head names its weights by a tensor prefix and fixed suffixes
//! (docs/bundle.md, "Heads"), not by `tensor_names` roles, so any number
//! of heads fit one artifact.

use serde::{Deserialize, Serialize};

use super::helpers::{fits, positive, required};
use super::{Format, Manifest};
use crate::status::{Result, invalid};

/// The most heads one bundle carries.
pub const MAX_HEADS: usize = 32;
/// The most BiLSTM layers an adapter stacks.
pub const MAX_ADAPTER_LAYERS: u32 = 3;
/// The widest BiLSTM adapter state.
pub const MAX_ADAPTER_HIDDEN: u32 = 4096;
/// The most labels one head scores.
pub const MAX_LABELS: usize = 4096;
/// The namespace every head's layer id is in.
pub const LAYER_NAMESPACE: &str = "turboembed:";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadSpec {
    /// Unique within the bundle: lowercase ASCII letters, digits, `_`, `-`.
    pub name: String,
    /// The layer id the analysis annotator writes the head's output to,
    /// in the `turboembed:` namespace.
    pub layer: String,
    pub kind: HeadKind,
    pub labels: Vec<String>,
    pub scheme: Scheme,
    pub word_pooling: WordPooling,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,
    pub decode: Decode,
    /// The name of the FORMAT_SAFETENSORS artifact holding the head's
    /// tensors, as `from` and `host_weights` name artifacts.
    pub artifact: String,
    pub tensor_prefix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadKind {
    /// One label per word, from the word's pooled subword states.
    #[serde(rename = "HEAD_TOKEN_TAGGER")]
    TokenTagger,
}

/// How a head's labels read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scheme {
    /// Each label is a tag in its own right (UPOS, XPOS, feats).
    #[serde(rename = "SCHEME_TAGS")]
    Tags,
    /// `O`, `B-T` and `I-T`: spans of type T.
    #[serde(rename = "SCHEME_BIO")]
    Bio,
    /// `O`, `B-T`, `I-T`, `L-T` and `U-T`: spans with marked ends and units.
    #[serde(rename = "SCHEME_BILOU")]
    Bilou,
}

/// Which of a word's subword states the head reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordPooling {
    #[serde(rename = "WORD_POOLING_FIRST")]
    First,
    #[serde(rename = "WORD_POOLING_MEAN")]
    Mean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adapter {
    pub bilstm: Bilstm,
}

/// Stacked bidirectional LSTM layers between the pooled words and the
/// output layer; each layer's output joins its two directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bilstm {
    pub hidden: u32,
    pub layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decode {
    /// The best-scoring label of each word on its own.
    #[serde(rename = "DECODE_ARGMAX")]
    Argmax,
    /// The best-scoring label sequence under a first-order CRF; under BIO
    /// and BILOU, sequences the scheme forbids are never chosen.
    #[serde(rename = "DECODE_CRF")]
    Crf,
}

/// A label read under a scheme: outside any span, or a part of a span of
/// a type, the type being the text after the two-character prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part<'a> {
    Outside,
    Begin(&'a str),
    Inside(&'a str),
    Last(&'a str),
    Unit(&'a str),
}

impl Scheme {
    /// `label` read under this scheme, or None if the scheme has no such
    /// label. Under SCHEME_TAGS every label is `Outside` of any span.
    pub fn part(self, label: &str) -> Option<Part<'_>> {
        if self == Scheme::Tags {
            return Some(Part::Outside);
        }
        if label == "O" {
            return Some(Part::Outside);
        }
        let (prefix, kind) = (label.get(..2)?, label.get(2..)?);
        if kind.is_empty() {
            return None;
        }
        match (prefix, self) {
            ("B-", _) => Some(Part::Begin(kind)),
            ("I-", _) => Some(Part::Inside(kind)),
            ("L-", Scheme::Bilou) => Some(Part::Last(kind)),
            ("U-", Scheme::Bilou) => Some(Part::Unit(kind)),
            _ => None,
        }
    }
}

fn is_name(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

impl HeadSpec {
    /// The rules a head meets on its own, whatever bundle carries it, with
    /// `at` the field path its errors name (`heads[0]`). A trainer checks
    /// a head by these before it builds one, so it never makes a head no
    /// bundle could carry.
    pub fn validate(&self, at: &str) -> Result<()> {
        let at = |field: &str| format!("{at}.{field}");
        required(&at("name"), &self.name)?;
        fits(&at("name"), &self.name, 64)?;
        if !is_name(&self.name) {
            return Err(invalid(format!(
                "manifest.json: {}: {:?} is not lowercase ASCII letters, digits, '_' and '-'",
                at("name"),
                self.name
            )));
        }
        required(&at("layer"), &self.layer)?;
        fits(&at("layer"), &self.layer, 128)?;
        if self.layer.strip_prefix(LAYER_NAMESPACE).is_none_or(|rest| !is_name(rest)) {
            return Err(invalid(format!(
                "manifest.json: {}: {:?} is not {LAYER_NAMESPACE} and a lowercase name",
                at("layer"),
                self.layer
            )));
        }
        required(&at("tensor_prefix"), &self.tensor_prefix)?;
        fits(&at("tensor_prefix"), &self.tensor_prefix, 128)?;
        if self.labels.is_empty() || self.labels.len() > MAX_LABELS {
            return Err(invalid(format!(
                "manifest.json: {}: {} labels, 1 to {MAX_LABELS}",
                at("labels"),
                self.labels.len()
            )));
        }
        for (j, l) in self.labels.iter().enumerate() {
            let field = at(&format!("labels[{j}]"));
            required(&field, l)?;
            fits(&field, l, 128)?;
            if self.labels[..j].contains(l) {
                return Err(invalid(format!("manifest.json: {field}: {l:?} is listed twice")));
            }
            if self.scheme.part(l).is_none() {
                return Err(invalid(format!("manifest.json: {field}: {l:?} is not a label of {:?}", self.scheme)));
            }
        }
        // With an outside label every length has a sequence the scheme
        // admits, so a CRF decode always has an answer.
        if self.scheme != Scheme::Tags && !self.labels.iter().any(|l| l == "O") {
            return Err(invalid(format!("manifest.json: {}: {:?} needs the label \"O\"", at("labels"), self.scheme)));
        }
        if let Some(a) = &self.adapter {
            positive(&at("adapter.bilstm.hidden"), a.bilstm.hidden)?;
            positive(&at("adapter.bilstm.layers"), a.bilstm.layers)?;
            if a.bilstm.hidden > MAX_ADAPTER_HIDDEN || a.bilstm.layers > MAX_ADAPTER_LAYERS {
                return Err(invalid(format!(
                    "manifest.json: {}: hidden {} and layers {}; at most {MAX_ADAPTER_HIDDEN} and {MAX_ADAPTER_LAYERS}",
                    at("adapter.bilstm"),
                    a.bilstm.hidden,
                    a.bilstm.layers
                )));
            }
        }
        Ok(())
    }
}

impl Manifest {
    pub(super) fn validate_heads(&self) -> Result<()> {
        if self.heads.is_empty() {
            return Ok(());
        }
        if self.architecture.is_none() {
            return Err(invalid("manifest.json: heads: requires an encoder architecture"));
        }
        if self.heads.len() > MAX_HEADS {
            return Err(invalid(format!("manifest.json: heads: {} heads, at most {MAX_HEADS}", self.heads.len())));
        }
        for (i, h) in self.heads.iter().enumerate() {
            let at = |field: &str| format!("heads[{i}].{field}");
            h.validate(&format!("heads[{i}]"))?;
            if self.heads[..i].iter().any(|o| o.name == h.name) {
                return Err(invalid(format!("manifest.json: {}: {:?} names two heads", at("name"), h.name)));
            }
            match self.artifacts.iter().find(|a| a.name == h.artifact) {
                Some(a) if a.format == Format::Safetensors => {}
                Some(_) => {
                    return Err(invalid(format!(
                        "manifest.json: {}: {:?} is not FORMAT_SAFETENSORS",
                        at("artifact"),
                        h.artifact
                    )));
                }
                None => {
                    return Err(invalid(format!("manifest.json: {}: no artifact {:?}", at("artifact"), h.artifact)));
                }
            }
        }
        Ok(())
    }
}
