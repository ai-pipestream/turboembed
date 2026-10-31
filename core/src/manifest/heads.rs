//! `heads`: the task heads a bundle carries beside its encoder, each read
//! on the host from the encoder's final-layer states (head/tagger.rs).
//! A list, so one bundle can carry several heads over one encoder pass.
//! Each head names its weights by a tensor prefix and fixed suffixes
//! (docs/bundle.md, "Heads"), not by `tensor_names` roles, so any number
//! of heads fit one artifact.

use serde::{Deserialize, Serialize};

use super::helpers::{fits, positive, required};
use super::{Format, LicenseClass, Manifest};
use crate::status::{Result, invalid};

/// The most heads one bundle carries.
pub const MAX_HEADS: usize = 32;
/// The most BiLSTM layers an adapter stacks.
pub const MAX_ADAPTER_LAYERS: u32 = 3;
/// The widest BiLSTM adapter state.
pub const MAX_ADAPTER_HIDDEN: u32 = 4096;
/// The most labels one head scores.
pub const MAX_LABELS: usize = 4096;
/// The widest biaffine MLP output.
pub const MAX_BIAFFINE: u32 = 4096;
/// The widest part-of-speech embedding a lemmatizer reads.
pub const MAX_POS_DIM: u32 = 1024;
/// The widest hashed word embedding, per feature.
pub const MAX_HASH_DIM: u32 = 1024;
/// The most rows one hashed feature table holds.
pub const MAX_HASH_ROWS: u32 = 1 << 20;
/// The most words one span of a span classifier covers.
pub const MAX_SPAN_WORDS: u32 = 128;
/// The widest span-width embedding.
pub const MAX_SPAN_WIDTH_DIM: u32 = 256;
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
    /// The arc and relation MLP widths of a HEAD_BIAFFINE_PARSER; absent
    /// for every other kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub biaffine: Option<Biaffine>,
    /// The part-of-speech tags a HEAD_EDIT_TREE_LEMMATIZER reads beside
    /// each word, from another head of the bundle; absent for every other
    /// kind, and optional for a lemmatizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pos_input: Option<PosInput>,
    /// Hashed embeddings of the word's own text joined after its pooled
    /// states (head/hash_embed.rs); a HEAD_TOKEN_TAGGER's option.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_embed: Option<HashEmbed>,
    /// The spans a HEAD_SPAN_CLASSIFIER or HEAD_OPEN_SPAN head scores;
    /// absent for every other kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<SpanShape>,
    /// The class of the strictest licence among the corpora the head was
    /// trained on (docs/zoo.md); the bundle's tier counts it
    /// ([`Manifest::license_tier`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_class: Option<LicenseClass>,
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
    /// A head word and a relation per word, from biaffine arc and
    /// relation scores (Dozat and Manning 2017), decoded to a tree. The
    /// labels are the relations, under SCHEME_TAGS.
    #[serde(rename = "HEAD_BIAFFINE_PARSER")]
    BiaffineParser,
    /// An edit tree per word (head/edit_tree.rs), the labels being the
    /// trees' labels under SCHEME_TAGS; the word's lemma is its tree
    /// applied to it.
    #[serde(rename = "HEAD_EDIT_TREE_LEMMATIZER")]
    EditTreeLemmatizer,
    /// Whether each word ends its sentence (head/boundary.rs): the labels
    /// are exactly `inside` and `last`, under SCHEME_TAGS.
    #[serde(rename = "HEAD_SENTENCE_BOUNDARY")]
    SentenceBoundary,
    /// How each token splits into syntactic words (head/split.rs), as UD
    /// splits a multi-word token: `_` keeps it whole, any other label is
    /// two or more edit trees, one a word, each applied to the token.
    #[serde(rename = "HEAD_TOKEN_SPLIT")]
    TokenSplit,
    /// A label for every span of up to `span.max_words` words, from its
    /// first and last word and its width (head/span.rs); `O` is no span.
    #[serde(rename = "HEAD_SPAN_CLASSIFIER")]
    SpanClassifier,
    /// Open-label span typing (head/open_span.rs): scores every span of up
    /// to `span.max_words` words against caller-supplied label vectors in
    /// the encoder's hidden space. `labels` records the training type
    /// names under SCHEME_TAGS.
    #[serde(rename = "HEAD_OPEN_SPAN")]
    OpenSpan,
}

impl HeadKind {
    /// Whether the head is a linear layer over each word's vector, as a
    /// token tagger is, its own kind deciding only how labels decode.
    pub fn scores_words(self) -> bool {
        matches!(self, HeadKind::TokenTagger | HeadKind::SentenceBoundary | HeadKind::TokenSplit)
    }
}

/// The labels of a HEAD_SENTENCE_BOUNDARY, in this order.
pub const BOUNDARY_LABELS: [&str; 2] = ["inside", "last"];
/// The label of a HEAD_TOKEN_SPLIT that keeps a token one word.
pub const KEEP_TOKEN: &str = "_";

/// The spans a span classifier reads: every run of 1 to `max_words`
/// words, each with a learned embedding `width_dim` wide of its width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanShape {
    pub max_words: u32,
    pub width_dim: u32,
}

/// The tags a lemmatizer reads: head `head` (a SCHEME_TAGS token tagger
/// of the same bundle, with `labels` labels) decodes each word's tag, and
/// the lemmatizer reads the tag's learned `dim`-wide embedding beside the
/// word's features.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosInput {
    pub head: String,
    pub labels: u32,
    pub dim: u32,
}

/// The widths of a biaffine parser's MLPs: `arc` for the dependent and
/// head views the arc scorer reads, `rel` for the relation scorer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Biaffine {
    pub arc: u32,
    pub rel: u32,
}

/// A word's identity as a head reads it beside the encoder: four
/// features of its text (NORM, PREFIX, SUFFIX, SHAPE), each looked up in
/// its own table of `rows[k]` rows `dim` wide by several hashes of the
/// feature and summed (head/hash_embed.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashEmbed {
    pub dim: u32,
    pub rows: [u32; 4],
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
    /// The best-scoring tree with one word on the root (Chu-Liu-Edmonds);
    /// a biaffine parser's decode.
    #[serde(rename = "DECODE_MST")]
    Mst,
    /// The best-scoring projective tree with one word on the root
    /// (Eisner); a biaffine parser's decode.
    #[serde(rename = "DECODE_EISNER")]
    Eisner,
    /// The best-scoring edit tree that fits the word, else the word
    /// itself; a lemmatizer's decode.
    #[serde(rename = "DECODE_EDIT_TREE")]
    EditTree,
    /// Every span whose best label is not `O`, overlapping or nested; a
    /// span classifier's decode.
    #[serde(rename = "DECODE_SPANS_NESTED")]
    SpansNested,
    /// The best-scoring spans that do not overlap, taken greedily from
    /// the highest score; a span classifier's decode.
    #[serde(rename = "DECODE_SPANS_FLAT")]
    SpansFlat,
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
        self.validate_kind_decode(&at)?;
        if self.kind == HeadKind::EditTreeLemmatizer {
            for (j, l) in self.labels.iter().enumerate() {
                crate::head::edit_tree::EditTree::parse(l)
                    .map_err(|e| invalid(format!("manifest.json: {}: {}", at(&format!("labels[{j}]")), e.message)))?;
            }
        }
        if let Some(p) = &self.pos_input {
            required(&at("pos_input.head"), &p.head)?;
            positive(&at("pos_input.labels"), p.labels)?;
            positive(&at("pos_input.dim"), p.dim)?;
            if p.labels as usize > MAX_LABELS || p.dim > MAX_POS_DIM || p.head == self.name {
                return Err(invalid(format!(
                    "manifest.json: {}: labels {} and dim {} (at most {MAX_LABELS} and {MAX_POS_DIM}) from head {:?}, \
                     another head",
                    at("pos_input"),
                    p.labels,
                    p.dim,
                    p.head
                )));
            }
        }
        if let Some(b) = &self.biaffine {
            positive(&at("biaffine.arc"), b.arc)?;
            positive(&at("biaffine.rel"), b.rel)?;
            if b.arc > MAX_BIAFFINE || b.rel > MAX_BIAFFINE {
                return Err(invalid(format!(
                    "manifest.json: {}: arc {} and rel {}; at most {MAX_BIAFFINE}",
                    at("biaffine"),
                    b.arc,
                    b.rel
                )));
            }
        }
        self.validate_kind_rules(&at)?;
        if let Some(h) = &self.hash_embed {
            if !self.kind.scores_words() {
                return Err(invalid(format!(
                    "manifest.json: {}: only a head that scores words, as a HEAD_TOKEN_TAGGER does, takes it",
                    at("hash_embed")
                )));
            }
            positive(&at("hash_embed.dim"), h.dim)?;
            if h.dim > MAX_HASH_DIM || h.rows.iter().any(|&r| r == 0 || r > MAX_HASH_ROWS) {
                return Err(invalid(format!(
                    "manifest.json: {}: dim {} and rows {:?}; dim 1 to {MAX_HASH_DIM}, rows 1 to {MAX_HASH_ROWS}",
                    at("hash_embed"),
                    h.dim,
                    h.rows
                )));
            }
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
    /// The bundle's licence tier: the strictest class of its encoder
    /// (`model.license_class`) and of every head's training corpora. None
    /// when the encoder's class is not recorded and no head records one.
    pub fn license_tier(&self) -> Option<LicenseClass> {
        self.model.license_class.into_iter().chain(self.heads.iter().filter_map(|h| h.license_class)).max()
    }

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
            if h.license_class.is_some() && self.model.license_class.is_none() {
                return Err(invalid(format!(
                    "manifest.json: {}: set, and model.license_class is not, so the bundle's tier would leave the \
                     encoder out",
                    at("license_class")
                )));
            }
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
            if let Some(p) = &h.pos_input {
                let tags = self.heads.iter().find(|o| o.name == p.head).filter(|o| {
                    o.kind == HeadKind::TokenTagger && o.scheme == Scheme::Tags && o.labels.len() == p.labels as usize
                });
                if tags.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {}: {:?} is not a SCHEME_TAGS token tagger head of {} labels",
                        at("pos_input.head"),
                        p.head,
                        p.labels
                    )));
                }
            }
        }
        Ok(())
    }
}
