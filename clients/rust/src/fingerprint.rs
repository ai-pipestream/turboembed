//! Deterministic domain-separated cryptographic fingerprints and index
//! generation handshakes (`#2338`).
//!
//! Computes `AnalysisFingerprint` over a field's normalizer chain, tokenizer,
//! stemmer, term source, and active `turbo_unicode::UNICODE_VERSION`, and
//! `BundleScoringFingerprint` over a loaded neural bundle's manifest digest
//! and scoring geometry (`task`, `dim`, `pooling`, `normalize`).

use turbo_sha256::{Digest, Sha256};
use turbo_stem::SnowballAlgorithm;
use turbo_unicode::UNICODE_VERSION;

use crate::error::{ClientError, Result};

const ANALYSIS_FINGERPRINT_DOMAIN: &str = "turboembed.analysis_fingerprint.v1";
const BUNDLE_FINGERPRINT_DOMAIN: &str = "turboembed.bundle_scoring_fingerprint.v1";

/// Character-level normalizer stage in a [`FieldAnalysisConfig`], preserving
/// exact byte alignment back to `ViewId::SOURCE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NormalizerKind {
    /// Removes format and invisible characters (`turbo-normalize::Invisible`).
    StripInvisible,
    /// Collapses Unicode `White_Space` runs (`WhitespaceMode::Collapse`).
    Whitespace,
    /// Unicode full case folding (`CaseFoldMode::Full`).
    FullCaseFold,
    /// Single-character Unicode simple case folding (`CaseFoldMode::Simple`).
    SimpleCaseFold,
    /// Unicode NFKC compatibility decomposition and canonical composition.
    Nfkc,
    /// Unicode NFC canonical decomposition and composition.
    Nfc,
    /// Strips Latin, Greek, and Cyrillic accents and folds 19 ligature/stroke letters.
    AccentFold,
    /// Repairs end-of-line hyphenated breaks (`DehyphenateMode::LineBreak`).
    Dehyphenate,
    /// Decodes HTML character and numeric entities (`HtmlEntityDecodeNormalizer`).
    HtmlEntityDecode,
    /// Strips HTML/XML markup tags (`HtmlTagStripNormalizer`).
    HtmlTagStrip,
    /// Repairs Windows-1252 / UTF-8 mojibake runs (`Mojibake`).
    Mojibake,
}

impl NormalizerKind {
    /// Stable wire tag folded into [`AnalysisFingerprint`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::StripInvisible => "strip_invisible",
            Self::Whitespace => "whitespace_collapse",
            Self::FullCaseFold => "full_case_fold",
            Self::SimpleCaseFold => "simple_case_fold",
            Self::Nfkc => "nfkc",
            Self::Nfc => "nfc",
            Self::AccentFold => "accent_fold",
            Self::Dehyphenate => "dehyphenate_line_break",
            Self::HtmlEntityDecode => "html_entity_decode",
            Self::HtmlTagStrip => "html_tag_strip",
            Self::Mojibake => "mojibake",
        }
    }

    const fn is_case_or_accent(self) -> bool {
        matches!(self, Self::FullCaseFold | Self::SimpleCaseFold | Self::AccentFold)
    }
}

/// Word segmentation strategy for a [`FieldAnalysisConfig`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TokenizerKind {
    /// Non-whitespace runs (`WhitespaceTokenizer`).
    Whitespace,
    /// Unicode UAX #29 word segmenter (`WordTokenizer`).
    #[default]
    Uax29,
    /// Character-class run tokenizer (`SimpleTokenizer`).
    Simple,
}

impl TokenizerKind {
    /// Stable wire tag folded into [`AnalysisFingerprint`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Whitespace => "whitespace",
            Self::Uax29 => "uax29",
            Self::Simple => "simple",
        }
    }
}

/// Stemming strategy applied after tokenization.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum StemmerKind {
    /// Keep normalized surface tokens unstemmed.
    #[default]
    None,
    /// Classic Porter (1980) English stemmer (`PorterStemmer`).
    Porter,
    /// Any of the 21 generated Snowball stemmers (`SnowballStemmer`).
    Snowball(SnowballAlgorithm),
}

impl StemmerKind {
    /// Stable wire tag folded into [`AnalysisFingerprint`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Porter => "porter_1980",
            Self::Snowball(alg) => match alg {
                SnowballAlgorithm::Arabic => "snowball:arabic",
                SnowballAlgorithm::Danish => "snowball:danish",
                SnowballAlgorithm::Dutch => "snowball:dutch",
                SnowballAlgorithm::Catalan => "snowball:catalan",
                SnowballAlgorithm::English => "snowball:english",
                SnowballAlgorithm::Finnish => "snowball:finnish",
                SnowballAlgorithm::French => "snowball:french",
                SnowballAlgorithm::German => "snowball:german",
                SnowballAlgorithm::Greek => "snowball:greek",
                SnowballAlgorithm::Hungarian => "snowball:hungarian",
                SnowballAlgorithm::Indonesian => "snowball:indonesian",
                SnowballAlgorithm::Irish => "snowball:irish",
                SnowballAlgorithm::Italian => "snowball:italian",
                SnowballAlgorithm::Norwegian => "snowball:norwegian",
                SnowballAlgorithm::Porter => "snowball:porter",
                SnowballAlgorithm::Portuguese => "snowball:portuguese",
                SnowballAlgorithm::Romanian => "snowball:romanian",
                SnowballAlgorithm::Russian => "snowball:russian",
                SnowballAlgorithm::Spanish => "snowball:spanish",
                SnowballAlgorithm::Swedish => "snowball:swedish",
                SnowballAlgorithm::Turkish => "snowball:turkish",
            },
        }
    }
}

/// Which lexical layer supplies indexed terms.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TermSourceKind {
    /// Normalized view token slices without stemming.
    Tokens,
    /// Stemmed token forms when a stemmer is active, else normalized token slices.
    #[default]
    Stems,
    /// Lowercased/folded token slices passed through the active stemmer.
    NormalizedStems,
}

impl TermSourceKind {
    /// Stable wire tag folded into [`AnalysisFingerprint`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Tokens => "tokens",
            Self::Stems => "stems",
            Self::NormalizedStems => "normalized_stems",
        }
    }
}

/// Complete lexical analysis specification for an indexed or queried field.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldAnalysisConfig {
    /// Ordered normalizers composed into a single derived `View`.
    pub normalizers: Vec<NormalizerKind>,
    /// Word tokenizer run over the normalized view.
    pub tokenizer: TokenizerKind,
    /// Optional stemmer applied to tokens.
    pub stemmer: StemmerKind,
    /// Lexical source used for index/query terms.
    pub term_source: TermSourceKind,
    /// Sentence break language code (`"eng"` by default).
    pub sentence_language: String,
    /// Whether to emit dehyphenation and inline-hyphen lattice arcs (`#2340`).
    pub emit_lattice: bool,
}

impl Default for FieldAnalysisConfig {
    fn default() -> Self {
        Self {
            normalizers: vec![
                NormalizerKind::StripInvisible,
                NormalizerKind::Nfkc,
                NormalizerKind::FullCaseFold,
                NormalizerKind::AccentFold,
            ],
            tokenizer: TokenizerKind::Uax29,
            stemmer: StemmerKind::Porter,
            term_source: TermSourceKind::Stems,
            sentence_language: "eng".to_owned(),
            emit_lattice: true,
        }
    }
}

impl FieldAnalysisConfig {
    /// Derives the case-preserving, unstemmed twin configuration used for
    /// dual-cased exact matching alongside a folded primary field.
    pub fn cased_twin(&self) -> Self {
        Self {
            normalizers: self.normalizers.iter().copied().filter(|n| !n.is_case_or_accent()).collect(),
            tokenizer: self.tokenizer,
            stemmer: StemmerKind::None,
            term_source: TermSourceKind::Tokens,
            sentence_language: self.sentence_language.clone(),
            emit_lattice: self.emit_lattice,
        }
    }

    /// Computes the deterministic [`AnalysisFingerprint`] for this configuration.
    pub fn fingerprint(&self) -> AnalysisFingerprint {
        AnalysisFingerprint::from_config(self)
    }
}

/// Domain-separated cryptographic digest and non-zero 64-bit identifier for a
/// [`FieldAnalysisConfig`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnalysisFingerprint {
    /// Full 256-bit SHA-256 digest.
    pub sha256: [u8; 32],
    /// Non-zero 64-bit truncated ID suitable for compact segment headers.
    pub id64: u64,
}

impl AnalysisFingerprint {
    /// Computes the fingerprint of `cfg` pinned to `UNICODE_VERSION`.
    pub fn from_config(cfg: &FieldAnalysisConfig) -> Self {
        let mut h = Sha256::new();
        h.str(ANALYSIS_FINGERPRINT_DOMAIN);
        h.u8(UNICODE_VERSION.0).u8(UNICODE_VERSION.1).u8(UNICODE_VERSION.2);
        h.usize(cfg.normalizers.len());
        for n in &cfg.normalizers {
            h.str(n.tag());
        }
        h.str(cfg.tokenizer.tag());
        h.str(cfg.stemmer.tag());
        h.str(cfg.term_source.tag());
        h.str(&cfg.sentence_language);
        h.u8(u8::from(cfg.emit_lattice));
        Self::from_digest(h.finish())
    }

    /// Wraps a raw 32-byte SHA-256 digest into an [`AnalysisFingerprint`].
    pub fn from_sha256(sha256: [u8; 32]) -> Self {
        Self::from_digest(Digest(sha256))
    }

    fn from_digest(d: Digest) -> Self {
        let mut first8 = [0u8; 8];
        first8.copy_from_slice(&d.0[..8]);
        let raw = u64::from_le_bytes(first8);
        let id64 = if raw == 0 { 1 } else { raw };
        Self { sha256: d.0, id64 }
    }

    /// Lowercase 64-character hex string of `sha256`.
    pub fn to_hex(&self) -> String {
        Digest(self.sha256).to_hex()
    }
}

/// Canonical neural model scoring descriptor hashed into [`BundleScoringFingerprint`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BundleScoringSpec {
    /// Model identifier (`manifest.model.id` or `model_info.model_id`).
    pub model_id: String,
    /// Model revision (`manifest.model.revision` or `model_info.revision`).
    pub revision: String,
    /// 64-hex SHA-256 of `manifest.json` (`model_info.manifest_sha256`).
    pub manifest_sha256: String,
    /// Canonical task label (`"embed"`, `"rerank"`, `"extractive_qa"`).
    pub task: String,
    /// Primary vector output dimension (`0` for sequence/span scorers).
    pub dim: u32,
    /// Canonical pooling label (`"mean"`, `"cls"`, `"last"`, `"model"`).
    pub pooling: String,
    /// Canonical normalization label (`"l2"`, `"none"`, `"model"`).
    pub normalize: String,
}

impl BundleScoringSpec {
    /// Computes the deterministic [`BundleScoringFingerprint`] for this spec.
    pub fn fingerprint(&self) -> BundleScoringFingerprint {
        BundleScoringFingerprint::from_spec(self)
    }
}

/// Domain-separated cryptographic fingerprint over a neural bundle's manifest
/// and scoring contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BundleScoringFingerprint {
    /// Full 256-bit SHA-256 digest.
    pub sha256: [u8; 32],
    /// Non-zero 64-bit truncated ID.
    pub id64: u64,
}

impl BundleScoringFingerprint {
    /// Hashes `spec` with length-prefixed framing under `BUNDLE_FINGERPRINT_DOMAIN`.
    pub fn from_spec(spec: &BundleScoringSpec) -> Self {
        fn canon(raw: &str, prefix: &str) -> String {
            let low = raw.trim().to_ascii_lowercase();
            low.strip_prefix(prefix).unwrap_or(&low).to_owned()
        }
        let mut h = Sha256::new();
        h.str(BUNDLE_FINGERPRINT_DOMAIN);
        h.str(spec.model_id.trim());
        h.str(spec.revision.trim());
        h.str(&spec.manifest_sha256.trim().to_ascii_lowercase());
        h.str(&canon(&spec.task, "task_"));
        h.u32(spec.dim);
        h.str(&canon(&spec.pooling, "pooling_"));
        h.str(&canon(&spec.normalize, "normalize_"));
        let d = h.finish();
        let mut first8 = [0u8; 8];
        first8.copy_from_slice(&d.0[..8]);
        let raw = u64::from_le_bytes(first8);
        Self { sha256: d.0, id64: if raw == 0 { 1 } else { raw } }
    }

    /// Lowercase 64-character hex string of `sha256`.
    pub fn to_hex(&self) -> String {
        Digest(self.sha256).to_hex()
    }
}

/// Persisted index-generation contract verified at client bind/connect time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexGenerationHandshake {
    /// Expected lexical analyzer fingerprint, when lexical indexing is active.
    pub analysis: Option<AnalysisFingerprint>,
    /// Expected neural bundle scoring fingerprint, when vector/rerank scoring is active.
    pub bundle: Option<BundleScoringFingerprint>,
}

impl IndexGenerationHandshake {
    /// Refuses with [`ClientError::FailedPrecondition`] if `actual` disagrees
    /// with the pinned `analysis` fingerprint.
    pub fn verify_analysis(&self, actual: &AnalysisFingerprint) -> Result<()> {
        if let Some(expected) = &self.analysis
            && expected != actual
        {
            return Err(ClientError::FailedPrecondition(format!(
                "analysis fingerprint mismatch: index generation expects {} (id64={:#018x}), client configured {} (id64={:#018x})",
                expected.to_hex(),
                expected.id64,
                actual.to_hex(),
                actual.id64,
            )));
        }
        Ok(())
    }

    /// Refuses with [`ClientError::FailedPrecondition`] if `actual` disagrees
    /// with the pinned `bundle` scoring fingerprint.
    pub fn verify_bundle(&self, actual: &BundleScoringFingerprint) -> Result<()> {
        if let Some(expected) = &self.bundle
            && expected != actual
        {
            return Err(ClientError::FailedPrecondition(format!(
                "bundle scoring fingerprint mismatch: index generation expects {} (id64={:#018x}), loaded model reported {} (id64={:#018x})",
                expected.to_hex(),
                expected.id64,
                actual.to_hex(),
                actual.id64,
            )));
        }
        Ok(())
    }
}
