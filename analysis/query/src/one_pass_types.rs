//! Data types for one-pass search query planning (`AnalyzeQuery`, `#2339`).

use std::sync::Arc;

use turbo_analysis::Span;
use turbo_analysis_layers::Currency;
use turbo_chunker::terms::TermForm;
use turbo_geo::GeoBoundingBox;
use turbo_matcher::PhraseSynonymExpander;

use crate::model::{
    EntityFilter, FilterEvidence, GlossaryFilter, LanguageEvidence, MoneyFilter, PlaceFilter, QuantityFilter,
    TemporalFilter,
};

/// Relational comparison operator attached to a numeric or temporal query filter.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum QueryComparisonOp {
    #[default]
    Eq,
    Gt,
    Gte,
    Lt,
    Lte,
    Range,
}

impl QueryComparisonOp {
    /// Canonical wire/code name of this comparison operator.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::Range => "range",
        }
    }

    /// Parses a comparison operator from its wire name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "eq" => Some(Self::Eq),
            "gt" => Some(Self::Gt),
            "gte" => Some(Self::Gte),
            "lt" => Some(Self::Lt),
            "lte" => Some(Self::Lte),
            "range" => Some(Self::Range),
            _ => None,
        }
    }
}

/// Tokenization strategy for a search field's BM25 query terms.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum FieldTokenizerKind {
    /// UAX #29 `WordTokenizer` (matches `QueryAnalysisProfile::document_analyzer`).
    #[default]
    Word,
    /// Whitespace-delimited runs (`WhitespaceTokenizer`), preserving exact
    /// ordinal gaps for standalone punctuation or stripped invisible tokens.
    Whitespace,
}

/// Per-field lexical analysis and synonym configuration for [`crate::OnePassQueryAnalyzer`].
#[derive(Clone)]
pub struct FieldQuerySpec {
    /// Target index field name (for example `"title"`, `"body"`, `"code"`).
    pub field: String,
    /// Tokenization strategy for this field.
    pub tokenizer: FieldTokenizerKind,
    /// Linguistic form indexed in this field (`Surface`, `Folded`, `Stem`, `Lemma`).
    pub term_form: TermForm,
    /// Optional ISO language override (`None` uses detected/default query language).
    pub language: Option<String>,
    /// Strip Unicode invisible format characters (`\u{00AD}`, `\u{200B}`, etc.) per token.
    pub strip_invisible: bool,
    /// Exclude stopwords from `residual_terms` (stopwords still advance `ordinal`).
    pub filter_stopwords: bool,
    /// Optional non-backtracking phrase/synonym automaton (`turbo-matcher`).
    pub synonyms: Option<Arc<PhraseSynonymExpander>>,
}

impl FieldQuerySpec {
    /// Creates a default stemmed word-tokenized field profile for `field`.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            tokenizer: FieldTokenizerKind::Word,
            term_form: TermForm::Stem,
            language: None,
            strip_invisible: true,
            filter_stopwords: true,
            synonyms: None,
        }
    }

    /// Overrides the [`FieldTokenizerKind`].
    pub fn with_tokenizer(mut self, tokenizer: FieldTokenizerKind) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Overrides the [`TermForm`].
    pub fn with_term_form(mut self, term_form: TermForm) -> Self {
        self.term_form = term_form;
        self
    }

    /// Overrides the per-field language code.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Attaches a [`PhraseSynonymExpander`] for multi-token query synonym expansion.
    pub fn with_synonyms(mut self, synonyms: Arc<PhraseSynonymExpander>) -> Self {
        self.synonyms = Some(synonyms);
        self
    }
}

/// One analyzed BM25 query term carrying its exact 0-based token ordinal
/// (counting tokens whose normalized form became empty) and `ViewId::SOURCE` span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrdinalQueryTerm {
    /// Analyzed lexical term in the field's [`TermForm`].
    pub term: String,
    /// Exact 0-based token ordinal in the field's token stream, counting
    /// empty-normalized tokens so `slop = 0` phrase and bigram ordinals match.
    pub ordinal: u32,
    /// Number of token positions spanned (`1` for single tokens).
    pub pos_len: u32,
    /// Exact UTF-8 byte span in `ViewId::SOURCE`.
    pub span: Span,
    /// Exact substring of `ViewId::SOURCE` covered by `span`.
    pub surface: String,
    /// Whether this token was flagged as a stopword.
    pub is_stopword: bool,
    /// Whether this token's source span overlaps an extracted structured filter.
    pub covered_by_filter: bool,
}

/// One multi-token or single-token synonym phrase matched by `turbo-matcher`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuerySynonymMatch {
    /// Inclusive start token ordinal in the field's token stream.
    pub start_ordinal: u32,
    /// Exclusive end token ordinal in the field's token stream.
    pub end_ordinal: u32,
    /// Number of base token positions covered (`end_ordinal - start_ordinal`).
    pub pos_len: u32,
    /// Exact UTF-8 byte span in `ViewId::SOURCE`.
    pub span: Span,
    /// Exact substring of `ViewId::SOURCE` covered by `span`.
    pub surface: String,
    /// Analyzed expansion terms in the field's [`TermForm`].
    pub terms: Vec<String>,
}

/// Ordinal-preserving per-field BM25 and synonym plan produced in one pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldQueryPlan {
    /// Field name from [`FieldQuerySpec::field`].
    pub field: String,
    /// Total token slots in this field, including empty-normalized tokens.
    pub total_token_ordinals: u32,
    /// Exact 0-based ordinals of tokens whose normalized/indexed form became empty.
    pub empty_normalized_ordinals: Vec<u32>,
    /// All non-empty analyzed BM25 terms in ordinal order.
    pub bm25_terms: Vec<OrdinalQueryTerm>,
    /// Non-stopword BM25 terms not covered by any structured [`StructuredFilterPlan`].
    pub residual_terms: Vec<OrdinalQueryTerm>,
    /// Multi-token and single-token synonym expansions matched in this field.
    pub synonyms: Vec<QuerySynonymMatch>,
}

/// ECB-converted monetary amount paired with a [`MoneyFilter`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConvertedMoneyValue {
    /// Target ISO 4217 currency code (for example `USD` or `EUR`).
    pub currency: Currency,
    /// Exact plain decimal representation of the converted amount.
    pub decimal_text: String,
    /// Converted `f64` scalar for numeric range binding.
    pub amount_f64: f64,
}

/// Typed filter payload in a [`StructuredFilterPlan`].
#[derive(Clone, Debug, PartialEq)]
pub enum StructuredFilterPayload {
    Temporal(TemporalFilter),
    Money { raw: MoneyFilter, amount_f64: f64, ecb_converted: Option<ConvertedMoneyValue> },
    Quantity { raw: QuantityFilter, si_base_unit: String, si_plain_value: String, si_value_f64: f64 },
    Place { raw: PlaceFilter, bbox: Option<GeoBoundingBox>, spine_path: String },
    Glossary(GlossaryFilter),
    Entity { raw: EntityFilter, kb_id: Option<String>, provider_id: Option<String>, canonical_name: Option<String> },
}

/// A structured filter extracted in [`OnePassQueryPlan`], carrying both its
/// full consumed span (including relational cues like `"over "` or `"since "`)
/// and its inner value span in `ViewId::SOURCE`.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredFilterPlan {
    /// Full consumed UTF-8 byte span in `ViewId::SOURCE` (used to exclude filter
    /// tokens and relational cues from `residual_terms`).
    pub span: Span,
    /// UTF-8 byte span of the inner mention value in `ViewId::SOURCE`.
    pub value_span: Span,
    /// Exact substring `&query[span.range()]`.
    pub surface: String,
    /// Relational comparison operator (`Eq`, `Gt`, `Gte`, `Lt`, `Lte`, `Range`).
    pub comparison: QueryComparisonOp,
    /// Provenance tying this filter to the layer and annotator that produced it.
    pub evidence: FilterEvidence,
    /// Typed filter payload.
    pub payload: StructuredFilterPayload,
}

/// Unified one-pass lexical, synonym, and structured-filter query plan (`#2339`).
#[derive(Clone, Debug, PartialEq)]
pub struct OnePassQueryPlan {
    /// Original source query text (`ViewId::SOURCE`).
    pub query: String,
    /// Document- and span-level language detection evidence.
    pub language: Option<LanguageEvidence>,
    /// Per-field ordinal-preserving BM25 terms and synonym matches.
    pub fields: Vec<FieldQueryPlan>,
    /// Extracted structured NLP filters in `ViewId::SOURCE` span order.
    pub filters: Vec<StructuredFilterPlan>,
    /// Primary field's residual non-filter, non-stopword [`OrdinalQueryTerm`]s.
    pub residual_terms: Vec<OrdinalQueryTerm>,
    /// Residual non-filter surface tokens joined by single ASCII spaces.
    pub free_text: String,
}
