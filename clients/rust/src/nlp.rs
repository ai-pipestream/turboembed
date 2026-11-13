//! In-process one-pass document NLP extractor and search column projector (`#2338`, `#2345`).
//!
//! Runs money, quantity, temporal, document-date, gazetteer geocoding, polarity,
//! noise/artifact/quality scoring, named-entity extraction + KB linking,
//! epistemic/negation modality, and near-duplicate sketching in one bounded
//! [`Analyzer`] pass, projecting every span back onto `ViewId::SOURCE`.

#[path = "nlp_project.rs"]
mod nlp_project;

use std::sync::Arc;

use turbo_analysis::{Analyzer, Bound, Document};
use turbo_analysis_layers::{Currency, Date, NumberNotation};
use turbo_depparse::DependencyAnnotator;
use turbo_doccat::PolarityAnnotator;
use turbo_entity_linker::EntityLinker;
use turbo_geo::{BundledGazetteer, GeocodeAnnotator, SpatialCoherenceGeocoder};
use turbo_namefind::{BuiltIn, DictionaryNameFinder, RegexNameFinder};
use turbo_normalize::{ArtifactDetector, Invisible, NoiseScorer, NormalizeStep, TextQualityAnnotator};
use turbo_numeric::{
    DatedMoneyConverter, DocumentDateSelector, EcbRates, MoneyConverter, MoneyExtractor, QuantityExtractor,
    TemporalExtractor,
};
use turbo_query::GazetteerCandidateAnnotator;
use turbo_relation::ModalityAnnotator;
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::WordTokenizer;

use crate::error::{ClientError, Result};
use crate::lexical::ProjectedSpanSet;
use nlp_project::project_document;

pub use turbo_analysis::CorpusIdf as VocabularyCorpusIdf;
pub use turbo_query::{
    FieldQueryPlan, FieldQuerySpec, FieldTokenizerKind, OnePassQueryAnalyzer, OnePassQueryBuilder, OnePassQueryPlan,
    OrdinalQueryTerm, QueryComparisonOp, QuerySynonymMatch, StructuredFilterPayload, StructuredFilterPlan,
};
pub use turbo_suggest::{
    MappedSurfaceLexicon, QuerySpellResult, QuerySpellchecker, SpellEdit, SpellcheckConfig, StemSurface,
    SurfaceCompletion, SurfaceLexiconBuilder,
};

const GEO_CANDIDATES_INSTANCE: &str = "client_geo_candidates";
const CLEAN_VIEW_INSTANCE: &str = "client_invisible_clean";

/// Extracted monetary amount projected onto `ViewId::SOURCE`.
#[derive(Clone, Debug, PartialEq)]
pub struct MoneyHit {
    pub spans: ProjectedSpanSet,
    pub currency: String,
    pub amount_f64: f64,
    pub plain_amount: String,
    pub per: String,
    pub ecb_converted: bool,
}

/// Extracted SI-normalized quantity or percentage projected onto `ViewId::SOURCE`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantityHit {
    pub spans: ProjectedSpanSet,
    pub unit: String,
    pub value_f64: f64,
    pub plain_value: String,
    pub is_percent: bool,
}

/// Normalized UTC calendar date.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UtcDateSummary {
    pub iso: String,
    pub epoch_seconds: i64,
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Extracted temporal interval projected onto `ViewId::SOURCE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalHit {
    pub spans: ProjectedSpanSet,
    pub granularity: String,
    pub origin: String,
    pub iso: String,
    pub start_epoch_seconds: i64,
    pub end_epoch_seconds: i64,
}

/// Disambiguated geospatial mention with coordinates and containment path.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoHit {
    pub spans: ProjectedSpanSet,
    pub record_id: String,
    pub canonical_name: String,
    pub country_code: String,
    pub latitude: f64,
    pub longitude: f64,
    pub spine_path: String,
    pub confidence: f64,
}

/// Document-level sentiment summary.
#[derive(Clone, Debug, PartialEq)]
pub struct SentimentSummary {
    pub label: Option<String>,
    pub probability: Option<f64>,
}

/// Monotone noise, artifact, and text-quality summary with explicit assessment flag.
#[derive(Clone, Debug, PartialEq)]
pub struct QualitySummary {
    pub assessed: bool,
    pub worst_noise_score: f64,
    pub union_noise_chars: i64,
    pub artifact_count: i64,
    pub composite_score: Option<f64>,
}

/// Entity mention partitioned by syntactic/epistemic negation modality.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityPolarityHit {
    pub spans: ProjectedSpanSet,
    pub surface: String,
    pub kind: String,
    pub kb_id: Option<String>,
    pub canonical_name: Option<String>,
    pub score: f64,
    pub epistemic: String,
    pub cue: Option<String>,
}

/// Deterministic near-duplicate sketch over the document's word sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentSketch {
    pub word_count: u32,
    pub exact: u128,
    pub shape: u64,
    pub simhash: u64,
    pub minhash: Vec<u32>,
}

/// Complete typed NLP column set extracted in one pass over a document.
#[derive(Clone, Debug)]
pub struct DocumentNlpProjection {
    pub money: Vec<MoneyHit>,
    pub quantities: Vec<QuantityHit>,
    pub document_date: Option<UtcDateSummary>,
    pub temporals: Vec<TemporalHit>,
    pub geo: Vec<GeoHit>,
    pub sentiment: SentimentSummary,
    pub quality: QualitySummary,
    pub affirmed_entities: Vec<EntityPolarityHit>,
    pub negated_entities: Vec<EntityPolarityHit>,
    pub sketch: DocumentSketch,
}

/// Builder for [`OnePassDocumentNlp`].
pub struct OnePassNlpBuilder {
    language: String,
    notation: NumberNotation,
    reference_date: Option<Date>,
    ecb: Option<(Arc<EcbRates>, Currency, Option<Date>)>,
    dict_finder: Option<DictionaryNameFinder>,
    entity_linker: Option<EntityLinker>,
    depparse: Option<DependencyAnnotator>,
    shingle_width: u8,
}

impl Default for OnePassNlpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OnePassNlpBuilder {
    /// Creates a builder with `"eng"` sentence breaking, `NumberNotation::LatinUs`,
    /// bundled Natural Earth geocoding, polarity, quality/noise/artifact scoring,
    /// built-in regex entities, modality, and 3-word shingle sketching.
    pub fn new() -> Self {
        Self {
            language: "eng".to_owned(),
            notation: NumberNotation::LatinUs,
            reference_date: None,
            ecb: None,
            dict_finder: None,
            entity_linker: None,
            depparse: None,
            shingle_width: 3,
        }
    }

    /// Sets the sentence segmentation language code.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Sets the numeric grouping/decimal notation (`LatinUs`, `LatinEu`, etc.).
    pub fn with_number_notation(mut self, notation: NumberNotation) -> Self {
        self.notation = notation;
        self
    }

    /// Anchors relative temporal expressions to `date`.
    pub fn with_reference_date(mut self, date: Date) -> Self {
        self.reference_date = Some(date);
        self
    }

    /// Enables ECB foreign-exchange normalization into `target` currency.
    pub fn with_ecb_rates(mut self, rates: Arc<EcbRates>, target: Currency, fixed_date: Option<Date>) -> Self {
        self.ecb = Some((rates, target, fixed_date));
        self
    }

    /// Attaches a [`DictionaryNameFinder`] for named entity recognition.
    pub fn with_dictionary_finder(mut self, finder: DictionaryNameFinder) -> Self {
        self.dict_finder = Some(finder);
        self
    }

    /// Attaches a knowledge-base [`EntityLinker`] over extracted entities.
    pub fn with_entity_linker(mut self, linker: EntityLinker) -> Self {
        self.entity_linker = Some(linker);
        self
    }

    /// Attaches a syntactic [`DependencyAnnotator`] for dependency-scoped negation.
    pub fn with_dependency_parser(mut self, parser: DependencyAnnotator) -> Self {
        self.depparse = Some(parser);
        self
    }

    /// Compiles the thread-safe [`OnePassDocumentNlp`] pipeline.
    pub fn build(self) -> Result<OnePassDocumentNlp> {
        let temporal = match self.reference_date {
            Some(d) => TemporalExtractor::with_reference(d),
            None => TemporalExtractor::new(),
        };
        let gaz = Arc::new(BundledGazetteer::parse().map_err(|e| ClientError::Internal(e.to_string()))?);
        let mut b = Analyzer::builder()
            .step(Bound::new(NormalizeStep::new(Invisible)).write(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(RuleSentenceDetector::new(&self.language)).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(WordTokenizer::new()).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(MoneyExtractor::new(self.notation)).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(QuantityExtractor::new(self.notation)).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(temporal).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(DocumentDateSelector))
            .step(
                GazetteerCandidateAnnotator::new(Arc::clone(&gaz))
                    .bound(GEO_CANDIDATES_INSTANCE)
                    .read(0, CLEAN_VIEW_INSTANCE),
            )
            .step(
                Bound::new(GeocodeAnnotator::new(SpatialCoherenceGeocoder::new(gaz)))
                    .read(0, CLEAN_VIEW_INSTANCE)
                    .read(1, GEO_CANDIDATES_INSTANCE),
            )
            .step(PolarityAnnotator::new().bound().read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(NoiseScorer::new()).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(ArtifactDetector::default()).read(0, CLEAN_VIEW_INSTANCE))
            .step(Bound::new(TextQualityAnnotator).read(0, CLEAN_VIEW_INSTANCE));

        if let Some((rates, target, fixed_date)) = self.ecb {
            b = match fixed_date {
                Some(as_of) => b.step(MoneyConverter::new(rates, target, as_of).bound()),
                None => b.step(DatedMoneyConverter::new(rates, target).bound()),
            };
        }
        if let Some(dp) = self.depparse {
            b = b.step(Bound::new(dp).read(0, CLEAN_VIEW_INSTANCE));
        }
        if let Some(df) = self.dict_finder {
            b = b.step(Bound::new(df));
        } else {
            b = b.step(Bound::new(RegexNameFinder::defaults(&BuiltIn::ALL)).read(0, CLEAN_VIEW_INSTANCE));
        }
        if let Some(el) = self.entity_linker {
            b = b.step(Bound::new(el).read(0, CLEAN_VIEW_INSTANCE));
        }
        b = b.step(Bound::new(ModalityAnnotator::new()).read(0, CLEAN_VIEW_INSTANCE));

        let analyzer = b.build()?;
        Ok(OnePassDocumentNlp { analyzer, shingle_width: self.shingle_width.max(1) })
    }
}

/// Thread-safe one-pass document NLP extractor running on the bounded `Analyzer` worker pool.
pub struct OnePassDocumentNlp {
    analyzer: Analyzer,
    shingle_width: u8,
}

impl OnePassDocumentNlp {
    /// Creates a builder for [`OnePassDocumentNlp`].
    pub fn builder() -> OnePassNlpBuilder {
        OnePassNlpBuilder::new()
    }

    /// Runs the full NLP DAG once over `source` and projects all typed columns onto `ViewId::SOURCE`.
    pub fn analyze(&self, source: &str) -> Result<DocumentNlpProjection> {
        let mut doc = Document::new(source);
        self.analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().expect("single document")?;
        project_document(source, &doc, self.shingle_width)
    }

    /// Runs the full NLP DAG across a batch of documents on the shared worker pool.
    pub fn analyze_batch(&self, texts: &[&str]) -> Vec<Result<DocumentNlpProjection>> {
        let mut docs: Vec<Document> = texts.iter().map(|&t| Document::new(t)).collect();
        let results = self.analyzer.run(&mut docs);
        texts
            .iter()
            .zip(docs.iter())
            .zip(results)
            .map(|((&src, doc), res)| {
                res?;
                project_document(src, doc, self.shingle_width)
            })
            .collect()
    }
}
