//! Unified one-pass search query planner (`AnalyzeQuery`, `#2339`).
//!
//! Produces ordinal-exact per-field BM25 terms (counting empty-normalized
//! tokens), multi-token `turbo-matcher` query synonym matches, enriched
//! structured NLP filters (`Temporal`, ECB `Money`, SI `Quantity`, `Place`
//! bbox/spine, `Glossary`, KB-linked `Entity`), and residual non-filter terms
//! in one pass over `ViewId::SOURCE`.

use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Error, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Currency, Date, DateTime, Language, NumberNotation, Sentences, Strs, Tokens};
use turbo_chunker::terms::{NormalizedTermIndexer, TermForm};
use turbo_entity_linker::EntityLinker;
use turbo_geo::Gazetteer;
use turbo_glossary::{Case, Glossary};
use turbo_matcher::{Input as MatcherInput, PhraseSynonymExpander};
use turbo_namefind::{DictionaryNameFinder, RegexNameFinder};
use turbo_normalize::{Invisible, NormalizeStep};
use turbo_numeric::money::EcbRates;
use turbo_stem::TermAnnotator;
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage};
use turbo_tokenize::{TokenizerOptions, WhitespaceTokenizer, WordTokenizer};

use crate::one_pass_filters::{FilterEnricherConfig, build_structured_filters};
use crate::one_pass_types::{
    FieldQueryPlan, FieldQuerySpec, FieldTokenizerKind, OnePassQueryPlan, OrdinalQueryTerm, QuerySynonymMatch,
    StructuredFilterPlan,
};
use crate::profile::{QueryAnalysisProfile, QueryProfileBuilder};

struct CompiledField {
    spec: FieldQuerySpec,
    term_annotator: TermAnnotator,
    stopwords: StopwordAnnotator,
    synonyms: Option<Arc<PhraseSynonymExpander>>,
}

/// Builder for [`OnePassQueryAnalyzer`] (`#2339`).
pub struct OnePassQueryBuilder {
    profile_builder: QueryProfileBuilder,
    default_language: String,
    default_term_form: TermForm,
    reference_date: Option<Date>,
    fields: Vec<FieldQuerySpec>,
    default_synonyms: Option<Arc<PhraseSynonymExpander>>,
    ecb: Option<(Arc<EcbRates>, Currency, Option<Date>)>,
    entity_linker: Option<Arc<EntityLinker>>,
}

impl Default for OnePassQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OnePassQueryBuilder {
    /// Creates a builder with default `"en"` stemmed settings.
    pub fn new() -> Self {
        Self {
            profile_builder: QueryProfileBuilder::new(),
            default_language: "en".to_owned(),
            default_term_form: TermForm::Stem,
            reference_date: None,
            fields: Vec::new(),
            default_synonyms: None,
            ecb: None,
            entity_linker: None,
        }
    }

    /// Sets the fallback language code (defaults to `"en"`).
    pub fn with_default_language(mut self, lang: impl Into<String>) -> Self {
        let lang = lang.into();
        self.default_language = lang.clone();
        self.profile_builder = self.profile_builder.with_default_language(lang);
        self
    }

    /// Sets the default [`TermForm`] used when no explicit fields are registered.
    pub fn with_term_form(mut self, form: TermForm) -> Self {
        self.default_term_form = form;
        self.profile_builder = self.profile_builder.with_term_form(form);
        self
    }

    /// Sets the [`NumberNotation`] for money and quantity extraction.
    pub fn with_number_notation(mut self, notation: NumberNotation) -> Self {
        self.profile_builder = self.profile_builder.with_number_notation(notation);
        self
    }

    /// Anchors relative temporal expressions (and default ECB FX lookups) to `date`.
    pub fn with_reference_date(mut self, date: Date) -> Self {
        self.reference_date = Some(date);
        self.profile_builder = self.profile_builder.with_reference_date(date);
        self
    }

    /// Anchors relative date/time expressions to `datetime`.
    pub fn with_reference_datetime(mut self, datetime: DateTime) -> Self {
        self.reference_date = Some(datetime.date());
        self.profile_builder = self.profile_builder.with_reference_datetime(datetime);
        self
    }

    /// Overrides the [`Gazetteer`] used for place resolution.
    pub fn with_gazetteer<G: Gazetteer + Send + Sync + 'static>(mut self, gazetteer: Arc<G>) -> Self {
        self.profile_builder = self.profile_builder.with_gazetteer(gazetteer);
        self
    }

    /// Disables gazetteer place resolution.
    pub fn without_gazetteer(mut self) -> Self {
        self.profile_builder = self.profile_builder.without_gazetteer();
        self
    }

    /// Attaches a domain [`Glossary`].
    pub fn with_glossary(mut self, glossary: Arc<Glossary>, case: Case) -> Self {
        self.profile_builder = self.profile_builder.with_glossary(glossary, case);
        self
    }

    /// Attaches a [`DictionaryNameFinder`] for named entity extraction.
    pub fn with_dictionary_finder(mut self, finder: DictionaryNameFinder) -> Self {
        self.profile_builder = self.profile_builder.with_dictionary_finder(finder);
        self
    }

    /// Overrides the [`RegexNameFinder`] used for pattern entity extraction.
    pub fn with_regex_finder(mut self, finder: RegexNameFinder) -> Self {
        self.profile_builder = self.profile_builder.with_regex_finder(finder);
        self
    }

    /// Disables regex pattern entity extraction.
    pub fn without_regex_finder(mut self) -> Self {
        self.profile_builder = self.profile_builder.without_regex_finder();
        self
    }

    /// Appends a [`FieldQuerySpec`] profile to evaluate in the single pass.
    pub fn add_field(mut self, spec: FieldQuerySpec) -> Self {
        self.fields.push(spec);
        self
    }

    /// Replaces the field list with `specs`.
    pub fn with_fields(mut self, specs: impl IntoIterator<Item = FieldQuerySpec>) -> Self {
        self.fields = specs.into_iter().collect();
        self
    }

    /// Sets a default [`PhraseSynonymExpander`] shared by fields without a field-specific expander.
    pub fn with_synonyms(mut self, synonyms: Arc<PhraseSynonymExpander>) -> Self {
        self.default_synonyms = Some(synonyms);
        self
    }

    /// Enables ECB reference-rate currency normalization to `target_currency`.
    pub fn with_ecb_rates(mut self, rates: Arc<EcbRates>, target_currency: Currency, as_of: Option<Date>) -> Self {
        self.ecb = Some((rates, target_currency, as_of));
        self
    }

    /// Attaches an [`EntityLinker`] to resolve extracted entity mentions to KB IDs.
    pub fn with_entity_linker(mut self, linker: Arc<EntityLinker>) -> Self {
        self.entity_linker = Some(linker);
        self
    }

    /// Builds the configured [`OnePassQueryAnalyzer`].
    pub fn build(mut self) -> Result<OnePassQueryAnalyzer, Error> {
        let profile = self.profile_builder.build()?;
        if self.fields.is_empty() {
            self.fields.push(FieldQuerySpec::new("default").with_term_form(self.default_term_form));
        }
        let mut compiled = Vec::with_capacity(self.fields.len());
        for spec in self.fields {
            let lang = spec.language.as_deref().unwrap_or(&self.default_language);
            let sw_lang = StopwordLanguage::from_code(lang).unwrap_or(StopwordLanguage::English);
            let synonyms = spec.synonyms.clone().or_else(|| self.default_synonyms.clone());
            compiled.push(CompiledField {
                term_annotator: TermAnnotator::by_language(Some(lang.to_owned())),
                stopwords: StopwordAnnotator::new(ListChoice::Detected { fallback: Some(sw_lang) }),
                synonyms,
                spec,
            });
        }
        let fallback_date = self.reference_date.unwrap_or_else(|| Date::new(2026, 1, 1).expect("valid anchor date"));
        let ecb = self.ecb.map(|(r, c, d)| (r, c, d.unwrap_or(fallback_date)));
        Ok(OnePassQueryAnalyzer {
            profile,
            fields: compiled,
            word_tok: WordTokenizer::new(),
            ws_tok: WhitespaceTokenizer::new(TokenizerOptions::default()),
            invisible_step: NormalizeStep::new(Invisible),
            ecb,
            entity_linker: self.entity_linker,
        })
    }
}

/// One-pass query analyzer producing per-field ordinal BM25 terms, multi-word
/// synonym matches, enriched structured NLP filters, and residual query terms.
pub struct OnePassQueryAnalyzer {
    profile: QueryAnalysisProfile,
    fields: Vec<CompiledField>,
    word_tok: WordTokenizer,
    ws_tok: WhitespaceTokenizer,
    invisible_step: NormalizeStep<Invisible>,
    ecb: Option<(Arc<EcbRates>, Currency, Date)>,
    entity_linker: Option<Arc<EntityLinker>>,
}

impl Default for OnePassQueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl OnePassQueryAnalyzer {
    /// Creates a [`OnePassQueryAnalyzer`] with default settings.
    pub fn new() -> Self {
        OnePassQueryBuilder::new().build().expect("default OnePassQueryAnalyzer is valid")
    }

    /// Creates a [`OnePassQueryBuilder`] for customizing fields, synonyms, and extractors.
    pub fn builder() -> OnePassQueryBuilder {
        OnePassQueryBuilder::new()
    }

    /// Returns the underlying [`QueryAnalysisProfile`].
    pub fn profile(&self) -> &QueryAnalysisProfile {
        &self.profile
    }

    /// Analyzes `query` in one pass and returns the [`OnePassQueryPlan`].
    pub fn analyze(&self, query: &str) -> Result<OnePassQueryPlan, Error> {
        let (_, plan) = self.analyze_document(query)?;
        Ok(plan)
    }

    /// Analyzes `query` in one pass and returns both the [`Document`] (carrying
    /// `ViewId::SOURCE`, any `Invisible`-normalized child view, and all layers)
    /// and the [`OnePassQueryPlan`].
    pub fn analyze_document(&self, query: &str) -> Result<(Document, OnePassQueryPlan), Error> {
        let (mut doc, base) = self.profile.analyze_document(query)?;
        let need_inv = self.fields.iter().any(|f| f.spec.strip_invisible)
            && query.chars().any(turbo_normalize::tables::is_invisible);
        if need_inv {
            let src_view = doc.require(&Key::<View>::DEFAULT)?;
            let (clean_view,) = self.invisible_step.annotate((src_view,))?;
            doc.add(
                &Key::<View>::named("invisible_clean"),
                clean_view,
                Provenance {
                    annotator: NormalizeStep::<Invisible>::ID,
                    version: NormalizeStep::<Invisible>::VERSION,
                    config: self.invisible_step.config_hash(),
                    model: None,
                    execution: None,
                    reads: vec![Key::<View>::DEFAULT.id()],
                    on: Vec::new(),
                },
            )?;
        }

        let cfg = FilterEnricherConfig {
            ecb: self.ecb.as_ref().map(|(r, c, d)| (r.as_ref(), *c, *d)),
            entity_linker: self.entity_linker.as_ref(),
        };
        let filters = build_structured_filters(&doc, &base, &cfg)?;

        let src_view = doc.require(&Key::<View>::DEFAULT)?;
        let sents = doc.get::<Sentences>();
        let clean_view = doc.get_key(&Key::<View>::named("invisible_clean"));
        let lang = doc.get::<Language>().and_then(|l| l.best().filter(|(c, _)| *c != "und").map(|_| l));

        let need_word = self.fields.iter().any(|f| f.spec.tokenizer == FieldTokenizerKind::Word);
        let need_ws = self.fields.iter().any(|f| f.spec.tokenizer == FieldTokenizerKind::Whitespace);
        let word_tokens = if need_word {
            match doc.get::<Tokens>() {
                Some(t) if t.spans.view() == ViewId::SOURCE => Some(Tokens { spans: t.spans.clone(), sentence: None }),
                _ => Some(self.word_tok.annotate((src_view, sents))?.0),
            }
        } else {
            None
        };
        let ws_tokens = if need_ws { Some(self.ws_tok.annotate((src_view, sents))?.0) } else { None };

        let mut field_plans = Vec::with_capacity(self.fields.len());
        for f in &self.fields {
            let raw_toks = match f.spec.tokenizer {
                FieldTokenizerKind::Word => word_tokens.as_ref().expect("word tokens computed"),
                FieldTokenizerKind::Whitespace => ws_tokens.as_ref().expect("whitespace tokens computed"),
            };
            let target_clean = if f.spec.strip_invisible { clean_view } else { None };
            field_plans.push(self.evaluate_field(f, src_view, target_clean, raw_toks, lang, &filters)?);
        }

        let residual_terms = field_plans.first().map(|p| p.residual_terms.clone()).unwrap_or_default();
        let free_text = residual_terms.iter().map(|t| t.surface.as_str()).collect::<Vec<_>>().join(" ");

        let plan = OnePassQueryPlan {
            query: query.to_owned(),
            language: base.language,
            fields: field_plans,
            filters,
            residual_terms,
            free_text,
        };
        Ok((doc, plan))
    }

    fn evaluate_field(
        &self,
        cf: &CompiledField,
        src_view: &View,
        clean_view: Option<&View>,
        raw_tokens: &Tokens,
        lang: Option<&Language>,
        filters: &[StructuredFilterPlan],
    ) -> Result<FieldQueryPlan, Error> {
        let src = src_view.text();
        let active_view = clean_view.unwrap_or(src_view);
        let total_token_ordinals = raw_tokens.spans.len() as u32;

        let mut clean_spans = Spans::with_capacity(active_view.id(), raw_tokens.spans.len());
        let mut clean_to_orig: Vec<(u32, Span)> = Vec::with_capacity(raw_tokens.spans.len());
        let mut empty_normalized_ordinals: Vec<u32> = Vec::new();

        for (ix, raw_span) in raw_tokens.spans.iter() {
            let ord = ix.get() as u32;
            let mapped = match clean_view {
                Some(cv) => cv.alignment().span_to_child(raw_span),
                None => raw_span,
            };
            if mapped.is_empty() {
                empty_normalized_ordinals.push(ord);
            } else {
                clean_spans.push(mapped);
                clean_to_orig.push((ord, raw_span));
            }
        }

        let clean_tokens = Tokens { spans: clean_spans, sentence: None };
        let field_lang = if cf.spec.language.is_some() { None } else { lang };
        let (ling_terms,) = cf.term_annotator.annotate((active_view, &clean_tokens, field_lang, None, None))?;
        let (vocab, indexed) =
            NormalizedTermIndexer { form: cf.spec.term_form }.annotate((&ling_terms, &clean_tokens))?;
        let (stopwords,) = cf.stopwords.annotate((active_view, &clean_tokens, field_lang))?;

        let mut bm25_terms = Vec::with_capacity(clean_to_orig.len());
        let mut residual_terms = Vec::with_capacity(clean_to_orig.len());

        for (j, &(ord, src_span)) in clean_to_orig.iter().enumerate() {
            let Some(v_ix) = indexed.term.get(j).copied().flatten() else {
                empty_normalized_ordinals.push(ord);
                continue;
            };
            let term = vocab.get(v_ix).to_owned();
            let is_stopword = stopwords.stop.get(j).copied().unwrap_or(false);
            let covered_by_filter = filters.iter().any(|flt| flt.span.overlaps(src_span));
            let oterm = OrdinalQueryTerm {
                term,
                ordinal: ord,
                pos_len: 1,
                span: src_span,
                surface: src[src_span.range()].to_owned(),
                is_stopword,
                covered_by_filter,
            };
            if !covered_by_filter && (!cf.spec.filter_stopwords || !is_stopword) {
                residual_terms.push(oterm.clone());
            }
            bm25_terms.push(oterm);
        }
        empty_normalized_ordinals.sort_unstable();

        let synonyms =
            self.match_synonyms(cf, active_view, src, &clean_tokens, &ling_terms.stem, &clean_to_orig, field_lang)?;

        Ok(FieldQueryPlan {
            field: cf.spec.field.clone(),
            total_token_ordinals,
            empty_normalized_ordinals,
            bm25_terms,
            residual_terms,
            synonyms,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn match_synonyms(
        &self,
        cf: &CompiledField,
        active_view: &View,
        src: &str,
        clean_tokens: &Tokens,
        stem_col: &Option<Strs>,
        clean_to_orig: &[(u32, Span)],
        field_lang: Option<&Language>,
    ) -> Result<Vec<QuerySynonymMatch>, Error> {
        let Some(expander) = &cf.synonyms else { return Ok(Vec::new()) };
        if clean_tokens.spans.is_empty() {
            return Ok(Vec::new());
        }
        let mut input = MatcherInput::new(active_view.text(), clean_tokens)
            .map_err(|e| Error::Annotator { annotator: "turbo-query:synonyms", detail: e.to_string() })?;
        if let Some(stems) = stem_col {
            input = input
                .with_stems(stems)
                .map_err(|e| Error::Annotator { annotator: "turbo-query:synonyms", detail: e.to_string() })?;
        }
        let raw_matches = expander
            .find_expansions(&input, clean_tokens)
            .map_err(|e| Error::Annotator { annotator: "turbo-query:synonyms", detail: e.to_string() })?;
        let mut out = Vec::with_capacity(raw_matches.len());
        for (c_start, c_end, _, targets) in raw_matches {
            if c_start >= c_end || c_end > clean_to_orig.len() {
                continue;
            }
            let start_ordinal = clean_to_orig[c_start].0;
            let end_ordinal = clean_to_orig[c_end - 1].0 + 1;
            let span = Span::new(clean_to_orig[c_start].1.start, clean_to_orig[c_end - 1].1.end);
            let surface = src[span.range()].to_owned();
            for target_seq in targets {
                let terms = self.analyze_synonym_target(cf, target_seq, field_lang)?;
                if !terms.is_empty() {
                    out.push(QuerySynonymMatch {
                        start_ordinal,
                        end_ordinal,
                        pos_len: end_ordinal - start_ordinal,
                        span,
                        surface: surface.clone(),
                        terms,
                    });
                }
            }
        }
        Ok(out)
    }

    fn analyze_synonym_target(
        &self,
        cf: &CompiledField,
        target_seq: &[String],
        field_lang: Option<&Language>,
    ) -> Result<Vec<String>, Error> {
        let joined = target_seq.join(" ");
        let syn_doc = Document::new(joined.as_str());
        let syn_view = syn_doc.require(&Key::<View>::DEFAULT)?;
        let (syn_tokens, _) = self.word_tok.annotate((syn_view, None))?;
        let (ling,) = cf.term_annotator.annotate((syn_view, &syn_tokens, field_lang, None, None))?;
        let (vocab, indexed) = NormalizedTermIndexer { form: cf.spec.term_form }.annotate((&ling, &syn_tokens))?;
        Ok(indexed.term.iter().filter_map(|slot| slot.map(|ix| vocab.get(ix).to_owned())).collect())
    }
}
