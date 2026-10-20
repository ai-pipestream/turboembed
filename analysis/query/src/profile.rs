//! [`QueryAnalysisProfile`] and [`QueryProfileBuilder`]: assembles the
//! query-time [`Analyzer`] and its paired document-side lexical [`Analyzer`]
//! so query terms and document terms are guaranteed to match while extracting
//! structured [`crate::QueryFilter`]s and [`crate::FilterEvidence`].

use std::sync::{Arc, OnceLock};

use turbo_analysis::{Analyzer, Annotator, Bound, Document, Error, IxRange, Refs, Span, Spans, View};
use turbo_analysis_layers::{Date, DateTime, Entities, NumberNotation, Times, Tokens};
use turbo_chunker::terms::{IndexedTerms, TermForm, Vocabulary};
use turbo_geo::{BundledGazetteer, Gazetteer};
use turbo_glossary::{Case, Glossary, GlossaryAnnotator, GlossaryMatcher, TextMatcher};
use turbo_langspan::SpanDetector;
use turbo_namefind::{BuiltIn, DictionaryNameFinder, RegexNameFinder, TokenNameFinder};
use turbo_numeric::{MoneyExtractor, QuantityExtractor, TemporalExtractor};

use crate::annotators::{
    QueryGeocodeAnnotator, QueryGrammarAnnotator, QuerySpanDetector, QueryTimeAnnotator, SequentialBuilder,
    default_span_detector,
};
use crate::extract::{DICT_ENTITIES, ES_TIMES, REGEX_ENTITIES, extract_query_analysis};
use crate::model::{QueryAnalysis, QueryCursorNumericAnnotator};

fn shared_bundled_gazetteer() -> Arc<BundledGazetteer> {
    static GAZETTEER: OnceLock<Arc<BundledGazetteer>> = OnceLock::new();
    Arc::clone(GAZETTEER.get_or_init(|| Arc::new(BundledGazetteer::parse().expect("valid bundled gazetteer"))))
}

type GeoStepInstaller = Box<dyn FnOnce(SequentialBuilder) -> SequentialBuilder + Send>;

/// Builder for configuring a [`QueryAnalysisProfile`].
pub struct QueryProfileBuilder {
    default_language: String,
    term_form: TermForm,
    notation: NumberNotation,
    reference_datetime: Option<DateTime>,
    geo_installer: Option<GeoStepInstaller>,
    glossary: Option<(Arc<Glossary>, Case)>,
    dictionary_finder: Option<DictionaryNameFinder>,
    regex_finder: Option<(RegexNameFinder, bool)>,
    span_detector_factory: Option<Box<dyn Fn() -> SpanDetector + Send + Sync>>,
}

impl Default for QueryProfileBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryProfileBuilder {
    /// Creates a builder with default settings:
    /// - fallback language `"en"`,
    /// - [`TermForm::Stem`] lexical indexing,
    /// - [`NumberNotation::LatinUs`],
    /// - [`BundledGazetteer`] place resolution,
    /// - built-in regex entity patterns (`EMAIL`, `URL`, `USA_PHONE_NUM`, `MGRS`, `DEGREES_MIN_SEC_LAT_LON`),
    /// - built-in multi-language [`SpanDetector`].
    pub fn new() -> Self {
        let gaz = shared_bundled_gazetteer();
        let installer: GeoStepInstaller = Box::new(move |builder: SequentialBuilder| install_geo_steps(builder, gaz));
        Self {
            default_language: "en".to_owned(),
            term_form: TermForm::Stem,
            notation: NumberNotation::LatinUs,
            reference_datetime: None,
            geo_installer: Some(installer),
            glossary: None,
            dictionary_finder: None,
            regex_finder: Some((RegexNameFinder::defaults(&BuiltIn::ALL), true)),
            span_detector_factory: None,
        }
    }

    /// Sets the fallback language code used when a query contains no letters
    /// or an unsupported script (defaults to `"en"`).
    pub fn with_default_language(mut self, lang: impl Into<String>) -> Self {
        self.default_language = lang.into();
        self
    }

    /// Sets the [`TermForm`] column indexed into `turbo:vocabulary` and `bm25_terms`
    /// on both the query profile and its paired document lexical analyzer.
    pub fn with_term_form(mut self, form: TermForm) -> Self {
        self.term_form = form;
        self
    }

    /// Sets the [`NumberNotation`] used by numeric, money, and quantity extractors.
    pub fn with_number_notation(mut self, notation: NumberNotation) -> Self {
        self.notation = notation;
        self
    }

    /// Sets the reference [`Date`] (at midnight UTC) for resolving relative
    /// temporal expressions such as `"tomorrow"` or `"3 weeks ago"`.
    pub fn with_reference_date(mut self, date: Date) -> Self {
        self.reference_datetime = Some(DateTime::midnight(date));
        self
    }

    /// Sets the reference [`DateTime`] for resolving relative date and time expressions.
    pub fn with_reference_datetime(mut self, datetime: DateTime) -> Self {
        self.reference_datetime = Some(datetime);
        self
    }

    /// Configures a custom [`Gazetteer`] for place candidate generation and geocoding.
    pub fn with_gazetteer<G>(mut self, gazetteer: Arc<G>) -> Self
    where
        G: Gazetteer + Send + Sync + 'static,
    {
        self.geo_installer = Some(Box::new(move |builder: SequentialBuilder| install_geo_steps(builder, gazetteer)));
        self
    }

    /// Disables gazetteer place extraction and geocoding.
    pub fn without_gazetteer(mut self) -> Self {
        self.geo_installer = None;
        self
    }

    /// Attaches a domain [`Glossary`] matched with the given [`Case`] sensitivity.
    pub fn with_glossary(mut self, glossary: Arc<Glossary>, case: Case) -> Self {
        self.glossary = Some((glossary, case));
        self
    }

    /// Attaches a [`DictionaryNameFinder`] whose matched entities are emitted
    /// as [`crate::EntityFilter`]s.
    pub fn with_dictionary_finder(mut self, finder: DictionaryNameFinder) -> Self {
        self.dictionary_finder = Some(finder);
        self
    }

    /// Overrides the [`RegexNameFinder`] used for pattern entity extraction.
    pub fn with_regex_finder(mut self, finder: RegexNameFinder) -> Self {
        self.regex_finder = Some((finder, false));
        self
    }

    /// Disables regex pattern entity extraction.
    pub fn without_regex_finder(mut self) -> Self {
        self.regex_finder = None;
        self
    }

    /// Supplies a custom factory for constructing the [`SpanDetector`] used by
    /// both the query analyzer and the paired document analyzer.
    pub fn with_span_detector_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> SpanDetector + Send + Sync + 'static,
    {
        self.span_detector_factory = Some(Box::new(factory));
        self
    }

    /// Builds the configured [`QueryAnalysisProfile`].
    pub fn build(mut self) -> Result<QueryAnalysisProfile, Error> {
        let fast_ascii_en =
            self.span_detector_factory.is_none() && matches!(self.default_language.as_str(), "en" | "eng");
        let make_span_detector = || match &self.span_detector_factory {
            Some(f) => QuerySpanDetector::new(f(), false),
            None => QuerySpanDetector::new(default_span_detector(), fast_ascii_en),
        };

        let document_analyzer = install_lexical_backbone(
            SequentialBuilder::new(),
            &self.default_language,
            self.term_form,
            make_span_detector(),
        )
        .build()?;

        let mut qb = install_lexical_backbone(
            SequentialBuilder::new(),
            &self.default_language,
            self.term_form,
            make_span_detector(),
        );

        // Money, quantity, and calendar temporal extractors in one step.
        let temporal_extractor = match self.reference_datetime {
            Some(dt) => TemporalExtractor::with_reference(dt.date()),
            None => TemporalExtractor::new(),
        };
        qb = qb.step_skip_when(
            Bound::new(QueryCursorNumericAnnotator {
                money: MoneyExtractor::new(self.notation),
                quantity: QuantityExtractor::new(self.notation),
                temporal: temporal_extractor,
            }),
            Some(|doc| !doc.source().bytes().any(|b| b.is_ascii_digit())),
        );

        // Trigger-gated grammar and time chart parsers (English + Spanish).
        qb = qb
            .step_skip_when(
                QueryGrammarAnnotator::english().bound(),
                Some(|doc| doc.get::<Tokens>().is_some_and(|t| !crate::model::has_grammar_trigger(doc.source(), t))),
            )
            .add(QueryTimeAnnotator::english(self.reference_datetime))
            .step(Bound::new(QueryTimeAnnotator::spanish(self.reference_datetime)).write(0, ES_TIMES));

        // Gazetteer candidate detection + population-prior geocoding.
        if let Some(installer) = self.geo_installer.take() {
            qb = installer(qb);
        }

        // Optional glossary matcher.
        if let Some((glossary, case)) = self.glossary.take()
            && !glossary.is_empty()
        {
            let text_matcher = TextMatcher::new(&glossary, case)
                .map_err(|e| Error::Annotator { annotator: GlossaryAnnotator::ID, detail: e.to_string() })?;
            qb = qb.add(GlossaryAnnotator::new(glossary, GlossaryMatcher::Text(text_matcher)));
        }

        // Optional dictionary and regex entity finders on named `opennlp:entities` instances.
        if let Some(dict_finder) = self.dictionary_finder.take() {
            qb = qb.step(Bound::new(dict_finder).write(0, DICT_ENTITIES));
        }
        if let Some((regex_finder, is_default_builtin)) = self.regex_finder.take() {
            let config_hash = <RegexNameFinder as Annotator>::config_hash(&regex_finder);
            let skip = is_default_builtin
                .then_some((|doc: &Document| !has_builtin_regex_cue(doc.source())) as fn(&Document) -> bool);
            qb = qb.step_skip_when(
                Bound::new(QueryRegexEntityAnnotator { finder: regex_finder, is_default_builtin, config_hash })
                    .write(0, REGEX_ENTITIES),
                skip,
            );
        }

        let _ = Times::new;
        let query_analyzer = qb.build()?;
        Ok(QueryAnalysisProfile {
            query_analyzer,
            document_analyzer,
            term_form: self.term_form,
            reference_datetime: self.reference_datetime,
        })
    }
}

struct QueryRegexEntityAnnotator {
    finder: RegexNameFinder,
    is_default_builtin: bool,
    config_hash: u64,
}

fn has_builtin_regex_cue(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut consec_digits = 0usize;
    for &b in bytes {
        if matches!(b, b'@' | b'/' | b'-' | b'(' | b':') || (b == b'.' && bytes.len() >= 4) {
            return true;
        }
        consec_digits = if b.is_ascii_digit() { consec_digits + 1 } else { 0 };
        if consec_digits >= 4 {
            return true;
        }
    }
    false
}

impl Annotator for QueryRegexEntityAnnotator {
    const ID: &'static str = RegexNameFinder::ID;
    type Reads = (View, Tokens);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        let text = view.text();
        if self.is_default_builtin && !has_builtin_regex_cue(text) {
            return Ok((Entities {
                spans: Spans::new(view.id()),
                tokens: Vec::new(),
                kind: Vec::new(),
                kinds: Arc::clone(self.finder.kinds()),
                score: Vec::new(),
            },));
        }
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let mut ws_spans: Vec<(u32, u32)> = Vec::new();
        let mut ws_words: Vec<&str> = Vec::new();
        let mut in_word = None;
        for (idx, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(s) = in_word.take() {
                    ws_spans.push((s as u32, idx as u32));
                    ws_words.push(&text[s..idx]);
                }
            } else if in_word.is_none() {
                in_word = Some(idx);
            }
        }
        if let Some(s) = in_word {
            ws_spans.push((s as u32, text.len() as u32));
            ws_words.push(&text[s..]);
        }
        let mut names = Vec::new();
        self.finder
            .find(&ws_words, &mut names)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::with_capacity(names.len()),
            kind: Vec::new(),
            kinds: Arc::clone(self.finder.kinds()),
            score: Vec::with_capacity(names.len()),
        };
        for n in names {
            let (a, b) = (n.start as usize, n.end as usize);
            if a >= b || b > ws_spans.len() {
                continue;
            }
            let span = Span::new(ws_spans[a].0, ws_spans[b - 1].1);
            let t_start = tok_ends.partition_point(|&e| e <= span.start) as u32;
            let t_end = tok_starts.partition_point(|&s| s < span.end) as u32;
            entities.spans.push(span);
            entities.tokens.push(IxRange::new(t_start, t_end.max(t_start)));
            entities.kind.push(n.kind);
            entities.score.push(0.0);
        }
        Ok((entities,))
    }

    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}

fn install_lexical_backbone(
    builder: SequentialBuilder,
    default_lang: &str,
    term_form: TermForm,
    span_detector: QuerySpanDetector,
) -> SequentialBuilder {
    let sent_lang = match default_lang {
        "de" | "deu" | "ger" => "de",
        "es" | "spa" => "es",
        "fr" | "fra" | "fre" => "fr",
        "it" | "ita" => "it",
        "nl" | "nld" | "dut" => "nl",
        "pt" | "por" => "pt",
        "ru" | "rus" => "ru",
        _ => "en",
    };
    builder.add(crate::QueryTermAnnotator::new(sent_lang, default_lang, term_form, span_detector))
}

fn install_geo_steps<G>(builder: SequentialBuilder, gazetteer: Arc<G>) -> SequentialBuilder
where
    G: Gazetteer + Send + Sync + 'static,
{
    builder.add(QueryGeocodeAnnotator::new(gazetteer))
}

/// Query-tuned [`Analyzer`] profile that turns a raw query into structured
/// [`crate::QueryFilter`]s, [`crate::FilterEvidence`], and BM25 lexical terms
/// guaranteed to match the paired document lexical pipeline.
pub struct QueryAnalysisProfile {
    query_analyzer: Analyzer,
    document_analyzer: Analyzer,
    term_form: TermForm,
    reference_datetime: Option<DateTime>,
}

impl Default for QueryAnalysisProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryAnalysisProfile {
    /// Creates a [`QueryAnalysisProfile`] with default settings.
    pub fn new() -> Self {
        QueryProfileBuilder::new().build().expect("default QueryAnalysisProfile is valid")
    }

    /// Creates a [`QueryProfileBuilder`] for customizing the profile.
    pub fn builder() -> QueryProfileBuilder {
        QueryProfileBuilder::new()
    }

    /// Creates a [`QueryAnalysisProfile`] anchored to `reference_date` (midnight UTC)
    /// for resolving relative temporal expressions.
    pub fn with_reference_date(reference_date: Date) -> Self {
        QueryProfileBuilder::new()
            .with_reference_date(reference_date)
            .build()
            .expect("QueryAnalysisProfile with reference_date is valid")
    }

    /// Creates a [`QueryAnalysisProfile`] anchored to `reference_datetime`
    /// for resolving relative date and time expressions.
    pub fn with_reference_datetime(reference_datetime: DateTime) -> Self {
        QueryProfileBuilder::new()
            .with_reference_datetime(reference_datetime)
            .build()
            .expect("QueryAnalysisProfile with reference_datetime is valid")
    }

    /// Returns the [`TermForm`] used by both the query profile and its paired document analyzer.
    pub fn term_form(&self) -> TermForm {
        self.term_form
    }

    /// Returns the configured reference [`DateTime`], if any.
    pub fn reference_datetime(&self) -> Option<DateTime> {
        self.reference_datetime
    }

    /// Returns a reference to the underlying query [`Analyzer`].
    pub fn query_analyzer(&self) -> &Analyzer {
        &self.query_analyzer
    }

    /// Returns a reference to the paired document lexical [`Analyzer`].
    pub fn document_analyzer(&self) -> &Analyzer {
        &self.document_analyzer
    }

    /// Analyzes `query` and returns its structured [`QueryAnalysis`].
    pub fn analyze(&self, query: &str) -> Result<QueryAnalysis, Error> {
        let (_, analysis) = self.analyze_document(query)?;
        Ok(analysis)
    }

    /// Analyzes `query` and returns both the layered [`Document`] and the extracted [`QueryAnalysis`].
    pub fn analyze_document(&self, query: &str) -> Result<(Document, QueryAnalysis), Error> {
        let mut doc = Document::new(query);
        self.query_analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().expect("single document result")?;
        let analysis = extract_query_analysis(&doc)?;
        Ok((doc, analysis))
    }

    /// Runs the paired document-side lexical pipeline on `text` and returns
    /// the indexed terms in token order (`turbo:indexed-terms` resolved against
    /// `turbo:vocabulary`).
    ///
    /// For any text, `profile.analyze(text)?.bm25_terms` is guaranteed to equal
    /// `profile.document_terms(text)?`.
    pub fn document_terms(&self, text: &str) -> Result<Vec<String>, Error> {
        let mut doc = Document::new(text);
        self.document_analyzer
            .run(std::slice::from_mut(&mut doc))
            .into_iter()
            .next()
            .expect("single document result")?;
        let Some(vocab) = doc.get::<Vocabulary>() else { return Ok(Vec::new()) };
        let Some(indexed) = doc.get::<IndexedTerms>() else { return Ok(Vec::new()) };
        let terms = indexed.term.iter().filter_map(|slot| slot.map(|ix| vocab.get(ix).to_owned())).collect();
        Ok(terms)
    }
}
