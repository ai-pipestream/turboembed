//! Query analysis profile (`turbo-query`): a short-input [`turbo_analysis::Analyzer`]
//! profile that turns a search query into a typed [`QueryAnalysis`] holding:
//! - normalized/stemmed [`QueryTerm`]s and `bm25_terms` guaranteed to match
//!   documents indexed with [`QueryAnalysisProfile::document_analyzer`] (or
//!   [`QueryAnalysisProfile::document_terms`]),
//! - per-span [`LanguageEvidence`] and [`LanguageSpanEvidence`] (`turbo:language-spans`
//!   and `opennlp:language`) for single-language and mixed-language queries,
//! - structured [`QueryFilter`]s ([`TemporalFilter`], [`MoneyFilter`], [`QuantityFilter`],
//!   [`PlaceFilter`], [`GlossaryFilter`], [`EntityFilter`]), each tied to an
//!   exact UTF-8 source byte [`turbo_analysis::Span`] and [`FilterEvidence`]
//!   recording the layer and annotator provenance.

pub mod annotators;
mod extract;
mod lang_model;
pub mod model;
pub mod profile;

pub use annotators::{GazetteerCandidateAnnotator, QueryGrammarAnnotator, QueryTimeAnnotator, default_span_detector};
pub use model::{
    EntityFilter, FilterEvidence, FilterKind, GlossaryFilter, LanguageEvidence, LanguageSpanEvidence, MoneyFilter,
    PlaceFilter, QuantityFilter, QueryAnalysis, QueryFilter, QueryTerm, TemporalFilter,
};
pub use profile::{QueryAnalysisProfile, QueryProfileBuilder};
pub use turbo_analysis::Span;
pub use turbo_analysis_layers::{
    Currency, Date, DateTime, Granularity, Modifier, NumberNotation, Origin, TimeGrain, TimeKind,
};
pub use turbo_chunker::terms::TermForm;
pub use turbo_glossary::{Case, Glossary};
pub use turbo_namefind::{BuiltIn, DictionaryNameFinder, Pattern, RegexNameFinder};

pub(crate) struct QueryTermAnnotator {
    sent: turbo_sentdetect::RuleSentenceDetector,
    tok: turbo_tokenize::WordTokenizer,
    span_detector: crate::annotators::QuerySpanDetector,
    inner: turbo_stem::TermAnnotator,
    fallback_stemmer: Option<turbo_stem::snowball::SnowballStemmer>,
    term_form: TermForm,
    stopwords: turbo_stopword::StopwordAnnotator,
}

impl QueryTermAnnotator {
    pub(crate) fn new(
        sent_lang: &str,
        default_lang: &str,
        term_form: TermForm,
        span_detector: crate::annotators::QuerySpanDetector,
    ) -> Self {
        let fallback_stemmer = turbo_stem::NormalizationProfiles::for_language(default_lang)
            .map(|p| turbo_stem::snowball::SnowballStemmer::new(p.stemmer_algorithm()));
        let fallback_sw = turbo_stopword::StopwordLanguage::from_code(default_lang)
            .unwrap_or(turbo_stopword::StopwordLanguage::English);
        Self {
            sent: turbo_sentdetect::RuleSentenceDetector::new(sent_lang),
            tok: turbo_tokenize::WordTokenizer::new(),
            span_detector,
            inner: turbo_stem::TermAnnotator::by_language(Some(default_lang.to_owned())),
            fallback_stemmer,
            term_form,
            stopwords: turbo_stopword::StopwordAnnotator::new(turbo_stopword::ListChoice::Detected {
                fallback: Some(fallback_sw),
            }),
        }
    }
}

impl turbo_analysis::Annotator for QueryTermAnnotator {
    const ID: &'static str = turbo_langspan::SpanDetector::ID;
    type Reads = (turbo_analysis::View,);
    type Writes = (
        turbo_analysis_layers::Sentences,
        turbo_analysis_layers::Tokens,
        turbo_analysis_layers::LanguageSpans,
        turbo_analysis_layers::Language,
        turbo_analysis_layers::Terms,
        turbo_chunker::terms::Vocabulary,
        turbo_chunker::terms::IndexedTerms,
        turbo_stopword::Stopwords,
    );

    fn annotate(&self, (view,): turbo_analysis::Refs<'_, Self>) -> Result<Self::Writes, turbo_analysis::Error> {
        let text = view.text();
        let edge_ws =
            |c: Option<char>| c.is_some_and(|c| turbo_unicode::WhitespaceMode::Unicode.is_whitespace(c as u32));
        let sentences = if !text.is_empty()
            && !edge_ws(text.chars().next())
            && !edge_ws(text.chars().next_back())
            && !text.bytes().any(|b| matches!(b, b'.' | b'!' | b'?' | b'\n' | b'\r'))
        {
            let mut s = turbo_analysis_layers::Sentences { spans: turbo_analysis::Spans::new(view.id()) };
            s.spans.push(Span::new(0, text.len() as u32));
            s
        } else {
            self.sent.annotate((view,))?.0
        };
        let (tokens, _) = self.tok.annotate((view, Some(&sentences)))?;
        let (lang_spans, lang) = self.span_detector.annotate((view, Some(&sentences), Some(&tokens)))?;
        let language = (!lang.code.is_empty()).then_some(&lang);
        let terms = if text.is_ascii() && text.len() <= 512 && tokens.spans.view() == view.id() {
            let stemmer = match language.and_then(turbo_analysis_layers::Language::best) {
                Some(("eng", _)) => {
                    Some(turbo_stem::snowball::SnowballStemmer::new(turbo_stem::snowball::SnowballAlgorithm::English))
                }
                Some((code, _)) => turbo_stem::NormalizationProfiles::for_language(code)
                    .map(|p| turbo_stem::snowball::SnowballStemmer::new(p.stemmer_algorithm())),
                None => self.fallback_stemmer,
            };
            let (starts, ends) = tokens.spans.columns();
            let mut surface = turbo_analysis_layers::Strs::default();
            let mut folded = turbo_analysis_layers::Strs::default();
            let mut stem_col = stemmer.is_some().then(turbo_analysis_layers::Strs::default);
            let mut units = Vec::with_capacity(32);
            let mut out_u16 = Vec::with_capacity(32);
            for (&s, &e) in starts.iter().zip(ends) {
                let tok = &text[s as usize..e as usize];
                surface.push(tok);
                crate::lang_model::with_lowercase(tok, |lower| {
                    folded.push(lower);
                    if let (Some(stems), Some(st)) = (&mut stem_col, stemmer) {
                        if !lower.bytes().any(|b| b.is_ascii_alphabetic()) {
                            stems.push(lower);
                        } else {
                            units.clear();
                            units.extend(lower.bytes().map(u16::from));
                            turbo_stem::stemmer::Stemmer::stem_utf16(&st, &units, &mut out_u16);
                            let mut sbuf = [0u8; 64];
                            if out_u16.len() <= 64 && out_u16.iter().all(|&u| u < 128) {
                                for (d, &u) in sbuf[..out_u16.len()].iter_mut().zip(&out_u16) {
                                    *d = u as u8;
                                }
                                stems.push(std::str::from_utf8(&sbuf[..out_u16.len()]).expect("ASCII"));
                            } else {
                                stems.push(&String::from_utf16_lossy(&out_u16));
                            }
                        }
                    }
                });
            }
            turbo_analysis_layers::Terms { surface, folded, stem: stem_col, lemma: None }
        } else {
            self.inner.annotate((view, &tokens, language, None, None))?.0
        };
        let (vocab, indexed) =
            turbo_chunker::terms::NormalizedTermIndexer { form: self.term_form }.annotate((&terms, &tokens))?;
        let (stopwords,) = self.stopwords.annotate((view, &tokens, language))?;
        Ok((sentences, tokens, lang_spans, lang, terms, vocab, indexed, stopwords))
    }

    fn config_hash(&self) -> u64 {
        [
            self.sent.config_hash(),
            self.tok.config_hash(),
            self.span_detector.config_hash(),
            self.inner.config_hash(),
            turbo_chunker::terms::NormalizedTermIndexer { form: self.term_form }.config_hash(),
            self.stopwords.config_hash(),
        ]
        .into_iter()
        .fold(turbo_analysis::Fnv1a::new(), |mut h, v| {
            h.u64(v);
            h
        })
        .finish()
    }
}

struct OrderedStep<S> {
    inner: S,
    after: Option<turbo_analysis::KeyId>,
    skip_when: Option<fn(&turbo_analysis::Document) -> bool>,
}

impl<S: turbo_analysis::annotate::Step> turbo_analysis::annotate::Step for OrderedStep<S> {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn reads(&self) -> Vec<turbo_analysis::KeyId> {
        let mut r = self.inner.reads();
        if let Some(dep) = self.after
            && !r.contains(&dep)
        {
            r.push(dep);
        }
        r
    }
    fn required(&self) -> Vec<bool> {
        let mut req = self.inner.required();
        if let Some(dep) = self.after
            && !self.inner.reads().contains(&dep)
        {
            req.push(false);
        }
        req
    }
    fn writes(&self) -> Vec<turbo_analysis::KeyId> {
        self.inner.writes()
    }
    fn anchors(&self) -> Vec<turbo_analysis::annotate::AnchorUse> {
        self.inner.anchors()
    }
    fn config_hash(&self) -> u64 {
        self.inner.config_hash()
    }
    fn compute(&self, docs: &[turbo_analysis::Document]) -> Box<dyn std::any::Any + Send> {
        if let [doc] = docs
            && self.skip_when.is_some_and(|pred| pred(doc))
        {
            return Box::new(());
        }
        self.inner.compute(docs)
    }
    fn store(
        &self,
        out: Box<dyn std::any::Any + Send>,
        docs: &mut [turbo_analysis::Document],
        status: &mut [Result<(), turbo_analysis::Error>],
    ) {
        if out.is::<()>() {
            return;
        }
        self.inner.store(out, docs, status);
    }
}

pub(crate) struct SequentialBuilder {
    inner: turbo_analysis::analyzer::Builder,
    last_write: Option<turbo_analysis::KeyId>,
}

impl SequentialBuilder {
    pub(crate) fn new() -> Self {
        Self { inner: turbo_analysis::Analyzer::builder(), last_write: None }
    }
    pub(crate) fn add<A: turbo_analysis::Annotator>(self, annotator: A) -> Self {
        self.step(turbo_analysis::Bound::new(annotator))
    }
    pub(crate) fn step(self, step: impl turbo_analysis::annotate::Step + 'static) -> Self {
        self.step_skip_when(step, None)
    }
    pub(crate) fn step_skip_when(
        mut self,
        step: impl turbo_analysis::annotate::Step + 'static,
        skip_when: Option<fn(&turbo_analysis::Document) -> bool>,
    ) -> Self {
        let next_write = step.writes().into_iter().next().or(self.last_write);
        self.inner = self.inner.step(OrderedStep { inner: step, after: self.last_write, skip_when });
        self.last_write = next_write;
        self
    }
    pub(crate) fn build(self) -> Result<turbo_analysis::Analyzer, turbo_analysis::Error> {
        self.inner.build()
    }
}

pub(crate) struct QueryCursorNumericAnnotator {
    pub(crate) money: turbo_numeric::MoneyExtractor,
    pub(crate) quantity: turbo_numeric::QuantityExtractor,
    pub(crate) temporal: turbo_numeric::TemporalExtractor,
}

impl turbo_analysis::Annotator for QueryCursorNumericAnnotator {
    const ID: &'static str = "turbo-numeric:cursor";
    type Reads = (turbo_analysis::View, turbo_analysis_layers::Tokens);
    type Writes = (turbo_analysis_layers::Money, turbo_analysis_layers::Quantities, turbo_analysis_layers::Temporals);

    fn annotate(&self, (view, tokens): turbo_analysis::Refs<'_, Self>) -> Result<Self::Writes, turbo_analysis::Error> {
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        let mut d_wins: Vec<(usize, usize, usize, bool)> = Vec::new();
        let (mut prev_d, mut ft, mut lt) = (usize::MAX, None, 0usize);
        for (k, (&s, &e)) in starts.iter().zip(ends).enumerate() {
            let tok = &text[s as usize..e as usize];
            if tok.bytes().any(|b| b.is_ascii_digit()) {
                let lo = if k >= 2 && (ends[k - 1] - starts[k - 1]) <= 2 { k - 2 } else { k.saturating_sub(1) };
                let hi = (k + 5).min(n);
                let is_clock_meridian = tok.len() >= 3
                    && matches!(&tok.as_bytes()[tok.len() - 2..], b"am" | b"pm" | b"AM" | b"PM" | b"Am" | b"Pm")
                    && tok.as_bytes()[..tok.len() - 2].iter().all(u8::is_ascii_digit);
                let next_tok = (k + 1 < n).then(|| &text[starts[k + 1] as usize..ends[k + 1] as usize]);
                let qty_cue = (!is_clock_meridian
                    && tok.bytes().any(|b| !b.is_ascii_digit() && !matches!(b, b'.' | b',' | b'-')))
                    || next_tok.is_some_and(|nt| {
                        nt.len() <= 20
                            && !matches!(
                                nt,
                                "at" | "on" | "in" | "to" | "of" | "by" | "for" | "from" | "with" | "and" | "or"
                            )
                    });
                if let Some(last) = d_wins.last_mut()
                    && prev_d != usize::MAX
                    && (k <= prev_d + 1 || k < last.1)
                {
                    last.1 = last.1.max(hi);
                    last.2 += 1;
                    last.3 |= qty_cue;
                } else {
                    let clamped_lo = d_wins.last().map_or(lo, |last| lo.max(last.1));
                    d_wins.push((clamped_lo.min(k), hi, 1, qty_cue));
                }
                prev_d = k;
            }
            if crate::lang_model::is_temporal_cursor_token(tok) {
                ft.get_or_insert(k);
                lt = k;
            }
        }
        // A lowercase ISO code counts only right after an amount
        // (`500 usd`, never `top 10`), and only from a small
        // allowlist of codes unambiguous in lowercase queries: a
        // denylist of English-word codes (php, cop, ron, top, ...)
        // can never be completed, so anything else stays a word.
        let token_at = |i: usize| &text[starts[i] as usize..ends[i] as usize];
        let lowercase_iso_after_amount = |i: usize| {
            const LOWERCASE_CODES: [&str; 18] = [
                "aud", "brl", "chf", "cny", "dkk", "eur", "gbp", "hkd", "inr", "jpy", "krw", "mxn", "nok", "nzd",
                "sek", "sgd", "usd", "zar",
            ];
            let w = token_at(i);
            if i == 0 || !LOWERCASE_CODES.contains(&w) {
                return false;
            }
            let prev = token_at(i - 1);
            let amount_before = prev.bytes().next().is_some_and(|b| b.is_ascii_digit())
                || (prev.len() > 1 && prev.starts_with('$') && prev.as_bytes()[1].is_ascii_digit());
            amount_before
                && turbo_analysis_layers::Currency::new([
                    w.as_bytes()[0].to_ascii_uppercase(),
                    w.as_bytes()[1].to_ascii_uppercase(),
                    w.as_bytes()[2].to_ascii_uppercase(),
                ])
                .is_some_and(turbo_numeric::money::is_iso_currency)
        };
        let mut money = turbo_analysis_layers::Money::new(view.id());
        let mut quantities = turbo_analysis_layers::Quantities::new(view.id());
        for (tok_lo, tok_hi, d_cnt, qty_cue) in d_wins {
            let (lo, hi) = (starts[tok_lo] as usize, ends[tok_hi - 1] as usize);
            let (off, sub_text) = (lo as u32, &text[lo..hi]);
            #[rustfmt::skip]
            let money_cue = !sub_text.is_ascii()
                || sub_text.bytes().any(|b| b == b'$' || b.is_ascii_uppercase())
                || (tok_lo..tok_hi).any(|i| {
                    let w = &text[starts[i] as usize..ends[i] as usize];
                    matches!(
                        w,
                        "dollar" | "dollars" | "euro" | "euros" | "pound" | "pounds" | "sterling" | "yen"
                            | "rupee" | "rupees" | "yuan" | "renminbi" | "ruble" | "rubles" | "rouble"
                            | "roubles" | "shekel" | "shekels" | "baht"
                    ) || lowercase_iso_after_amount(i)
                });
            // The money extractor reads ISO codes in capitals only
            // (its document behavior is pinned by JVM goldens), but
            // queries write `500 usd`: scan a copy with such tokens
            // uppercased in place, which keeps every byte offset
            // unchanged.
            let mut owned: Option<String> = None;
            for (off_i, &start) in starts[tok_lo..tok_hi].iter().enumerate() {
                if lowercase_iso_after_amount(tok_lo + off_i) {
                    let buf = owned.get_or_insert_with(|| sub_text.to_owned());
                    let at = start as usize - lo;
                    buf[at..at + 3].make_ascii_uppercase();
                }
            }
            let sub_text: &str = owned.as_deref().unwrap_or(sub_text);
            let mut m_cnt = 0usize;
            if money_cue {
                let mut sm = turbo_analysis_layers::Money::new(view.id());
                self.money.scan(sub_text, &mut sm);
                m_cnt = sm.spans.len();
                for (ix, sp) in sm.spans.iter() {
                    let m = sm.get(ix);
                    money.push(turbo_analysis_layers::MoneyMention {
                        span: Span::new(sp.start + off, sp.end + off),
                        ..m
                    });
                }
            }
            if qty_cue && m_cnt < d_cnt {
                let mut sq = turbo_analysis_layers::Quantities::new(view.id());
                self.quantity.scan(sub_text, &mut sq);
                for (ix, sp) in sq.spans.iter() {
                    let q = sq.get(ix);
                    quantities.push(turbo_analysis_layers::QuantityMention {
                        span: Span::new(sp.start + off, sp.end + off),
                        ..q
                    });
                }
            }
        }
        let mut temporals = turbo_analysis_layers::Temporals::new(view.id());
        if let Some(first_t) = ft {
            let lo = starts[first_t.saturating_sub(4)] as usize;
            let hi = ends[(lt + 4).min(n - 1)] as usize;
            let off = lo as u32;
            let mut sub = turbo_analysis_layers::Temporals::new(view.id());
            self.temporal.extract(&text[lo..hi], &mut sub);
            for (ix, sp) in sub.spans.iter() {
                let mut m = sub.get(ix);
                m.span = Span::new(sp.start + off, sp.end + off);
                temporals.push(m);
            }
        }
        Ok((money, quantities, temporals))
    }

    fn config_hash(&self) -> u64 {
        [self.money.config_hash(), self.quantity.config_hash(), self.temporal.config_hash()]
            .into_iter()
            .fold(turbo_analysis::Fnv1a::new(), |mut h, v| {
                h.u64(v);
                h
            })
            .finish()
    }
}

/// The quantity extractor for queries: the default short units
/// plus their spelled-out English forms, since queries write
/// `10 kilometers` and `2 kilograms`, not only abbreviations.
pub(crate) fn query_quantity_extractor(
    notation: turbo_analysis_layers::NumberNotation,
) -> turbo_numeric::QuantityExtractor {
    const LONG_UNITS: [&str; 57] = [
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
        "meter",
        "meters",
        "metre",
        "metres",
        "centimeter",
        "centimeters",
        "millimeter",
        "millimeters",
        "mile",
        "miles",
        "yard",
        "yards",
        "inch",
        "inches",
        "feet",
        "kilogram",
        "kilograms",
        "gram",
        "grams",
        "milligram",
        "milligrams",
        "ton",
        "tons",
        "liter",
        "liters",
        "milliliter",
        "milliliters",
        "millisecond",
        "milliseconds",
        "microsecond",
        "microseconds",
        "minute",
        "minutes",
        "hour",
        "hours",
        "kilobyte",
        "kilobytes",
        "megabyte",
        "megabytes",
        "gigabyte",
        "gigabytes",
        "terabyte",
        "terabytes",
        "petabyte",
        "petabytes",
        "kilowatt",
        "kilowatts",
        "megawatt",
        "megawatts",
        "calorie",
        "calories",
        "kilocalorie",
        "kilocalories",
    ];
    turbo_numeric::QuantityExtractor::with_units(
        turbo_numeric::DEFAULT_UNITS.iter().chain(LONG_UNITS.iter()).copied(),
        notation,
    )
    .unwrap_or_else(|_| turbo_numeric::QuantityExtractor::new(notation))
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
