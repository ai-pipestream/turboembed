//! Token-level term analysis ([`Dimension`], [`Term`], [`TermAnalyzer`]) and
//! [`TermAnalyzingGlossaryMatcher`].

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use turbo_analysis::{Span, ViewBuilder};
use turbo_normalize::{
    AccentFold, CaseFold, ConfusableSkeleton, Dashes, DynNormalizer, EmojiToEmoticon, Normalizer, UnicodeForm,
    Whitespace,
};
use turbo_stem::{Algorithm, Stemmer};
use turbo_tokenize::uax29::{WordTokenizer, word_tokens};

use crate::selection;
use crate::{GlossaryEntry, GlossaryError, GlossaryMatch, GlossaryMatcher, is_blank};

/// A normalization layer in the [`Term`] stack, ordered in canonical
/// application order from least aggressive to most aggressive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dimension {
    /// The original token text.
    Original,
    /// Unicode canonical composition (NFC).
    Nfc,
    /// Unicode compatibility composition (NFKC).
    Nfkc,
    /// Unicode whitespace folded to ASCII spaces.
    Whitespace,
    /// Unicode dashes folded to the ASCII hyphen-minus.
    Dash,
    /// Case folding (default root locale or custom).
    CaseFold,
    /// Unicode full case folding (including expanding folds such as `ß` -> `ss`).
    FullCaseFold,
    /// Diacritic and accent folding.
    AccentFold,
    /// Emoji folded to ASCII emoticons where a mapping exists.
    EmojiFold,
    /// Confusable (homoglyph) skeleton folding per UTS #39.
    ConfusableFold,
    /// Stemming through a configured [`Stemmer`].
    Stem,
    /// Lemmatization (requires part-of-speech tags; unsupported on raw text).
    Lemma,
}

/// One analyzed token carrying its source span (when segmented from raw text)
/// and its normalized representations across configured [`Dimension`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    span: Option<Span>,
    original: String,
    final_dimension: Dimension,
    layers: BTreeMap<Dimension, String>,
}

impl Term {
    /// The byte span of this token in the segmented text, or `None` for
    /// pre-tokenized input.
    pub fn span(&self) -> Option<Span> {
        self.span
    }

    /// The original token text.
    pub fn original(&self) -> &str {
        &self.original
    }

    /// The token at the analyzer's final configured dimension, or
    /// [`Self::original`] when no dimensions were configured.
    pub fn normalized(&self) -> &str {
        self.layers.get(&self.final_dimension).map(String::as_str).unwrap_or(&self.original)
    }

    /// Returns the cached value at `dimension` if it was computed.
    pub fn at(&self, dimension: Dimension) -> Option<&str> {
        if dimension == Dimension::Original {
            Some(&self.original)
        } else {
            self.layers.get(&dimension).map(String::as_str)
        }
    }
}

/// Segments text with the UAX #29 [`WordTokenizer`] and applies a configured
/// stack of normalization [`Dimension`]s to each token.
#[derive(Clone)]
pub struct TermAnalyzer {
    chain: Vec<Dimension>,
    final_dimension: Dimension,
    transforms: BTreeMap<Dimension, Arc<dyn DynNormalizer>>,
    stemmer: Option<Arc<dyn Stemmer + Send + Sync>>,
    tokenizer: WordTokenizer,
}

impl std::fmt::Debug for TermAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermAnalyzer")
            .field("chain", &self.chain)
            .field("final_dimension", &self.final_dimension)
            .field("has_stemmer", &self.stemmer.is_some())
            .field("tokenizer", &self.tokenizer)
            .finish()
    }
}

impl TermAnalyzer {
    /// Returns a new [`TermAnalyzerBuilder`] with no dimensions enabled and
    /// the default UAX #29 [`WordTokenizer`].
    pub fn builder() -> TermAnalyzerBuilder {
        TermAnalyzerBuilder::new()
    }

    /// The configured dimensions in pipeline order (never includes
    /// [`Dimension::Original`]).
    pub fn dimensions(&self) -> &[Dimension] {
        &self.chain
    }

    /// The final configured dimension, or [`Dimension::Original`] when none
    /// are configured.
    pub fn final_dimension(&self) -> Dimension {
        self.final_dimension
    }

    /// Segments `text` with the UAX #29 word tokenizer and returns one
    /// [`Term`] per word token, in order.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::LemmaDimensionUnsupported`] if
    /// [`Dimension::Lemma`] is configured.
    pub fn analyze(&self, text: &str) -> Result<Vec<Term>, GlossaryError> {
        if self.chain.contains(&Dimension::Lemma) {
            return Err(GlossaryError::LemmaDimensionUnsupported);
        }
        let mut spans = Vec::new();
        let mut types = Vec::new();
        word_tokens(text, self.tokenizer.max_token_length(), &mut spans, &mut types);

        let mut terms = Vec::with_capacity(spans.len());
        for span in spans {
            let original = text[span.range()].to_owned();
            let mut layers = BTreeMap::new();
            let mut value = original.clone();
            for &dim in &self.chain {
                value = self.apply_dimension(dim, &value)?;
                layers.insert(dim, value.clone());
            }
            terms.push(Term { span: Some(span), original, final_dimension: self.final_dimension, layers });
        }
        Ok(terms)
    }

    fn apply_dimension(&self, dimension: Dimension, input: &str) -> Result<String, GlossaryError> {
        if let Some(custom) = self.transforms.get(&dimension) {
            let mut vb = ViewBuilder::new(input);
            custom.normalize_dyn(input, &mut vb);
            return Ok(vb.finish().0);
        }
        match dimension {
            Dimension::Original => Ok(input.to_owned()),
            Dimension::Nfc => Ok(apply_norm(&UnicodeForm::nfc(), input)),
            Dimension::Nfkc => Ok(apply_norm(&UnicodeForm::nfkc(), input)),
            Dimension::Whitespace => Ok(apply_norm(&Whitespace::collapse(), input)),
            Dimension::Dash => Ok(apply_norm(&Dashes::new(), input)),
            Dimension::CaseFold => Ok(apply_norm(&CaseFold::lowercase(), input)),
            Dimension::FullCaseFold => Ok(apply_norm(&CaseFold::full(), input)),
            Dimension::AccentFold => Ok(apply_norm(&AccentFold::new(), input)),
            Dimension::EmojiFold => Ok(apply_norm(&EmojiToEmoticon, input)),
            Dimension::ConfusableFold => Ok(apply_norm(&ConfusableSkeleton, input)),
            Dimension::Stem => {
                let stemmer = self.stemmer.as_ref().ok_or(GlossaryError::MissingStemmer)?;
                Ok(stemmer.stem(input))
            }
            Dimension::Lemma => Err(GlossaryError::LemmaDimensionUnsupported),
        }
    }
}

fn apply_norm<N: Normalizer>(norm: &N, input: &str) -> String {
    let mut vb = ViewBuilder::new(input);
    norm.normalize(input, &mut vb);
    vb.finish().0
}

/// Builder for [`TermAnalyzer`].
#[derive(Clone, Default)]
pub struct TermAnalyzerBuilder {
    chain: Vec<Dimension>,
    transforms: BTreeMap<Dimension, Arc<dyn DynNormalizer>>,
    stemmer: Option<Arc<dyn Stemmer + Send + Sync>>,
    tokenizer: WordTokenizer,
}

impl TermAnalyzerBuilder {
    /// Creates a new builder with no dimensions enabled.
    pub fn new() -> Self {
        Self::default()
    }

    fn enable(&mut self, dim: Dimension) {
        if !self.chain.contains(&dim) {
            self.chain.push(dim);
        }
    }

    /// Enables [`Dimension::Nfc`].
    pub fn nfc(mut self) -> Self {
        self.enable(Dimension::Nfc);
        self
    }

    /// Enables [`Dimension::Nfkc`].
    pub fn nfkc(mut self) -> Self {
        self.enable(Dimension::Nfkc);
        self
    }

    /// Enables [`Dimension::Whitespace`].
    pub fn whitespace(mut self) -> Self {
        self.enable(Dimension::Whitespace);
        self
    }

    /// Enables [`Dimension::Whitespace`] with a custom normalizer.
    pub fn whitespace_with(mut self, normalizer: Arc<dyn DynNormalizer>) -> Self {
        self.transforms.insert(Dimension::Whitespace, normalizer);
        self.enable(Dimension::Whitespace);
        self
    }

    /// Enables [`Dimension::Dash`].
    pub fn dash(mut self) -> Self {
        self.enable(Dimension::Dash);
        self
    }

    /// Enables [`Dimension::Dash`] with a custom normalizer.
    pub fn dash_with(mut self, normalizer: Arc<dyn DynNormalizer>) -> Self {
        self.transforms.insert(Dimension::Dash, normalizer);
        self.enable(Dimension::Dash);
        self
    }

    /// Enables [`Dimension::CaseFold`].
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::ConflictingCaseFolds`] if
    /// [`Dimension::FullCaseFold`] is already enabled.
    pub fn case_fold(mut self) -> Result<Self, GlossaryError> {
        if self.chain.contains(&Dimension::FullCaseFold) {
            return Err(GlossaryError::ConflictingCaseFolds);
        }
        self.enable(Dimension::CaseFold);
        Ok(self)
    }

    /// Enables [`Dimension::CaseFold`] with Turkish dotted/dotless `i` rules.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::ConflictingCaseFolds`] if
    /// [`Dimension::FullCaseFold`] is already enabled.
    pub fn case_fold_turkish(mut self) -> Result<Self, GlossaryError> {
        if self.chain.contains(&Dimension::FullCaseFold) {
            return Err(GlossaryError::ConflictingCaseFolds);
        }
        self.transforms.insert(Dimension::CaseFold, Arc::new(CaseFold::turkic_simple()));
        self.enable(Dimension::CaseFold);
        Ok(self)
    }

    /// Enables [`Dimension::FullCaseFold`].
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::ConflictingCaseFolds`] if
    /// [`Dimension::CaseFold`] is already enabled.
    pub fn full_case_fold(mut self) -> Result<Self, GlossaryError> {
        if self.chain.contains(&Dimension::CaseFold) {
            return Err(GlossaryError::ConflictingCaseFolds);
        }
        self.enable(Dimension::FullCaseFold);
        Ok(self)
    }

    /// Enables [`Dimension::AccentFold`].
    pub fn accent_fold(mut self) -> Self {
        self.enable(Dimension::AccentFold);
        self
    }

    /// Enables [`Dimension::EmojiFold`].
    pub fn emoji_fold(mut self) -> Self {
        self.enable(Dimension::EmojiFold);
        self
    }

    /// Enables [`Dimension::ConfusableFold`].
    pub fn confusable_fold(mut self) -> Self {
        self.enable(Dimension::ConfusableFold);
        self
    }

    /// Enables a character-level [`Dimension`] with a custom normalizer.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::InvalidTransformDimension`] for `Original`,
    /// `Stem`, or `Lemma`, or [`GlossaryError::ConflictingCaseFolds`] if both
    /// `CaseFold` and `FullCaseFold` are configured.
    pub fn transform(
        mut self,
        dimension: Dimension,
        normalizer: Arc<dyn DynNormalizer>,
    ) -> Result<Self, GlossaryError> {
        if matches!(dimension, Dimension::Original | Dimension::Stem | Dimension::Lemma) {
            return Err(GlossaryError::InvalidTransformDimension(dimension));
        }
        if dimension == Dimension::CaseFold && self.chain.contains(&Dimension::FullCaseFold) {
            return Err(GlossaryError::ConflictingCaseFolds);
        }
        if dimension == Dimension::FullCaseFold && self.chain.contains(&Dimension::CaseFold) {
            return Err(GlossaryError::ConflictingCaseFolds);
        }
        self.transforms.insert(dimension, normalizer);
        self.enable(dimension);
        Ok(self)
    }

    /// Enables [`Dimension::Stem`] with the given [`Stemmer`].
    pub fn stem<S: Stemmer + Send + Sync + 'static>(mut self, stemmer: S) -> Self {
        self.stemmer = Some(Arc::new(stemmer));
        self.enable(Dimension::Stem);
        self
    }

    /// Enables [`Dimension::Stem`] with a [`turbo_stem::Algorithm`].
    pub fn stem_algorithm(self, algorithm: Algorithm) -> Self {
        self.stem(algorithm)
    }

    /// Enables [`Dimension::Lemma`] (note: [`TermAnalyzingGlossaryMatcher`]
    /// rejects analyzers with [`Dimension::Lemma`] enabled because raw text
    /// has no POS tags).
    pub fn lemmatize(mut self) -> Self {
        self.enable(Dimension::Lemma);
        self
    }

    /// Sets the [`WordTokenizer`] used to segment text.
    pub fn tokenizer(mut self, tokenizer: WordTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Sets the maximum token length (in UTF-16 code units) of the
    /// [`WordTokenizer`].
    pub fn max_token_length(mut self, max_token_length: NonZeroU32) -> Self {
        self.tokenizer = WordTokenizer::with_max_token_length(max_token_length);
        self
    }

    /// Builds the configured [`TermAnalyzer`].
    pub fn build(mut self) -> TermAnalyzer {
        self.chain.sort();
        self.chain.dedup();
        let final_dimension = self.chain.last().copied().unwrap_or(Dimension::Original);
        TermAnalyzer {
            chain: self.chain,
            final_dimension,
            transforms: self.transforms,
            stemmer: self.stemmer,
            tokenizer: self.tokenizer,
        }
    }
}

#[derive(Clone, Copy)]
struct TokenCandidateHit {
    start: u32,
    end: u32,
    token_len: u32,
    pattern_index: u32,
}

/// A [`GlossaryMatcher`] that analyzes both glossary terms and input text
/// through a [`TermAnalyzer`] and matches sequences of normalized tokens with
/// a token-level Aho-Corasick automaton.
///
/// Matches span from the start of the first matched token to the end of the
/// last matched token in the original text (mapped through the optional
/// `pre_tokenizer_normalizer`'s [`turbo_analysis::Alignment`]). Overlapping
/// matches are resolved deterministically by preferring the earliest start
/// offset, then the longest character span, then the highest matched token
/// count, and finally the earliest entry in the input list.
pub struct TermAnalyzingGlossaryMatcher {
    entries: Vec<GlossaryEntry>,
    analyzer: TermAnalyzer,
    pre_tokenizer_normalizer: Option<Arc<dyn DynNormalizer>>,
    transitions: Vec<BTreeMap<String, u32>>,
    fail: Box<[u32]>,
    outputs: Vec<Box<[u32]>>,
    pattern_token_lengths: Box<[u32]>,
}

impl std::fmt::Debug for TermAnalyzingGlossaryMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermAnalyzingGlossaryMatcher")
            .field("entries_len", &self.entries.len())
            .field("analyzer", &self.analyzer)
            .field("has_pre_tokenizer_normalizer", &self.pre_tokenizer_normalizer.is_some())
            .finish()
    }
}

impl TermAnalyzingGlossaryMatcher {
    /// Builds a token-analyzing glossary matcher for `entries` and `analyzer`.
    ///
    /// # Errors
    ///
    /// Returns an error if `entries` is empty, if `analyzer` enables
    /// [`Dimension::Lemma`], or if any entry produces zero tokens or a blank
    /// normalized token.
    pub fn new(entries: &[GlossaryEntry], analyzer: TermAnalyzer) -> Result<Self, GlossaryError> {
        Self::build(entries, analyzer, None)
    }

    /// Builds a token-analyzing glossary matcher that runs
    /// `pre_tokenizer_normalizer` on both glossary terms and input text before
    /// tokenizing with `analyzer`.
    pub fn with_normalizer(
        entries: &[GlossaryEntry],
        analyzer: TermAnalyzer,
        pre_tokenizer_normalizer: Arc<dyn DynNormalizer>,
    ) -> Result<Self, GlossaryError> {
        Self::build(entries, analyzer, Some(pre_tokenizer_normalizer))
    }

    fn build(
        entries: &[GlossaryEntry],
        analyzer: TermAnalyzer,
        pre_tokenizer_normalizer: Option<Arc<dyn DynNormalizer>>,
    ) -> Result<Self, GlossaryError> {
        if entries.is_empty() {
            return Err(GlossaryError::EmptyEntries);
        }
        if analyzer.dimensions().contains(&Dimension::Lemma) {
            return Err(GlossaryError::LemmaDimensionUnsupported);
        }

        let entries_vec = entries.to_vec();
        let mut pattern_token_lengths = Vec::with_capacity(entries_vec.len());
        let mut builder = TokenAutomatonBuilder::new();

        for (i, entry) in entries_vec.iter().enumerate() {
            let normalized_owned;
            let raw_term = if let Some(norm) = &pre_tokenizer_normalizer {
                let mut vb = ViewBuilder::new(entry.term());
                norm.normalize_dyn(entry.term(), &mut vb);
                normalized_owned = vb.finish().0;
                normalized_owned.as_str()
            } else {
                entry.term()
            };
            let terms = analyzer.analyze(raw_term)?;
            if terms.is_empty() {
                return Err(GlossaryError::ZeroTokensTerm {
                    entry_id: entry.id().to_owned(),
                    term: entry.term().to_owned(),
                });
            }
            let mut tokens = Vec::with_capacity(terms.len());
            for term in &terms {
                let value = term.normalized();
                if is_blank(value) {
                    return Err(GlossaryError::BlankTokenInTerm {
                        entry_id: entry.id().to_owned(),
                        term: entry.term().to_owned(),
                        token: term.original().to_owned(),
                    });
                }
                tokens.push(value.to_owned());
            }
            pattern_token_lengths.push(tokens.len() as u32);
            builder.insert(&tokens, i as u32);
        }

        let (transitions, fail, outputs) = builder.build();
        Ok(Self {
            entries: entries_vec,
            analyzer,
            pre_tokenizer_normalizer,
            transitions,
            fail,
            outputs,
            pattern_token_lengths: pattern_token_lengths.into_boxed_slice(),
        })
    }

    #[inline]
    fn step(&self, mut state: u32, token: &str) -> u32 {
        loop {
            if let Some(&next) = self.transitions[state as usize].get(token) {
                return next;
            }
            if state == 0 {
                return 0;
            }
            state = self.fail[state as usize];
        }
    }
}

impl GlossaryMatcher for TermAnalyzingGlossaryMatcher {
    fn match_text(&self, text: &str) -> Vec<GlossaryMatch> {
        let normalized_pair = self.pre_tokenizer_normalizer.as_ref().map(|norm| {
            let mut vb = ViewBuilder::new(text);
            norm.normalize_dyn(text, &mut vb);
            vb.finish()
        });
        let scan_text = match &normalized_pair {
            Some((norm_str, _)) => norm_str.as_str(),
            None => text,
        };
        if scan_text.is_empty() {
            return Vec::new();
        }

        let Ok(analyzed) = self.analyzer.analyze(scan_text) else {
            return Vec::new();
        };
        if analyzed.is_empty() {
            return Vec::new();
        }

        let mut token_values: Vec<&str> = Vec::with_capacity(analyzed.len());
        let mut token_spans: Vec<Span> = Vec::with_capacity(analyzed.len());
        for term in &analyzed {
            let value = term.normalized();
            if !is_blank(value)
                && let Some(span) = term.span()
            {
                let orig_span = match &normalized_pair {
                    Some((_, alignment)) => alignment.span_to_parent(span),
                    None => span,
                };
                token_values.push(value);
                token_spans.push(orig_span);
            }
        }
        if token_values.is_empty() {
            return Vec::new();
        }

        let mut hits: Vec<TokenCandidateHit> = Vec::new();
        let mut state: u32 = 0;

        for (t, &token) in token_values.iter().enumerate() {
            state = self.step(state, token);
            let out = &self.outputs[state as usize];
            if !out.is_empty() {
                for &pattern_index in out.iter() {
                    let token_len = self.pattern_token_lengths[pattern_index as usize];
                    let first_token = t + 1 - token_len as usize;
                    let start = token_spans[first_token].start;
                    let end = token_spans[t].end;
                    hits.push(TokenCandidateHit { start, end, token_len, pattern_index });
                }
            }
        }

        if hits.is_empty() {
            return Vec::new();
        }

        hits.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
                .then_with(|| b.token_len.cmp(&a.token_len))
                .then_with(|| a.pattern_index.cmp(&b.pattern_index))
        });

        let selected = selection::select(hits, |h| Span::new(h.start, h.end));
        selected
            .into_iter()
            .map(|hit| {
                let entry = &self.entries[hit.pattern_index as usize];
                GlossaryMatch::from_entry(Span::new(hit.start, hit.end), entry)
            })
            .collect()
    }
}

struct TokenAutomatonBuilder {
    transitions: Vec<BTreeMap<String, u32>>,
    outputs: Vec<Vec<u32>>,
}

impl TokenAutomatonBuilder {
    fn new() -> Self {
        Self { transitions: vec![BTreeMap::new()], outputs: vec![Vec::new()] }
    }

    fn insert(&mut self, tokens: &[String], pattern_index: u32) {
        let mut state: u32 = 0;
        for token in tokens {
            if let Some(&next) = self.transitions[state as usize].get(token) {
                state = next;
            } else {
                let next = self.transitions.len() as u32;
                self.transitions.push(BTreeMap::new());
                self.outputs.push(Vec::new());
                self.transitions[state as usize].insert(token.clone(), next);
                state = next;
            }
        }
        self.outputs[state as usize].push(pattern_index);
    }

    #[allow(clippy::type_complexity)]
    fn build(mut self) -> (Vec<BTreeMap<String, u32>>, Box<[u32]>, Vec<Box<[u32]>>) {
        let num_states = self.transitions.len();
        let mut fail = vec![0u32; num_states];
        let mut queue = Vec::with_capacity(num_states);

        for &target in self.transitions[0].values() {
            fail[target as usize] = 0;
            queue.push(target);
        }

        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;
            let edges: Vec<(String, u32)> = self.transitions[r as usize].iter().map(|(k, &v)| (k.clone(), v)).collect();
            for (token, u) in edges {
                queue.push(u);
                let mut f = fail[r as usize];
                let fallback = loop {
                    if let Some(&target) = self.transitions[f as usize].get(&token) {
                        break target;
                    }
                    if f == 0 {
                        break 0;
                    }
                    f = fail[f as usize];
                };
                fail[u as usize] = fallback;
                if !self.outputs[fallback as usize].is_empty() {
                    let inherited = self.outputs[fallback as usize].clone();
                    self.outputs[u as usize].extend_from_slice(&inherited);
                }
            }
        }

        let compiled_outputs = self.outputs.into_iter().map(Vec::into_boxed_slice).collect();
        (self.transitions, fail.into_boxed_slice(), compiled_outputs)
    }
}
