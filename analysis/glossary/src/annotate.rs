//! Analysis layer ([`GlossaryTerms`]) and annotators ([`GlossaryAnnotator`],
//! [`TokenGlossaryAnnotator`]) for glossary term matching.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, Layer, Overlap, Refs, Span, Spans, TextSpans, View, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_unicode::java::code_point;

use crate::selection;
use crate::{GlossaryEntry, GlossaryError, GlossaryMatch, GlossaryMatcher, is_blank};

/// Layer of matched glossary terms (`turbo:glossary-terms`), anchored to a
/// [`View`] and carrying each matched entry's concept/term identifier and
/// canonical glossary surface form.
#[derive(Debug)]
pub struct GlossaryTerms {
    pub spans: Spans<GlossaryTerms>,
    pub id: Strs,
    pub term: Strs,
}

impl GlossaryTerms {
    /// Creates an empty [`GlossaryTerms`] layer anchored to `view`.
    pub fn new(view: ViewId) -> GlossaryTerms {
        GlossaryTerms { spans: Spans::new(view), id: Strs::default(), term: Strs::default() }
    }

    /// Creates a [`GlossaryTerms`] layer with preallocated capacity `n`.
    pub fn with_capacity(view: ViewId, n: usize) -> GlossaryTerms {
        GlossaryTerms { spans: Spans::with_capacity(view, n), id: Strs::default(), term: Strs::default() }
    }

    /// Appends a matched glossary entry to the layer.
    pub fn push(&mut self, span: Span, id: &str, term: &str) -> Ix<GlossaryTerms> {
        self.id.push(id);
        self.term.push(term);
        self.spans.push(span)
    }

    /// Returns the number of matched glossary terms in this layer.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Returns whether this layer contains no matches.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Returns the `(span, id, term)` triple at index `i`.
    pub fn get(&self, i: usize) -> (Span, &str, &str) {
        (self.spans.get(Ix::new(i as u32)), self.id.get(i), self.term.get(i))
    }

    /// Iterates over all `(span, id, term)` triples in this layer.
    pub fn iter(&self) -> impl Iterator<Item = (Span, &str, &str)> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

impl Layer for GlossaryTerms {
    const ID: &'static str = "turbo:glossary-terms";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.id.len() != n || self.term.len() != n {
            return Err("an id and a term for every glossary match".into());
        }
        Ok(())
    }
}

/// An [`Annotator`] that runs a [`GlossaryMatcher`] over a [`View`] and writes
/// a [`GlossaryTerms`] layer.
#[derive(Clone)]
pub struct GlossaryAnnotator {
    matcher: Arc<dyn GlossaryMatcher>,
}

impl std::fmt::Debug for GlossaryAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlossaryAnnotator").finish()
    }
}

impl GlossaryAnnotator {
    /// Creates a new [`GlossaryAnnotator`] wrapping `matcher`.
    pub fn new<M: GlossaryMatcher + 'static>(matcher: M) -> Self {
        Self { matcher: Arc::new(matcher) }
    }

    /// Creates a new [`GlossaryAnnotator`] from an `Arc<dyn GlossaryMatcher>`.
    pub fn from_arc(matcher: Arc<dyn GlossaryMatcher>) -> Self {
        Self { matcher }
    }

    /// Runs the underlying [`GlossaryMatcher`] on `text` and returns the
    /// matches directly.
    pub fn annotate_text(&self, text: &str) -> Vec<GlossaryMatch> {
        self.matcher.match_text(text)
    }
}

impl Annotator for GlossaryAnnotator {
    const ID: &'static str = "turbo:glossary-annotator";
    type Reads = (View,);
    type Writes = (GlossaryTerms,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(GlossaryTerms,), Error> {
        let matches = self.matcher.match_text(view.text());
        Ok((matches_to_layer(view.id(), &matches),))
    }
}

/// Converts a slice of [`GlossaryMatch`] items into a [`GlossaryTerms`] layer
/// anchored to `view`.
pub fn matches_to_layer(view: ViewId, matches: &[GlossaryMatch]) -> GlossaryTerms {
    let mut layer = GlossaryTerms::with_capacity(view, matches.len());
    for m in matches {
        layer.push(m.span(), m.id(), m.term());
    }
    layer
}

#[derive(Clone, Copy)]
struct TokenSpanHit {
    start: u32,
    end: u32,
    token_len: u32,
    pattern_index: u32,
}

/// An [`Annotator`] that matches glossary entries as contiguous sequences of
/// [`Tokens`] in a [`View`] and writes a [`GlossaryTerms`] layer.
///
/// Each glossary entry's term is split on Unicode whitespace into token strings
/// at construction time. At annotation time, token spans from the [`Tokens`]
/// layer are matched with a token-level Aho-Corasick automaton, and overlapping
/// matches are resolved longest-first (`start ASC, end DESC, token_len DESC,
/// pattern_index ASC`).
pub struct TokenGlossaryAnnotator {
    entries: Vec<GlossaryEntry>,
    ignore_case: bool,
    transitions: Vec<std::collections::BTreeMap<String, u32>>,
    fail: Box<[u32]>,
    outputs: Vec<Box<[u32]>>,
    pattern_token_lengths: Box<[u32]>,
}

impl std::fmt::Debug for TokenGlossaryAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenGlossaryAnnotator")
            .field("entries_len", &self.entries.len())
            .field("ignore_case", &self.ignore_case)
            .finish()
    }
}

impl TokenGlossaryAnnotator {
    /// Builds a token-anchored glossary annotator from `entries`, splitting
    /// multi-word glossary terms on Unicode whitespace.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::EmptyEntries`] if `entries` is empty, or
    /// [`GlossaryError::ZeroTokensTerm`] if an entry yields no tokens.
    pub fn new(entries: &[GlossaryEntry], ignore_case: bool) -> Result<Self, GlossaryError> {
        if entries.is_empty() {
            return Err(GlossaryError::EmptyEntries);
        }
        let entries_vec = entries.to_vec();
        let mut pattern_token_lengths = Vec::with_capacity(entries_vec.len());
        let mut transitions = vec![std::collections::BTreeMap::<String, u32>::new()];
        let mut outputs = vec![Vec::<u32>::new()];

        for (i, entry) in entries_vec.iter().enumerate() {
            let tokens: Vec<String> = entry
                .term()
                .split(|c: char| turbo_unicode::WhitespaceMode::Unicode.is_whitespace(c as u32))
                .filter(|s| !s.is_empty() && !is_blank(s))
                .map(|s| fold_token(s, ignore_case))
                .collect();
            if tokens.is_empty() {
                return Err(GlossaryError::ZeroTokensTerm {
                    entry_id: entry.id().to_owned(),
                    term: entry.term().to_owned(),
                });
            }
            pattern_token_lengths.push(tokens.len() as u32);
            let mut state = 0u32;
            for tok in tokens {
                if let Some(&next) = transitions[state as usize].get(&tok) {
                    state = next;
                } else {
                    let next = transitions.len() as u32;
                    transitions.push(std::collections::BTreeMap::new());
                    outputs.push(Vec::new());
                    transitions[state as usize].insert(tok, next);
                    state = next;
                }
            }
            outputs[state as usize].push(i as u32);
        }

        let num_states = transitions.len();
        let mut fail = vec![0u32; num_states];
        let mut queue = Vec::with_capacity(num_states);
        for &target in transitions[0].values() {
            fail[target as usize] = 0;
            queue.push(target);
        }
        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;
            let edges: Vec<(String, u32)> = transitions[r as usize].iter().map(|(k, &v)| (k.clone(), v)).collect();
            for (tok, u) in edges {
                queue.push(u);
                let mut f = fail[r as usize];
                let fallback = loop {
                    if let Some(&target) = transitions[f as usize].get(&tok) {
                        break target;
                    }
                    if f == 0 {
                        break 0;
                    }
                    f = fail[f as usize];
                };
                fail[u as usize] = fallback;
                if !outputs[fallback as usize].is_empty() {
                    let inherited = outputs[fallback as usize].clone();
                    outputs[u as usize].extend_from_slice(&inherited);
                }
            }
        }

        let compiled_outputs = outputs.into_iter().map(Vec::into_boxed_slice).collect();
        Ok(Self {
            entries: entries_vec,
            ignore_case,
            transitions,
            fail: fail.into_boxed_slice(),
            outputs: compiled_outputs,
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

    /// Matches glossary entries against `tokens` in `view` and returns the
    /// resolved [`GlossaryMatch`] list.
    pub fn match_tokens(&self, view: &View, tokens: &Tokens) -> Vec<GlossaryMatch> {
        let text = view.text();
        if tokens.spans.is_empty() {
            return Vec::new();
        }

        let mut hits: Vec<TokenSpanHit> = Vec::new();
        let mut state = 0u32;

        for (t, (_, span)) in tokens.spans.iter().enumerate() {
            let raw = &text[span.range()];
            let key = fold_token(raw, self.ignore_case);
            state = self.step(state, &key);
            let out = &self.outputs[state as usize];
            if !out.is_empty() {
                for &pattern_index in out.iter() {
                    let token_len = self.pattern_token_lengths[pattern_index as usize];
                    let first_tok = t + 1 - token_len as usize;
                    let first_span = tokens.spans.get(Ix::new(first_tok as u32));
                    hits.push(TokenSpanHit { start: first_span.start, end: span.end, token_len, pattern_index });
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

impl Annotator for TokenGlossaryAnnotator {
    const ID: &'static str = "turbo:token-glossary-annotator";
    type Reads = (View, Tokens);
    type Writes = (GlossaryTerms,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(GlossaryTerms,), Error> {
        let matches = self.match_tokens(view, tokens);
        Ok((matches_to_layer(view.id(), &matches),))
    }
}

fn fold_token(token: &str, ignore_case: bool) -> String {
    if !ignore_case {
        return token.to_owned();
    }
    token.chars().map(|c| char::from_u32(code_point::to_lower_case(c as u32)).unwrap_or(c)).collect()
}
