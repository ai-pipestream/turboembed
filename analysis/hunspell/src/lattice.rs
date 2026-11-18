//! Hunspell compound-decomposition lattice emission (issue #2340).

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};
use turbo_stem_layers::Stems;
use turbo_unicode::java::to_lowercase;

use crate::dictionary::Dictionary;
use crate::stemmer::{CompoundSegment, Stemmer};

fn maybe_fold(text: &str, lowercase: bool) -> String {
    if !lowercase {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    to_lowercase(text, &mut out);
    out
}

/// Populates `builder` with decomposed compound part arcs (`CompoundPart`)
/// alongside whole-compound arcs (`CompoundWhole`) for tokens decomposed by
/// `stemmer`.
pub fn emit_hunspell_lattice(
    stemmer: &Stemmer<'_>,
    view_text: &str,
    tokens: &Tokens,
    builder: &mut TokenGraphBuilder,
    lowercase_stems: bool,
) {
    let n = tokens.spans.len().min(builder.token_count());
    let (starts, ends) = tokens.spans.columns();
    let mut cache: HashMap<&str, Option<Vec<CompoundSegment>>> = HashMap::new();
    for i in 0..n {
        let span = Span::new(starts[i], ends[i]);
        let Some(word) = view_text.get(span.range()) else {
            continue;
        };
        let segments = cache.entry(word).or_insert_with(|| stemmer.decompose_word(word));
        if let Some(parts) = segments {
            let mapped = parts.iter().map(|seg| {
                let sub_span =
                    Span::new(span.start + seg.byte_range.start as u32, span.start + seg.byte_range.end as u32);
                (sub_span, maybe_fold(&seg.stem, lowercase_stems), TokenArcKind::CompoundPart)
            });
            builder.split_token(i, mapped, TokenArcKind::CompoundWhole);
        }
    }
}

/// Pipeline step writing both `turbo:stems` and `turbo:token-graph` from a
/// Hunspell [`Dictionary`], decomposing compounds into sub-token lattice arcs
/// while keeping whole-word compound arcs spanning all parts.
pub struct HunspellLatticeAnnotator {
    dictionary: Arc<Dictionary>,
    lowercase: bool,
}

impl HunspellLatticeAnnotator {
    pub fn new(dictionary: Arc<Dictionary>) -> Self {
        HunspellLatticeAnnotator { dictionary, lowercase: true }
    }

    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }
}

impl Annotator for HunspellLatticeAnnotator {
    const ID: &'static str = "turbo-hunspell:compound-lattice";
    type Reads = (View, Tokens);
    type Writes = (Stems, TokenGraph);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Stems, TokenGraph), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens belong to another view".into() });
        }
        let text = view.text();
        let stemmer = Stemmer::new(&self.dictionary);
        let (starts, ends) = tokens.spans.columns();
        let mut stems = Stems::default();
        let mut base_spans = Vec::with_capacity(starts.len());
        let mut base_terms = Vec::with_capacity(starts.len());
        let mut seen: HashMap<&str, (String, String)> = HashMap::with_capacity(starts.len());

        for (&start, &end) in starts.iter().zip(ends) {
            let Some(word) = text.get(start as usize..end as usize) else {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("token span {start}..{end} is out of bounds or cuts a UTF-8 character"),
                });
            };
            let (first_stem, whole_term) = seen
                .entry(word)
                .or_insert_with(|| {
                    let s = stemmer.stem(word).unwrap_or_else(|| word.to_owned());
                    let whole = maybe_fold(word, self.lowercase);
                    (s, whole)
                })
                .clone();
            stems.stem.push(&first_stem);
            base_spans.push(Span::new(start, end));
            base_terms.push(whole_term);
        }

        let mut builder = TokenGraphBuilder::from_base_tokens(view.id(), base_spans, base_terms);
        let mut decomposed_cache: HashMap<&str, Option<Vec<CompoundSegment>>> = HashMap::new();
        for i in 0..starts.len() {
            let span = Span::new(starts[i], ends[i]);
            let word = &text[span.range()];
            let segments = decomposed_cache.entry(word).or_insert_with(|| stemmer.decompose_word(word));
            if let Some(parts) = segments {
                let mapped = parts.iter().map(|seg| {
                    let sub_span =
                        Span::new(span.start + seg.byte_range.start as u32, span.start + seg.byte_range.end as u32);
                    (sub_span, maybe_fold(&seg.stem, self.lowercase), TokenArcKind::CompoundPart)
                });
                builder.split_token(i, mapped, TokenArcKind::CompoundWhole);
            } else {
                let stem_term = maybe_fold(stems.stem.get(i), self.lowercase);
                if stem_term != builder.base_term(i) {
                    builder.add_token_term(i, builder.base_span(i), stem_term, TokenArcKind::Primary);
                }
            }
        }
        let graph = builder.finish().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((stems, graph))
    }

    fn config_hash(&self) -> u64 {
        self.dictionary.fingerprint() ^ u64::from(self.lowercase)
    }
}
