//! Glossary alias and concept synonym lattice emission (issue #2340).

use std::collections::HashMap;

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Stems, TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};
use turbo_glossary_layers::GlossaryTerms;
use turbo_unicode::java::to_lowercase;
use turbo_unicode::whitespace::is_whitespace;

fn split_glossary_words(term: &str, lowercase: bool) -> Vec<String> {
    term.split(|c: char| is_whitespace(c as u32))
        .filter(|w| !w.is_empty())
        .map(|w| {
            if lowercase {
                let mut s = String::with_capacity(w.len());
                to_lowercase(w, &mut s);
                s
            } else {
                w.to_owned()
            }
        })
        .collect()
}

/// Emits [`TokenArcKind::Synonym`] arcs onto `builder` for every matched
/// [`GlossaryTerms`] item, expanding all co-indexed aliases sharing the same
/// glossary `id` (and optionally the concept `id` itself).
pub fn emit_glossary_lattice(
    glossary_terms: &GlossaryTerms,
    builder: &mut TokenGraphBuilder,
    lowercase: bool,
    emit_concept_id: bool,
) {
    let glossary = &glossary_terms.glossary;
    let mut by_id: HashMap<&str, Vec<Vec<String>>> = HashMap::new();
    for (_, entry) in glossary.iter() {
        let words = split_glossary_words(entry.term, lowercase);
        if !words.is_empty() {
            by_id.entry(entry.id).or_default().push(words);
        }
    }

    for i in 0..glossary_terms.spans.len() {
        let Some(entry) = glossary_terms.try_entry_of(i) else {
            continue;
        };
        let tok_range = glossary_terms.tokens[i];
        let range = tok_range.start.get()..tok_range.end.get();
        if range.is_empty() || range.end > builder.token_count() {
            continue;
        }
        let span = glossary_terms.spans.get(turbo_analysis::Ix::new(i as u32));
        let matched_words = split_glossary_words(entry.term, lowercase);
        if let Some(aliases) = by_id.get(entry.id) {
            for alias in aliases {
                if *alias != matched_words {
                    builder.add_span_sequence(range.clone(), span, alias.iter().cloned(), TokenArcKind::Synonym);
                }
            }
        }
        if emit_concept_id {
            let id_words = split_glossary_words(entry.id, lowercase);
            if !id_words.is_empty() && id_words != matched_words {
                builder.add_span_sequence(range, span, id_words, TokenArcKind::Synonym);
            }
        }
    }
}

/// Pipeline step emitting a [`TokenGraph`] from [`GlossaryTerms`] alias groups.
#[derive(Copy, Clone, Debug, Default)]
pub struct GlossaryLatticeAnnotator {
    lowercase: bool,
    emit_concept_id: bool,
}

impl GlossaryLatticeAnnotator {
    pub fn new() -> Self {
        GlossaryLatticeAnnotator { lowercase: true, emit_concept_id: false }
    }

    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn with_concept_id(mut self, emit_concept_id: bool) -> Self {
        self.emit_concept_id = emit_concept_id;
        self
    }
}

impl Annotator for GlossaryLatticeAnnotator {
    const ID: &'static str = "turbo-glossary:token-lattice";
    type Reads = (View, Tokens, GlossaryTerms, Option<Stems>);
    type Writes = (TokenGraph,);

    fn annotate(&self, (view, tokens, glossary_terms, stems): Refs<'_, Self>) -> Result<(TokenGraph,), Error> {
        if tokens.spans.view() != view.id() || glossary_terms.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "tokens and glossary_terms must belong to the target view".into(),
            });
        }
        let mut builder = TokenGraphBuilder::from_tokens(view.text(), tokens, stems)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        emit_glossary_lattice(glossary_terms, &mut builder, self.lowercase, self.emit_concept_id);
        let graph = builder.finish().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((graph,))
    }

    fn config_hash(&self) -> u64 {
        u64::from(self.lowercase) | (u64::from(self.emit_concept_id) << 1)
    }
}
