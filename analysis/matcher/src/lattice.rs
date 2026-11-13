//! Multi-token phrase synonym lattice emission built on [`PhraseMatcher`]
//! (issue #2340).

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Lemmas, Stems, TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};

use crate::attr::{Attr, Field};
use crate::error::{MatchError, PatternError};
use crate::input::Input;
use crate::phrase::PhraseMatcher;

/// Maps multi-token or single-token phrases matched by a [`PhraseMatcher`]
/// onto synonym token sequences in a [`TokenGraphBuilder`].
pub struct PhraseSynonymExpander {
    matcher: PhraseMatcher,
    expansions: Vec<Vec<Vec<String>>>,
}

impl PhraseSynonymExpander {
    pub fn new(field: Field) -> Self {
        PhraseSynonymExpander { matcher: PhraseMatcher::new(field), expansions: Vec::new() }
    }

    pub fn lower() -> Self {
        Self::new(Field::Attr(Attr::Lower))
    }

    /// Registers a directional synonym `trigger -> target_tokens`.
    pub fn add_synonym(&mut self, trigger: &[&str], target_tokens: &[&str]) -> Result<(), PatternError> {
        if trigger.is_empty() || target_tokens.is_empty() {
            return Err(PatternError::Empty);
        }
        let target: Vec<String> = target_tokens.iter().map(|&s| s.to_owned()).collect();
        let key = format!("syn:{}", self.expansions.len());
        if let Some(existing_label) = self.matcher.label(&key) {
            let idx = existing_label.0 as usize;
            self.expansions[idx].push(target);
            return Ok(());
        }
        self.matcher.add(&key, trigger)?;
        let label = self.matcher.label(&key).expect("label just added");
        let idx = label.0 as usize;
        if self.expansions.len() <= idx {
            self.expansions.resize_with(idx + 1, Vec::new);
        }
        self.expansions[idx].push(target);
        Ok(())
    }

    /// Registers a symmetric equivalence class where every member phrase
    /// expands to every other member phrase in the group.
    pub fn add_equivalent_group(&mut self, phrases: &[&[&str]]) -> Result<(), PatternError> {
        for (i, &lhs) in phrases.iter().enumerate() {
            for (j, &rhs) in phrases.iter().enumerate() {
                if i != j {
                    self.add_synonym(lhs, rhs)?;
                }
            }
        }
        Ok(())
    }

    /// Emits [`TokenArcKind::Synonym`] arcs onto `builder` for every phrase
    /// match in `input`.
    pub fn emit(&self, input: &Input<'_>, tokens: &Tokens, builder: &mut TokenGraphBuilder) -> Result<(), MatchError> {
        let matches = self.matcher.find(input)?;
        let (starts, ends) = tokens.spans.columns();
        for m in matches {
            let start = m.tokens.start.get();
            let end = m.tokens.end.get();
            if start >= end || end > builder.token_count() || end > starts.len() {
                continue;
            }
            let span = Span::new(starts[start], ends[end - 1]);
            if let Some(targets) = self.expansions.get(m.label.0 as usize) {
                for target in targets {
                    builder.add_span_sequence(start..end, span, target.iter().cloned(), TokenArcKind::Synonym);
                }
            }
        }
        Ok(())
    }
}

/// Pipeline step emitting a [`TokenGraph`] with [`PhraseSynonymExpander`] arcs.
#[derive(Clone)]
pub struct PhraseSynonymAnnotator {
    expander: Arc<PhraseSynonymExpander>,
}

impl PhraseSynonymAnnotator {
    pub fn new(expander: Arc<PhraseSynonymExpander>) -> Self {
        PhraseSynonymAnnotator { expander }
    }
}

impl Annotator for PhraseSynonymAnnotator {
    const ID: &'static str = "turbo-matcher:phrase-synonym-lattice";
    type Reads = (View, Tokens, Option<Stems>, Option<Lemmas>);
    type Writes = (TokenGraph,);

    fn annotate(&self, (view, tokens, stems, lemmas): Refs<'_, Self>) -> Result<(TokenGraph,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens belong to another view".into() });
        }
        let mut input = Input::new(view.text(), tokens)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        if let Some(s) = stems {
            input = input
                .with_stems(&s.stem)
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        }
        if let Some(l) = lemmas {
            input = input
                .with_lemmas(&l.lemma)
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        }
        let mut builder = TokenGraphBuilder::from_tokens(view.text(), tokens, stems)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        self.expander
            .emit(&input, tokens, &mut builder)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        let graph = builder.finish().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((graph,))
    }
}
