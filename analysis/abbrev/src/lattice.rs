//! Intra-document short-form <-> long-form abbreviation lattice emission
//! (issue #2340).

use turbo_analysis::{Annotator, Edges, Error, Refs, View};
use turbo_analysis_layers::{
    AbbreviationForm, Abbreviations, Stems, TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens,
};
use turbo_unicode::java::to_lowercase;

fn maybe_fold(s: &str, lowercase: bool) -> String {
    if !lowercase {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    to_lowercase(s, &mut out);
    out
}

/// Emits bidirectional short-form <-> long-form arcs onto `builder` for every
/// definition and later mention recorded in `abbreviations` + `links`.
pub fn emit_abbreviation_lattice(
    abbreviations: &Abbreviations,
    links: &Edges<Abbreviations, Abbreviations>,
    tokens: &Tokens,
    builder: &mut TokenGraphBuilder,
    lowercase: bool,
) {
    for (ix, span) in abbreviations.spans.iter() {
        let form = abbreviations.form(ix);
        if form == AbbreviationForm::Long {
            continue;
        }
        let Some(long_ix) = abbreviations.long_form(links, ix) else {
            continue;
        };
        let long_span = abbreviations.spans.get(long_ix);
        let short_toks = tokens.spans.within(span);
        let long_toks = tokens.spans.within(long_span);
        let s_range = short_toks.start.get()..short_toks.end.get();
        let l_range = long_toks.start.get()..long_toks.end.get();
        if s_range.is_empty() || l_range.is_empty() {
            continue;
        }
        if s_range.end > builder.token_count() || l_range.end > builder.token_count() {
            continue;
        }
        let short_words: Vec<String> = s_range.clone().map(|t| maybe_fold(builder.base_term(t), lowercase)).collect();
        let long_words: Vec<String> = l_range.clone().map(|t| maybe_fold(builder.base_term(t), lowercase)).collect();
        if short_words == long_words {
            continue;
        }
        if form == AbbreviationForm::Short {
            builder.add_span_sequence(l_range, long_span, short_words, TokenArcKind::AbbreviationShort);
        }
        builder.add_span_sequence(s_range, span, long_words, TokenArcKind::AbbreviationLong);
    }
}

/// Pipeline step emitting a [`TokenGraph`] enriched with bidirectional
/// abbreviation short-form and long-form arcs.
#[derive(Copy, Clone, Debug, Default)]
pub struct AbbreviationLatticeAnnotator {
    lowercase: bool,
}

impl AbbreviationLatticeAnnotator {
    pub fn new() -> Self {
        AbbreviationLatticeAnnotator { lowercase: false }
    }

    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }
}

impl Annotator for AbbreviationLatticeAnnotator {
    const ID: &'static str = "turbo-abbrev:token-lattice";
    type Reads = (View, Tokens, Abbreviations, Edges<Abbreviations, Abbreviations>, Option<Stems>);
    type Writes = (TokenGraph,);

    fn annotate(&self, (view, tokens, abbrevs, links, stems): Refs<'_, Self>) -> Result<(TokenGraph,), Error> {
        if tokens.spans.view() != view.id() || abbrevs.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "tokens and abbreviations must belong to the target view".into(),
            });
        }
        let mut builder = TokenGraphBuilder::from_tokens(view.text(), tokens, stems)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        if self.lowercase {
            for i in 0..builder.token_count() {
                let folded = maybe_fold(builder.base_term(i), true);
                if folded != builder.base_term(i) {
                    builder.add_token_term(i, builder.base_span(i), folded, TokenArcKind::Primary);
                }
            }
        }
        emit_abbreviation_lattice(abbrevs, links, tokens, &mut builder, self.lowercase);
        let graph = builder.finish().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((graph,))
    }

    fn config_hash(&self) -> u64 {
        u64::from(self.lowercase)
    }
}
