//! [`QuotationAnnotator`]: direct, indirect and mixed reported speech
//! (`turbo:quotations`) with speaker, reporting verb and speaker chain.
//!
//! Three sieves, most precise first:
//!
//! 1. With `Dependencies`, a verb of speech outside quotes that governs a
//!    token of a quote, or is governed by one (`"...", she said` attached
//!    as `parataxis`), attributes the quote to its subject; a `ccomp` of
//!    a verb of speech is indirect speech, or mixed when it quotes part of
//!    itself.
//! 2. Otherwise a verb of speech within a few tokens after the closing
//!    mark (`"...," she said`, `"...," said Smith`) or before the opening
//!    one (`She said, "..."`) attributes it; a quote interrupted by its
//!    attribution (`"A," she said, "B."`) gets it on both halves.
//! 3. A speaker is resolved to the coreference chain (`Groups<Mentions>`)
//!    of the mention that covers it, and an unattributed quote after two
//!    quotes of two different chains is given the chain before last, the
//!    turn-taking of a dialog.

use std::hash::Hasher;

use turbo_analysis::{Annotator, Document, Error, Groups, Key, Refs, View};
use turbo_analysis_layers::{Dependencies, Quotations, Tokens};

use crate::hash::Fnv;
use crate::layer::Mentions;

mod sieves;

/// Tuning for [`QuotationAnnotator`]; the default is the best setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuotationConfig {
    /// The most tokens between a quotation mark and its verb of speech for
    /// the surface sieve. Values larger than the document search to its ends.
    pub max_cue_distance: usize,
    /// Give an unattributed quote the speaker before last in a dialog.
    pub alternate_dialog: bool,
    /// Emit indirect and mixed speech found from `ccomp` arcs.
    pub indirect: bool,
}

impl Default for QuotationConfig {
    fn default() -> Self {
        QuotationConfig { max_cue_distance: 3, alternate_dialog: true, indirect: true }
    }
}

/// Writes [`Quotations`] for the default view.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuotationAnnotator {
    pub config: QuotationConfig,
}

impl QuotationAnnotator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: QuotationConfig) -> Self {
        self.config = config;
        self
    }

    /// The quotations of the default layers of `doc`, without adding them.
    /// Refuses inputs bound to another instance, as the Analyzer does.
    pub fn quotations(&self, doc: &Document) -> Result<Quotations, Error> {
        // Match Bound's alignment rule before pairing default input instances.
        let reads = [
            Key::<View>::DEFAULT.id(),
            Key::<Tokens>::DEFAULT.id(),
            Key::<Dependencies>::DEFAULT.id(),
            Key::<Mentions>::DEFAULT.id(),
            Key::<Groups<Mentions>>::DEFAULT.id(),
        ];
        for &layer in &reads {
            for &on in doc.made_on(layer) {
                if !reads.contains(&on)
                    && let Some(read) = reads.iter().find(|r| r.ty == on.ty && doc.present(**r))
                {
                    return Err(Error::Unaligned { layer, on, read: read.name });
                }
            }
        }
        let (q,) = self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<Tokens>::DEFAULT)?,
            doc.get_key(&Key::<Dependencies>::DEFAULT),
            doc.get_key(&Key::<Mentions>::DEFAULT),
            doc.get_key(&Key::<Groups<Mentions>>::DEFAULT),
        ))?;
        Ok(q)
    }
}

impl Annotator for QuotationAnnotator {
    const ID: &'static str = "turbo:quotation_annotator";
    const VERSION: &'static str = "2";

    type Reads = (View, Tokens, Option<Dependencies>, Option<Mentions>, Option<Groups<Mentions>>);
    type Writes = (Quotations,);

    /// FNV-1a over the version and every setting.
    fn config_hash(&self) -> u64 {
        let mut h = Fnv::default();
        h.str(Self::VERSION);
        h.write_u64(self.config.max_cue_distance as u64);
        h.write_u8(u8::from(self.config.alternate_dialog));
        h.write_u8(u8::from(self.config.indirect));
        h.finish()
    }

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (view, tokens, deps, mentions, chains) = input;
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are not of the view read".into() });
        }
        if mentions.is_some_and(|m| m.spans.view() != view.id()) {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "the mentions are not of the view read".into(),
            });
        }
        let text = view.text();
        let spans: Vec<_> = tokens.spans.iter().map(|(_, s)| s).collect();
        let forms: Vec<&str> = spans.iter().map(|s| text.get(s.start as usize..s.end as usize).unwrap_or("")).collect();
        let input = sieves::Input {
            spans: &spans,
            forms: &forms,
            deps: deps.filter(|d| d.head.len() == spans.len()),
            mentions,
            chains,
        };
        let mut out = Quotations::new(view.id());
        for q in sieves::run(&input, self.config) {
            out.push(q);
        }
        Ok((out,))
    }
}
