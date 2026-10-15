//! The ruler as an analysis step.

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Entities, Lemmas, Numbers, PosTags, Sentences, Tokens};
use turbo_matcher_layers::Matches;
use turbo_stem_layers::Stems;
use turbo_stopword_layers::Stopwords;

use crate::attr::{Attr, Flag};
use crate::ruler::Ruler;
use crate::seq::Sequence;

/// Reads a text view, its `opennlp:tokens` and whichever token layers the
/// rules test, and writes `turbo:matches`. Bind the output to a named
/// instance (`Bound::write(0, "places")`) to keep several rule sets apart.
///
/// The optional layers are attached only when a rule reads them: sentences
/// for `Flag::SentStart` when the tokens do not say which sentence each is
/// in, number mentions for `Flag::LikeNum`. A rule that reads a layer the
/// document lacks fails the step, and a rule that reads `Attr::Column`
/// cannot run as a step, since a column is the caller's own.
pub struct RulerAnnotator {
    ruler: Ruler,
}

impl RulerAnnotator {
    pub fn new(ruler: Ruler) -> RulerAnnotator {
        RulerAnnotator { ruler }
    }

    pub fn ruler(&self) -> &Ruler {
        &self.ruler
    }

    fn reads(&self, attr: Attr) -> bool {
        self.ruler.matcher().is_some_and(|m| m.reads().contains(&attr))
            || self.ruler.phrases().iter().any(|p| p.on() == crate::phrase::PhraseOn::Attr(attr))
    }

    fn tests(&self, flag: Flag) -> bool {
        self.ruler.matcher().is_some_and(|m| m.tests_flag(flag))
            || self.ruler.phrases().iter().any(|p| p.on() == crate::phrase::PhraseOn::Flag(flag))
    }
}

fn fail(detail: impl ToString) -> Error {
    Error::Annotator { annotator: RulerAnnotator::ID, detail: detail.to_string() }
}

impl Annotator for RulerAnnotator {
    const ID: &'static str = "turbo-matcher:ruler";
    #[allow(clippy::type_complexity)]
    type Reads = (
        View,
        Tokens,
        Option<Sentences>,
        Option<PosTags>,
        Option<Lemmas>,
        Option<Entities>,
        Option<Stopwords>,
        Option<Stems>,
        Option<Numbers>,
    );
    type Writes = (Matches,);

    fn annotate(
        &self,
        (view, tokens, sentences, pos, lemmas, entities, stopwords, stems, numbers): Refs<'_, Self>,
    ) -> Result<(Matches,), Error> {
        let mut seq = Sequence::tokens(view, tokens).map_err(fail)?;
        if self.reads(Attr::Pos)
            && let Some(p) = pos
        {
            seq = seq.with_pos(p).map_err(fail)?;
        }
        if self.reads(Attr::Lemma)
            && let Some(l) = lemmas
        {
            seq = seq.with_lemmas(l).map_err(fail)?;
        }
        if self.reads(Attr::Stem)
            && let Some(s) = stems
        {
            seq = seq.with_stems(s).map_err(fail)?;
        }
        if self.reads(Attr::Entity)
            && let Some(e) = entities
        {
            seq = seq.with_entities(e).map_err(fail)?;
        }
        if self.tests(Flag::Stop)
            && let Some(s) = stopwords
        {
            seq = seq.with_stopwords(s).map_err(fail)?;
        }
        if self.tests(Flag::SentStart)
            && tokens.sentence.is_none()
            && let Some(s) = sentences
        {
            seq = seq.with_sentences(view, s).map_err(fail)?;
        }
        if self.tests(Flag::LikeNum)
            && let Some(n) = numbers
        {
            seq = seq.with_numbers(view, n).map_err(fail)?;
        }
        let layer = self.ruler.layer(&seq, view.id()).map_err(fail)?;
        Ok((layer,))
    }

    fn config_hash(&self) -> u64 {
        self.ruler.config_hash()
    }
}
