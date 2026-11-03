//! `POSTaggerME.tag`: the best tag sequence of each sentence by beam
//! search, and the annotator that writes it as `PosTags`.

use std::sync::{Arc, Mutex};

use turbo_analysis::{Annotator, Error, Key, Locality, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Sentences, Tag, Tokens};
use turbo_sequence::BeamSearch;
use turbo_sequence::postag::PosContext;

use crate::model::PosModel;

/// Tags the tokens of each sentence with a maxent model, writing a tag
/// and its probability per token.
///
/// The search is `BeamSearch.bestSequence` over the model's context
/// generator and sequence validator, as `POSTaggerME.tag(String[])` runs
/// it, so the tags are the JVM's and the probabilities (`probs()`) are
/// bit for bit the JVM's doubles. A sentence is tagged on its own; the
/// features of a token never see another sentence.
pub struct PosTagger {
    model: Arc<PosModel>,
    /// Context generators not in use. Each keeps a cache of the sentence
    /// it is on, so a search takes one for itself and gives it back.
    contexts: Mutex<Vec<PosContext>>,
}

impl std::fmt::Debug for PosTagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PosTagger").field("model", &self.model).finish_non_exhaustive()
    }
}

impl PosTagger {
    pub fn new(model: impl Into<Arc<PosModel>>) -> PosTagger {
        PosTagger { model: model.into(), contexts: Mutex::new(Vec::new()) }
    }

    /// The model it runs.
    pub fn pos_model(&self) -> &PosModel {
        &self.model
    }

    /// Tags one sentence: appends a tag and its probability per word to
    /// `tags` and `probs`, in word order, as `tag` and `probs` give them.
    /// Returns false, appending nothing, when no tag sequence scored
    /// above the search's minimum, where `tag` returns an array of nulls.
    pub fn tag(&self, words: &[&str], tags: &mut Vec<Tag>, probs: &mut Vec<f64>) -> Result<bool, Error> {
        let mut context = self.take_context()?;
        let found = self.tag_with(words, &mut context, tags, probs);
        self.contexts.lock().unwrap_or_else(|e| e.into_inner()).push(context);
        Ok(found)
    }

    /// [`tag`](Self::tag) with a context generator the caller holds, such
    /// as one from [`PosModel::new_context`].
    pub fn tag_with(
        &self,
        words: &[&str],
        context: &mut PosContext,
        tags: &mut Vec<Tag>,
        probs: &mut Vec<f64>,
    ) -> bool {
        self.tag_sentence(&self.search(), words, context, tags, probs)
    }

    /// The search over the model, built once per call of `tag` and once
    /// per document of `annotate`.
    fn search(&self) -> BeamSearch<'_> {
        BeamSearch::new(self.model.beam_size(), self.model.maxent())
            .expect("the outcomes were checked to be text when the model was opened")
    }

    /// One sentence through `search`: what `tag_with` appends.
    fn tag_sentence(
        &self,
        search: &BeamSearch<'_>,
        words: &[&str],
        context: &mut PosContext,
        tags: &mut Vec<Tag>,
        probs: &mut Vec<f64>,
    ) -> bool {
        let Some(best) = search.best_sequence(words, context, &self.model.validator()) else {
            return false;
        };
        tags.extend(best.outcomes().iter().map(|o| self.model.tag_of(o).expect("an outcome of the classifier")));
        probs.extend_from_slice(best.probs());
        true
    }

    /// A context generator from the pool, or one freshly built.
    fn take_context(&self) -> Result<PosContext, Error> {
        if let Some(c) = self.contexts.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(c);
        }
        self.model.new_context().map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })
    }
}

/// Slices `text[starts[i]..ends[i]]` for `i in from..to` into `words`, returning
/// [`Error::Annotator`] if any token span is inverted, out of bounds, or off a
/// UTF-8 character boundary.
pub(crate) fn slice_sentence_words<'a>(
    annotator: &'static str,
    text: &'a str,
    starts: &[u32],
    ends: &[u32],
    from: usize,
    to: usize,
    words: &mut Vec<&'a str>,
) -> Result<(), Error> {
    words.clear();
    words.reserve(to - from);
    for i in from..to {
        let start = starts[i] as usize;
        let end = ends[i] as usize;
        let word = text.get(start..end).ok_or_else(|| Error::Annotator {
            annotator,
            detail: format!("token {i} span {start}..{end} is outside the text or off a UTF-8 boundary"),
        })?;
        words.push(word);
    }
    Ok(())
}

impl Annotator for PosTagger {
    const ID: &'static str = "turbo-postag:maxent";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    /// Tags each sentence of the tokens (each run of tokens with the same
    /// sentence, or all the tokens when they were not made per sentence).
    /// The tokens must be spans of the view read.
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is tagged",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        let mut tags = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        let mut words: Vec<&str> = Vec::new();
        let mut context = self.take_context()?;
        let search = self.search();
        let ranges: Vec<(usize, usize)> = tokens.sentence_ranges().map(|(_, r)| (r.start.get(), r.end.get())).collect();
        for (from, to) in ranges {
            if let Err(err) = slice_sentence_words(Self::ID, text, starts, ends, from, to, &mut words) {
                self.contexts.lock().unwrap_or_else(|e| e.into_inner()).push(context);
                return Err(err);
            }
            if !self.tag_sentence(&search, &words, &mut context, &mut tags, &mut probs) {
                self.contexts.lock().unwrap_or_else(|e| e.into_inner()).push(context);
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("no tag sequence of tokens {from}..{to} scored above the minimum"),
                });
            }
        }
        self.contexts.lock().unwrap_or_else(|e| e.into_inner()).push(context);
        Ok((PosTags { tagset: self.model.tagset().clone(), tags, prob: Some(probs) },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    fn requires(&self) -> &[turbo_analysis::KeyId] {
        const REQUIRES: &[turbo_analysis::KeyId] =
            &[turbo_analysis::Key::<View>::DEFAULT.id(), turbo_analysis::Key::<Tokens>::DEFAULT.id()];
        REQUIRES
    }

    fn produces(&self) -> &[turbo_analysis::KeyId] {
        const PRODUCES: &[turbo_analysis::KeyId] = &[turbo_analysis::Key::<PosTags>::DEFAULT.id()];
        PRODUCES
    }

    /// FNV-1a over the tag format asked for, the beam size and the context
    /// cache size, the three settings a `POSTaggerME` is built with.
    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new()
            .u64(match self.model.requested_format() {
                turbo_sequence::postag::PosTagFormat::Ud => 1,
                turbo_sequence::postag::PosTagFormat::Penn => 2,
                turbo_sequence::postag::PosTagFormat::Custom => 3,
                turbo_sequence::postag::PosTagFormat::Unknown => 4,
            })
            .u64(self.model.beam_size() as u64)
            .u64(self.model.context_cache_size() as u64)
            .finish()
    }

    fn locality(&self) -> Locality {
        Locality::Partition(Key::<Sentences>::DEFAULT.id())
    }
}
