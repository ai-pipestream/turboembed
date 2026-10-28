//! `ChunkerME.chunk` and `chunkAsSpans`: the best chunk tag sequence of
//! each sentence by beam search, and the annotator that writes it as
//! `ChunkTags` and the phrases it makes as `PhraseChunks`.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, IxRange, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{ChunkTags, PhraseChunks, PosTags, Tag, Tokens};
use turbo_sequence::BeamSearch;
use turbo_sequence::chunker::{ChunkerValidator, DefaultChunkerContext, TokenTag};

use crate::model::ChunkerModel;
use crate::phrases::phrases;

/// Chunks the tagged tokens of each sentence with a maxent model: a chunk
/// tag and its probability per token, and the phrases the tags make.
///
/// The search is `BeamSearch.bestSequence` over
/// `DefaultChunkerContextGenerator` and `DefaultChunkerSequenceValidator`,
/// as `ChunkerME.chunk(String[], String[])` runs it, so the tags are the
/// JVM's and the probabilities (`probs()`) are bit for bit the JVM's
/// doubles; the phrases are `ChunkSample.phrasesAsSpanList` over those
/// tags, as `chunkAsSpans` returns them. A sentence is chunked on its
/// own.
#[derive(Clone, Debug)]
pub struct PhraseChunker {
    model: Arc<ChunkerModel>,
}

impl PhraseChunker {
    pub fn new(model: impl Into<Arc<ChunkerModel>>) -> PhraseChunker {
        PhraseChunker { model: model.into() }
    }

    /// The model it runs.
    pub fn chunker_model(&self) -> &ChunkerModel {
        &self.model
    }

    /// `chunk` for one sentence: appends the best chunk tag of each tagged
    /// word, as the classifier's outcome, and its probability, or appends
    /// nothing and returns false when no sequence scored above the
    /// search's minimum (where `chunk` returns an array of nulls).
    pub fn outcomes<'m>(&'m self, words: &[TokenTag<'_>], outcomes: &mut Vec<&'m str>, probs: &mut Vec<f64>) -> bool {
        Self::outcomes_with(&self.search(), words, &mut self.model.context(), &self.model.validator(), outcomes, probs)
    }

    /// The search over the model, built once per call of `outcomes` and
    /// once per document of `annotate`.
    fn search(&self) -> BeamSearch<'_> {
        BeamSearch::new(self.model.beam_size(), self.model.maxent())
            .expect("the outcomes were checked to be text when the model was opened")
    }

    /// One sentence through `search`: what `outcomes` appends.
    fn outcomes_with<'m>(
        search: &BeamSearch<'m>,
        words: &[TokenTag<'_>],
        context: &mut DefaultChunkerContext,
        validator: &ChunkerValidator,
        outcomes: &mut Vec<&'m str>,
        probs: &mut Vec<f64>,
    ) -> bool {
        let Some(best) = search.best_sequence(words, context, validator) else {
            return false;
        };
        outcomes.extend_from_slice(best.outcomes());
        probs.extend_from_slice(best.probs());
        true
    }

    /// [`outcomes`](Self::outcomes) with the tags as tags of the model's
    /// [tag set](ChunkerModel::tagset).
    pub fn chunk(&self, words: &[TokenTag<'_>], tags: &mut Vec<Tag>, probs: &mut Vec<f64>) -> bool {
        let mut outcomes = Vec::with_capacity(words.len());
        if !self.outcomes(words, &mut outcomes, probs) {
            return false;
        }
        tags.extend(outcomes.iter().map(|o| self.model.tag_of(o).expect("an outcome of the classifier")));
        true
    }
}

/// The `(from, to)` token ranges of each sentence of `tokens`, or one
/// range over all of them when they were not made per sentence.
pub(crate) fn sentence_ranges(tokens: &Tokens) -> Vec<(usize, usize)> {
    tokens.sentence_ranges().map(|(_, r)| (r.start.get(), r.end.get())).collect()
}

impl Annotator for PhraseChunker {
    const ID: &'static str = "turbo-phrase-chunker:maxent";
    type Reads = (View, Tokens, PosTags);
    type Writes = (ChunkTags, PhraseChunks);

    /// Chunks each sentence of the tokens (each run of tokens with the
    /// same sentence, or all the tokens when they were not made per
    /// sentence) with the tags' labels. The tokens must be spans of the
    /// view read, and the tags one per token. A phrase's span runs from
    /// the start of its first token to the end of its last.
    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(ChunkTags, PhraseChunks), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is chunked",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        if starts.len() != ends.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("token span start count ({}) does not match end count ({})", starts.len(), ends.len()),
            });
        }
        let n = starts.len();
        if let Some(col) = &tokens.sentence
            && col.len() != n
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} sentence indices for {n} tokens", col.len()),
            });
        }
        if tags.tags.len() != n {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} tags for {n} tokens", tags.tags.len()),
            });
        }
        let tagset_len = tags.tagset.len();
        for (i, &tag) in tags.tags.iter().enumerate() {
            if usize::from(tag.0) >= tagset_len {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "tag index {} at token {i} is out of bounds for tagset of length {tagset_len}",
                        tag.0
                    ),
                });
            }
        }
        for i in 0..n {
            let s = starts[i] as usize;
            let e = ends[i] as usize;
            if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "token {i} span {s}..{e} is not a valid UTF-8 slice of view text of length {}",
                        text.len()
                    ),
                });
            }
        }
        let mut chunk_tags = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        let mut spans = Spans::new(view.id());
        let mut ranges = Vec::new();
        let mut kind = Vec::new();
        let mut words: Vec<TokenTag<'_>> = Vec::new();
        let search = self.search();
        let mut context = self.model.context();
        let validator = self.model.validator();
        let mut outcomes: Vec<&str> = Vec::new();
        for (from, to) in sentence_ranges(tokens) {
            for i in (from + 1)..to {
                if starts[i] < starts[i - 1] || ends[i] < ends[i - 1] {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!(
                            "token {i} span {}..{} precedes previous token in sentence {from}..{to}",
                            starts[i], ends[i]
                        ),
                    });
                }
            }
            words.clear();
            words.extend((from..to).map(|i| TokenTag {
                token: &text[starts[i] as usize..ends[i] as usize],
                tag: tags.tagset.label(tags.tags[i]),
            }));
            outcomes.clear();
            if !Self::outcomes_with(&search, &words, &mut context, &validator, &mut outcomes, &mut probs) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("no chunk tag sequence of tokens {from}..{to} scored above the minimum"),
                });
            }
            chunk_tags.extend(outcomes.iter().map(|o| self.model.tag_of(o).expect("an outcome of the classifier")));
            let found = phrases(&outcomes)
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: format!("tokens {from}..{to}: {e}") })?;
            for p in found {
                spans.push(Span::new(starts[from + p.start], ends[from + p.end - 1]));
                ranges.push(IxRange::new((from + p.start) as u32, (from + p.end) as u32));
                kind.push(self.model.kinds().tag(p.kind).expect("a kind of an outcome of the classifier"));
            }
        }
        let tags = ChunkTags { tagset: self.model.tagset().clone(), tags: chunk_tags, prob: Some(probs) };
        let chunks = PhraseChunks { spans, tokens: ranges, kind, kinds: self.model.kinds().clone() };
        Ok((tags, chunks))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    /// FNV-1a over the beam size, the one setting a `ChunkerME` is built
    /// with.
    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new().u64(self.model.beam_size() as u64).finish()
    }
}
