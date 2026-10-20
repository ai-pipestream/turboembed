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
    let n = tokens.spans.columns().0.len();
    let mut ranges = Vec::new();
    let mut from = 0;
    while from < n {
        let to = match &tokens.sentence {
            Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
            None => n,
        };
        ranges.push((from, to));
        from = to;
    }
    ranges
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
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in (self.model.beam_size() as u64).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Status, ViewId};
    use turbo_analysis_layers::{Sentences, TagSet};

    use super::*;

    fn fixture() -> PhraseChunker {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/chunker-fixture.bin");
        PhraseChunker::new(ChunkerModel::read(path).unwrap())
    }

    fn tagged<'a>(words: &[(&'a str, &'a str)]) -> Vec<TokenTag<'a>> {
        words.iter().map(|&(token, tag)| TokenTag { token, tag }).collect()
    }

    fn labels(c: &PhraseChunker, words: &[(&str, &str)]) -> Vec<String> {
        let (mut tags, mut probs) = (Vec::new(), Vec::new());
        assert!(c.chunk(&tagged(words), &mut tags, &mut probs));
        assert_eq!(tags.len(), probs.len());
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        tags.iter().map(|&t| c.chunker_model().tagset().label(t).to_string()).collect()
    }

    #[test]
    fn chunks_a_sentence_it_was_trained_on() {
        let c = fixture();
        let words =
            [("The", "DT"), ("cat", "NN"), ("sat", "VBD"), ("on", "IN"), ("the", "DT"), ("mat", "NN"), (".", ".")];
        assert_eq!(labels(&c, &words), ["B-NP", "I-NP", "B-VP", "B-PP", "B-NP", "I-NP", "O"]);
        assert_eq!(labels(&c, &[]), Vec::<String>::new());
        // The outcomes are the labels of the tags.
        let (mut outcomes, mut probs) = (Vec::new(), Vec::new());
        assert!(c.outcomes(&tagged(&words), &mut outcomes, &mut probs));
        assert_eq!(outcomes, ["B-NP", "I-NP", "B-VP", "B-PP", "B-NP", "I-NP", "O"]);
        assert_eq!(probs.len(), 7);
    }

    /// Tokens over `text` at the given byte spans, each in sentence `s`.
    fn tokens(view: ViewId, spans: &[(u32, u32, u32)]) -> Tokens {
        let mut out = Spans::new(view);
        for &(a, b, _) in spans {
            out.push(Span::new(a, b));
        }
        Tokens { spans: out, sentence: Some(spans.iter().map(|&(_, _, s)| Ix::new(s)).collect()) }
    }

    fn penn(labels: &[&str]) -> PosTags {
        let tagset = TagSet::new("penn", &["DT", "NN", "VBD", "IN", "."]);
        PosTags { tagset: tagset.clone(), tags: labels.iter().map(|l| tagset.tag(l).unwrap()).collect(), prob: None }
    }

    #[test]
    fn the_annotator_writes_tags_and_phrases_per_sentence() {
        let text = "The cat sat on the mat . Dogs bark .";
        let mut doc = Document::new(text);
        let view = doc.get::<View>().unwrap().id();
        let mut sentences = Spans::new(view);
        sentences.push(Span::new(0, 24));
        sentences.push(Span::new(25, 36));
        let toks = tokens(
            view,
            &[
                (0, 3, 0),
                (4, 7, 0),
                (8, 11, 0),
                (12, 14, 0),
                (15, 18, 0),
                (19, 22, 0),
                (23, 24, 0),
                (25, 29, 1),
                (30, 34, 1),
                (35, 36, 1),
            ],
        );
        let tags = penn(&["DT", "NN", "VBD", "IN", "DT", "NN", ".", "NN", "VBD", "."]);
        let by = || Provenance::by("test");
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentences }, by()).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, toks, by()).unwrap();
        doc.add(&Key::<PosTags>::DEFAULT, tags, by()).unwrap();
        let chunker = fixture();
        let config = chunker.config_hash();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(chunker)
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let key = Key::<ChunkTags>::DEFAULT;
        assert_eq!(doc.status(key.id()), Some(&Status::Complete));
        let tags = doc.get::<ChunkTags>().unwrap();
        let got: Vec<&str> = tags.tags.iter().map(|&t| tags.tagset.label(t)).collect();
        assert_eq!(got[..7], ["B-NP", "I-NP", "B-VP", "B-PP", "B-NP", "I-NP", "O"]);
        assert_eq!(got.len(), 10);
        assert_eq!(tags.prob.as_ref().unwrap().len(), 10);
        let chunks = doc.get::<PhraseChunks>().unwrap();
        let (start, end) = chunks.spans.columns();
        let shown: Vec<(&str, &str, (usize, usize))> = (0..chunks.spans.len())
            .map(|i| {
                let r = &chunks.tokens[i];
                (
                    &text[start[i] as usize..end[i] as usize],
                    chunks.kinds.label(chunks.kind[i]),
                    (r.start.get(), r.end.get()),
                )
            })
            .collect();
        assert_eq!(
            shown[..4],
            [("The cat", "NP", (0, 2)), ("sat", "VP", (2, 3)), ("on", "PP", (3, 4)), ("the mat", "NP", (4, 6))]
        );
        // The second sentence's phrases index its own tokens.
        assert!(shown[4..].iter().all(|&(_, _, (s, _))| s >= 7), "{shown:?}");
        let p = doc.provenance(key.id()).unwrap();
        assert_eq!(p.annotator, "turbo-phrase-chunker:maxent");
        assert_eq!(p.config, config);
        assert_eq!(p.model.as_ref().unwrap().id, "opennlp:ChunkerME:en");
        assert_eq!(p.reads, [Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id(), Key::<PosTags>::DEFAULT.id()]);
        let p = doc.provenance(Key::<PhraseChunks>::DEFAULT.id()).unwrap();
        assert_eq!(p.annotator, "turbo-phrase-chunker:maxent");
    }

    #[test]
    fn sentences_are_chunked_one_at_a_time() {
        let c = fixture();
        let text = "The cat . The cat .";
        let doc = Document::new(text);
        let view = doc.get::<View>().unwrap();
        let two = tokens(view.id(), &[(0, 3, 0), (4, 7, 0), (8, 9, 0), (10, 13, 1), (14, 17, 1), (18, 19, 1)]);
        let tags = penn(&["DT", "NN", ".", "DT", "NN", "."]);
        let (got, _) = c.annotate((view, &two, &tags)).unwrap();
        let (mut want, mut probs) = (Vec::new(), Vec::new());
        c.chunk(&tagged(&[("The", "DT"), ("cat", "NN"), (".", ".")]), &mut want, &mut probs);
        c.chunk(&tagged(&[("The", "DT"), ("cat", "NN"), (".", ".")]), &mut want, &mut probs);
        assert_eq!(got.tags, want);
        assert_eq!(got.prob.as_deref().unwrap(), &probs[..]);
        // Without a sentence column the six tokens are one sentence.
        let mut one = two;
        one.sentence = None;
        let (got, _) = c.annotate((view, &one, &tags)).unwrap();
        want.clear();
        probs.clear();
        let all = [("The", "DT"), ("cat", "NN"), (".", "."), ("The", "DT"), ("cat", "NN"), (".", ".")];
        c.chunk(&tagged(&all), &mut want, &mut probs);
        assert_eq!(got.tags, want);
        assert_eq!(got.prob.as_deref().unwrap(), &probs[..]);
    }

    #[test]
    fn tokens_in_another_view_and_tag_counts_are_checked() {
        let doc = Document::new("ab cd");
        let view = doc.get::<View>().unwrap();
        let c = fixture();
        let e = c.annotate((view, &tokens(ViewId(3), &[(0, 2, 0)]), &penn(&["DT"]))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator {
                annotator: "turbo-phrase-chunker:maxent",
                detail: "the tokens are in view 3, not in view 0 that is chunked".into()
            }
        );
        let e = c.annotate((view, &tokens(view.id(), &[(0, 2, 0), (3, 5, 0)]), &penn(&["DT"]))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator { annotator: "turbo-phrase-chunker:maxent", detail: "1 tags for 2 tokens".into() }
        );
    }

    #[test]
    fn malformed_token_spans_sentence_column_and_tag_indices_are_rejected() {
        let doc = Document::new("The \u{00E9} cat");
        let view = doc.get::<View>().unwrap();
        let c = fixture();

        // Sentence column shorter or longer than tokens.
        let mut short_sent = tokens(view.id(), &[(0, 3, 0), (7, 10, 0)]);
        short_sent.sentence = Some(vec![Ix::new(0)]);
        assert!(matches!(c.annotate((view, &short_sent, &penn(&["DT", "NN"]))), Err(Error::Annotator { .. })));
        let mut long_sent = tokens(view.id(), &[(0, 3, 0)]);
        long_sent.sentence = Some(vec![Ix::new(0), Ix::new(0)]);
        assert!(matches!(c.annotate((view, &long_sent, &penn(&["DT"]))), Err(Error::Annotator { .. })));

        // Out-of-bounds POS tag index.
        let bad_tags = PosTags { tagset: TagSet::new("penn", &["DT"]), tags: vec![Tag(5)], prob: None };
        assert!(matches!(
            c.annotate((view, &tokens(view.id(), &[(0, 3, 0)]), &bad_tags)),
            Err(Error::Annotator { .. })
        ));

        // Inverted, out-of-bounds, and mid-codepoint UTF-8 token spans.
        for (s, e) in [(5, 3), (0, 99), (4, 5)] {
            let mut spans = Spans::new(view.id());
            spans.push(Span { start: s, end: e });
            let bad_span = Tokens { spans, sentence: None };
            assert!(matches!(c.annotate((view, &bad_span, &penn(&["DT"]))), Err(Error::Annotator { .. })));
        }

        // Out-of-order tokens within a sentence.
        let unsorted = tokens(view.id(), &[(7, 10, 0), (0, 3, 0)]);
        assert!(matches!(c.annotate((view, &unsorted, &penn(&["DT", "NN"]))), Err(Error::Annotator { .. })));
    }
}
