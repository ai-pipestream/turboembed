//! `POSTaggerME.tag`: the best tag sequence of each sentence by beam
//! search, and the annotator that writes it as `PosTags`.

use std::sync::{Arc, Mutex};

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Tag, Tokens};
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
        let model = &self.model;
        let search = BeamSearch::new(model.beam_size(), model.maxent());
        let Some(best) = search.best_sequence(words, context, &model.validator()) else {
            return false;
        };
        tags.extend(best.outcomes().iter().map(|o| model.tag_of(o).expect("an outcome of the classifier")));
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
        let mut from = 0;
        while from < n {
            let to = match &tokens.sentence {
                Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
                None => n,
            };
            words.clear();
            words.extend((from..to).map(|i| &text[starts[i] as usize..ends[i] as usize]));
            if !self.tag_with(&words, &mut context, &mut tags, &mut probs) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("no tag sequence of tokens {from}..{to} scored above the minimum"),
                });
            }
            from = to;
        }
        self.contexts.lock().unwrap_or_else(|e| e.into_inner()).push(context);
        Ok((PosTags { tagset: self.model.tagset().clone(), tags, prob: Some(probs) },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    /// FNV-1a over the tag format asked for, the beam size and the context
    /// cache size, the three settings a `POSTaggerME` is built with.
    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            for b in x.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        mix(match self.model.requested_format() {
            turbo_sequence::postag::PosTagFormat::Ud => 1,
            turbo_sequence::postag::PosTagFormat::Penn => 2,
            turbo_sequence::postag::PosTagFormat::Custom => 3,
            turbo_sequence::postag::PosTagFormat::Unknown => 4,
        });
        mix(self.model.beam_size() as u64);
        mix(self.model.context_cache_size() as u64);
        h
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Ix, Key, Layer, Span, Spans, Status, ViewId};
    use turbo_analysis_layers::Sentences;
    use turbo_sequence::postag::PosTagFormat;

    use super::*;

    /// The tagger over the English model, or `None` when it is not
    /// downloaded.
    fn english() -> Option<PosTagger> {
        Some(PosTagger::new(PosModel::read(english_path()?).unwrap()))
    }

    fn english_path() -> Option<std::path::PathBuf> {
        crate::test_models::published(crate::test_models::ENGLISH)
    }

    fn labels(t: &PosTagger, words: &[&str]) -> Vec<String> {
        let (mut tags, mut probs) = (Vec::new(), Vec::new());
        assert!(t.tag(words, &mut tags, &mut probs).unwrap());
        assert_eq!(tags.len(), probs.len());
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        tags.iter().map(|&x| t.pos_model().tagset().label(x).to_string()).collect()
    }

    #[test]
    fn tags_a_sentence() {
        let Some(t) = english() else { return };
        assert_eq!(labels(&t, &["The", "cat", "sat", "."]), ["DET", "NOUN", "VERB", "PUNCT"]);
        assert_eq!(labels(&t, &[]), Vec::<String>::new());
    }

    #[test]
    fn penn_tags_are_the_converted_outcomes() {
        let Some(path) = english_path() else { return };
        let t = PosTagger::new(PosModel::read(path).unwrap().with_format(PosTagFormat::Penn));
        assert_eq!(labels(&t, &["The", "cat", "sat", "."]), ["DT", "NN", "VB", "."]);
    }

    /// Tokens over `text` at the given byte spans, each in sentence `s`.
    fn tokens(view: ViewId, spans: &[(u32, u32, u32)]) -> Tokens {
        let mut out = Spans::new(view);
        for &(a, b, _) in spans {
            out.push(Span::new(a, b));
        }
        Tokens { spans: out, sentence: Some(spans.iter().map(|&(_, _, s)| Ix::new(s)).collect()) }
    }

    #[test]
    fn sentences_are_tagged_one_at_a_time() {
        let Some(t) = english() else { return };
        let text = "I run . Dogs run .";
        let doc = Document::new(text);
        let view = doc.get::<View>().unwrap();
        let two = tokens(view.id(), &[(0, 1, 0), (2, 5, 0), (6, 7, 0), (8, 12, 1), (13, 16, 1), (17, 18, 1)]);
        let (pos,) = t.annotate((view, &two)).unwrap();
        assert_eq!(pos.tags.len(), 6);
        let (mut tags, mut probs) = (Vec::new(), Vec::new());
        t.tag(&["I", "run", "."], &mut tags, &mut probs).unwrap();
        t.tag(&["Dogs", "run", "."], &mut tags, &mut probs).unwrap();
        assert_eq!(pos.tags, tags);
        assert_eq!(pos.prob.as_deref().unwrap(), &probs[..]);
        // Without a sentence column the six tokens are one sentence, and
        // the second "run" sees the first.
        let mut one = two;
        one.sentence = None;
        let (pos,) = t.annotate((view, &one)).unwrap();
        tags.clear();
        probs.clear();
        t.tag(&["I", "run", ".", "Dogs", "run", "."], &mut tags, &mut probs).unwrap();
        assert_eq!(pos.tags, tags);
        assert_eq!(pos.prob.as_deref().unwrap(), &probs[..]);
    }

    #[test]
    fn tokens_in_another_view_are_refused() {
        let doc = Document::new("ab");
        let Some(t) = english() else { return };
        let e = t.annotate((doc.get::<View>().unwrap(), &tokens(ViewId(3), &[(0, 1, 0)]))).err().unwrap();
        assert_eq!(
            e,
            Error::Annotator {
                annotator: "turbo-postag:maxent",
                detail: "the tokens are in view 3, not in view 0 that is tagged".into()
            }
        );
    }

    #[test]
    fn in_a_pipeline_after_given_sentences_and_tokens() {
        let text = "Birds sing .";
        let mut doc = Document::new(text);
        let view = doc.get::<View>().unwrap().id();
        let mut s = Spans::new(view);
        s.push(Span::new(0, 12));
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: s }, turbo_analysis::Provenance::by("test")).unwrap();
        let toks = tokens(view, &[(0, 5, 0), (6, 10, 0), (11, 12, 0)]);
        doc.add(&Key::<Tokens>::DEFAULT, toks, turbo_analysis::Provenance::by("test")).unwrap();
        let Some(tagger) = english() else { return };
        let config = tagger.config_hash();
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(tagger)
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let key = Key::<PosTags>::DEFAULT;
        assert_eq!(doc.status(key.id()), Some(&Status::Complete));
        let pos = doc.get::<PosTags>().unwrap();
        let got: Vec<&str> = pos.tags.iter().map(|&t| pos.tagset.label(t)).collect();
        assert_eq!(got, ["NOUN", "VERB", "PUNCT"]);
        let p = doc.provenance(key.id()).unwrap();
        assert_eq!(p.annotator, "turbo-postag:maxent");
        assert_eq!(p.config, config);
        assert_eq!(p.model.as_ref().unwrap().id, "opennlp:POSTaggerME:en");
        assert_eq!(p.reads, [Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()]);
        assert_eq!(pos.len(), 3);
    }

    #[test]
    fn the_configuration_hash_tells_settings_apart() {
        let Some(path) = english_path() else { return };
        let open = || PosModel::read(&path).unwrap();
        let a = PosTagger::new(open()).config_hash();
        let b = PosTagger::new(open().with_format(PosTagFormat::Penn)).config_hash();
        let c = PosTagger::new(open().with_beam_size(1).unwrap()).config_hash();
        let d = PosTagger::new(open().with_context_cache(0)).config_hash();
        assert_eq!(a, PosTagger::new(open()).config_hash());
        assert!(a != b && a != c && a != d && b != c && b != d && c != d);
    }
}
