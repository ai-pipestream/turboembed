//! The one encoder read behind a request's head steps (`steps/heads.rs`).
//!
//! [`HeadGroup`] holds the request's heads and one encoder, and reads the
//! document once, pooling the states every head asks for
//! (`WordEncoder::token_features_pooled`); each step decodes from that
//! read. The read's windows follow one plan:
//!
//! - With no whole-view head among the steps, the view's tokens are read
//!   a sentence at a time (`HeadRunner::sequences` over the document's
//!   sentences), as each tagger crate annotator reads them on its own.
//! - With a learned sentence head or a coreference head, the view's
//!   tokens are read once as one sequence (in windows of whole words when
//!   longer than a row, words.rs). Those heads decode from it; every
//!   other head decodes from its slice for each sentence, as `MultiTask`
//!   slices it under a sentence head.
//!
//! Either way a step adds no encoder read: [`HeadGroup::rows`] counts the
//! rows, and later steps over the same view and tokens reuse the cache.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::run::tail_taps;
use turbo::manifest::{HeadSpec, WordPooling};
use turbo::status::Error as CoreError;
use turbo_analysis::{Execution, Fnv1a, ModelRef, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{
    Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Subwords, Tokens, Words,
};
use turbo_tagger::annotate::{HeadRunner, find_head, open_words};
use turbo_tagger::multi::{Head, HeadLayer, Input};
use turbo_tagger::{CorefHead, RelationHead, SpanHead, WordEncoder};

use super::HeadStep;
use crate::api::TURBO_E_INVALID_ARGUMENT;

/// A head of the group, as its step decodes it.
pub(crate) enum GroupHead {
    /// A head a tagger crate layer writes from shared features.
    Layer(Box<Head>),
    Span(Box<SpanHead>),
    Relation(Box<RelationHead>),
    Coref(Box<CorefHead>),
}

/// Each pooling's features of each sequence read.
type Features = Arc<Vec<Vec<Vec<f32>>>>;

impl GroupHead {
    fn spec(&self) -> &HeadSpec {
        match self {
            GroupHead::Layer(h) => h.spec(),
            GroupHead::Span(h) => h.head().spec(),
            GroupHead::Relation(h) => h.head().spec(),
            GroupHead::Coref(h) => h.head().spec(),
        }
    }

    fn hidden(&self) -> Vec<usize> {
        match self {
            GroupHead::Layer(h) => h.hidden(),
            GroupHead::Span(h) => vec![h.head().hidden()],
            GroupHead::Relation(h) => vec![h.head().hidden()],
            GroupHead::Coref(h) => vec![h.head().hidden()],
        }
    }

    fn poolings(&self) -> Vec<WordPooling> {
        match self {
            GroupHead::Layer(h) => h.poolings(),
            _ => vec![self.spec().word_pooling],
        }
    }

    pub(crate) fn taps(&self) -> usize {
        match self {
            GroupHead::Layer(h) => h.taps().into_iter().max().unwrap_or(1),
            _ => self.spec().layer_mix.map_or(1, |m| m.layers as usize),
        }
    }
}

fn open_head(step: HeadStep, bundle: &Bundle, spec: &HeadSpec) -> Result<GroupHead, CoreError> {
    Ok(match step {
        HeadStep::Pos => GroupHead::Layer(Box::new(PosTags::open(bundle, spec)?)),
        HeadStep::Morphology => GroupHead::Layer(Box::new(MorphFeatures::<Tokens>::open(bundle, spec)?)),
        HeadStep::Entities => GroupHead::Layer(Box::new(Entities::open(bundle, spec)?)),
        HeadStep::Dependencies => GroupHead::Layer(Box::new(Dependencies::<Tokens>::open(bundle, spec)?)),
        HeadStep::Lemmas => GroupHead::Layer(Box::new(Lemmas::open(bundle, spec)?)),
        HeadStep::Sentences => GroupHead::Layer(Box::new(Sentences::open(bundle, spec)?)),
        HeadStep::Words => GroupHead::Layer(Box::new(Words::open(bundle, spec)?)),
        HeadStep::Spans => GroupHead::Span(Box::new(SpanHead::load(bundle, spec)?)),
        HeadStep::Relations => GroupHead::Relation(Box::new(RelationHead::load(bundle, spec)?)),
        HeadStep::Coref => GroupHead::Coref(Box::new(CorefHead::load(bundle, spec)?)),
    })
}

/// One read of a document's words, and what it was read from.
struct Cached {
    key: u64,
    seqs: Vec<Vec<usize>>,
    features: Features,
}

/// The heads of one request and the encoder read they share.
pub struct HeadGroup {
    words: Mutex<WordEncoder>,
    heads: Vec<(HeadStep, GroupHead)>,
    poolings: Vec<WordPooling>,
    /// Whether the read is of the whole view as one sequence.
    whole: bool,
    taps: usize,
    hidden: usize,
    cache: Mutex<Option<Cached>>,
    pub(crate) model: ModelRef,
    pub(crate) execution: Execution,
    pub(crate) width: usize,
}

impl HeadGroup {
    /// The heads of the bundle at `dir` for `steps`, each the one head
    /// the step takes, and the bundle's encoder.
    pub fn open(dir: &Path, steps: &[HeadStep], o: &EncoderOptions) -> Result<Arc<HeadGroup>, CoreError> {
        let bundle = Bundle::open(dir)?;
        let mut heads = Vec::with_capacity(steps.len());
        for &s in steps {
            let spec = s
                .resolve(&bundle, None)
                .map_err(|e| CoreError::new(TURBO_E_INVALID_ARGUMENT, e.failure("the group", "").message))?;
            heads.push((s, spec.name.clone()));
        }
        HeadGroup::open_resolved(dir, &heads, o)
    }

    /// The heads of the bundle at `dir` named for each step, and the
    /// bundle's encoder.
    pub fn open_resolved(
        dir: &Path,
        steps: &[(HeadStep, String)],
        o: &EncoderOptions,
    ) -> Result<Arc<HeadGroup>, CoreError> {
        let bundle = Bundle::open(dir)?;
        let refuse = |m: String| CoreError::new(TURBO_E_INVALID_ARGUMENT, m);
        let mut heads = Vec::with_capacity(steps.len());
        for (s, name) in steps {
            let spec = find_head(&bundle, Some(name), s.layer(), s.kind())?;
            heads.push((*s, open_head(*s, &bundle, spec)?));
        }
        let taps = heads.iter().map(|(_, h)| h.taps()).max().unwrap_or(1);
        let words = open_words(dir, o, taps)?;
        for (_, h) in &heads {
            if let Some(&n) = h.hidden().iter().find(|&&n| n != words.hidden()) {
                return Err(refuse(format!(
                    "head {:?} reads {n} values a word, the encoder gives {}",
                    h.spec().name,
                    words.hidden()
                )));
            }
        }
        let mut poolings: Vec<WordPooling> = Vec::new();
        for p in heads.iter().flat_map(|(_, h)| h.poolings()) {
            if !poolings.contains(&p) {
                poolings.push(p);
            }
        }
        let m = &bundle.manifest.model;
        Ok(Arc::new(HeadGroup {
            model: ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() },
            execution: words.execution().clone(),
            width: words.width(),
            whole: heads.iter().any(|(s, _)| s.whole_view()),
            taps: words.taps(),
            hidden: words.hidden(),
            words: Mutex::new(words),
            heads,
            poolings,
            cache: Mutex::new(None),
        }))
    }

    /// Rows the encoder has read: one a window, however many steps
    /// decoded from it.
    pub fn rows(&self) -> u64 {
        self.words.lock().unwrap_or_else(PoisonError::into_inner).rows()
    }

    /// The steps' heads, in the order opened.
    pub fn steps(&self) -> Vec<HeadStep> {
        self.heads.iter().map(|(s, _)| *s).collect()
    }

    /// Whether the group reads the whole view as one sequence.
    pub fn reads_whole_view(&self) -> bool {
        self.whole
    }

    pub(crate) fn head(&self, step: HeadStep) -> Result<&GroupHead, String> {
        self.heads
            .iter()
            .find(|(s, _)| *s == step)
            .map(|(_, h)| h)
            .ok_or_else(|| format!("{} was not opened with this group", step.name()))
    }

    /// The read the plan makes of `tokens` (of `sentences`' sequences, or
    /// of the whole view), from the cache when it was made before.
    fn read(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<(Vec<Vec<usize>>, Features), String> {
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let (read, seqs) = if self.whole {
            (&all, HeadRunner::sequences(view, &all, None)?)
        } else {
            (tokens, HeadRunner::sequences(view, tokens, sentences)?)
        };
        let (starts, ends) = tokens.spans.columns();
        let mut h = Fnv1a::new();
        h.str(view.text()).u64(view.id().0 as u64);
        for &x in starts.iter().chain(ends) {
            h.u64(x as u64);
        }
        let key = h.finish();
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(c) = &*cache
            && c.key == key
            && c.seqs == seqs
        {
            return Ok((seqs, Arc::clone(&c.features)));
        }
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let f = Arc::new(words.token_features_pooled(view, read, &seqs, subwords, &self.poolings)?);
        *cache = Some(Cached { key, seqs: seqs.clone(), features: Arc::clone(&f) });
        Ok((seqs, f))
    }

    /// The words of each sentence (`HeadRunner::sequences`) as every
    /// sentence-scoped head reads them: the read itself, or its slices
    /// for each sentence when the read is of the whole view.
    pub(crate) fn input<'a>(
        &self,
        view: &'a View,
        tokens: &'a Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Input<'a>, String> {
        let (read, features) = self.read(view, tokens, sentences, subwords)?;
        if !self.whole {
            return Ok(Input::new(view, tokens, read, self.poolings.clone(), features).with_taps(self.taps));
        }
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let mut ranges = Vec::with_capacity(seqs.len());
        for s in &seqs {
            let (Some(&a), Some(&b)) = (s.first(), s.last()) else { return Err("an empty sentence".into()) };
            if b + 1 - a != s.len() || read.first().is_none_or(|w| b >= w.len()) {
                return Err("a sentence whose tokens are not a run of the view's".into());
            }
            ranges.push(a..b + 1);
        }
        let (taps, hidden) = (self.taps, self.hidden);
        let sliced: Vec<Vec<Vec<f32>>> = features
            .iter()
            .map(|per| {
                ranges
                    .iter()
                    .map(|r| {
                        let whole = &per[0];
                        let stride = whole.len() / taps;
                        (0..taps)
                            .flat_map(|t| {
                                whole[t * stride + r.start * hidden..t * stride + r.end * hidden].iter().copied()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Ok(Input::new(view, tokens, seqs, self.poolings.clone(), Arc::new(sliced)).with_taps(taps))
    }

    /// The whole view's tokens as one sequence, as a whole-view head
    /// reads them; the group must read the whole view.
    pub(crate) fn whole<R>(
        &self,
        view: &View,
        tokens: &Tokens,
        subwords: Option<&Subwords>,
        decode: impl FnOnce(&Input<'_>) -> Result<R, String>,
    ) -> Result<R, String> {
        if !self.whole {
            return Err("the group reads a sentence at a time".into());
        }
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let (seqs, features) = self.read(view, tokens, None, subwords)?;
        let input = Input::new(view, &all, seqs, self.poolings.clone(), features).with_taps(self.taps);
        decode(&input)
    }

    /// The features of `input` the head of `step` reads: its pooling,
    /// its last layer taps.
    pub(crate) fn features_for(&self, step: HeadStep, input: &Input<'_>) -> Result<Vec<Vec<f32>>, String> {
        let h = self.head(step)?;
        let f = input.features(h.spec().word_pooling)?;
        Ok(tail_taps(f, self.taps, h.taps()).into_owned())
    }
}
