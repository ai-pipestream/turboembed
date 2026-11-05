//! One encoder pass, many heads (#1517): [`MultiTask`] reads each window of a
//! document's words through the bundle's encoder once, pools the states every
//! head asks for from that one read (`WordEncoder::token_features_pooled`),
//! and decodes each output layer on the host.
//!
//! Without a sentence head (or when prior sentences are supplied), each layer
//! matches its single-head annotator ([`crate::PosTagger`],
//! [`crate::EntityTagger`], [`crate::DependencyParser`], [`crate::Lemmatizer`],
//! [`crate::WordTagger`]). When [`Sentences`] is among the outputs and no prior
//! sentences are given, the encoder runs once over the whole view, each
//! [`Sentences`] output decodes its own [`SentenceBoundary`] head, and the
//! other heads' sequences are sliced by the first [`SentenceBoundary`] head.

use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::{BiaffineParser, EditTreeLemmatizer, SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadSpec, LicenseClass, WordPooling};
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, ModelRef, Refs, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Sentences, Subwords, Tokens};

use crate::annotate::{HeadRunner, forms_of, model_ref, open_words, refuse, tail_taps};
use crate::segment::sentence_ranges;
use crate::words::WordEncoder;

/// A loaded head and the label set its layer names.
pub enum Head {
    /// A token tagger, `set` its tags or, under BIO and BILOU, its kinds.
    Tagger {
        tagger: TokenTagger,
        set: Arc<TagSet>,
    },
    Parser {
        parser: BiaffineParser,
        relations: Arc<TagSet>,
    },
    /// A lemmatizer and the tagger its `pos_input` names.
    Lemma {
        lemmatizer: EditTreeLemmatizer,
        pos: Option<TokenTagger>,
    },
    /// A sentence boundary head.
    Sentence {
        boundary: SentenceBoundary,
    },
    /// A token split head writing syntactic words.
    Split {
        splitter: TokenSplitter,
    },
}

fn spec_taps(spec: &HeadSpec) -> usize {
    spec.layer_mix.map_or(1, |m| m.layers as usize)
}

fn pooling_of(poolings: &[WordPooling], pooling: WordPooling) -> Result<usize, String> {
    poolings.iter().position(|&q| q == pooling).ok_or_else(|| "a pooling no head asked for".into())
}

fn slice_tapped(seq: &[f32], taps: usize, hidden: usize, r: &std::ops::Range<usize>) -> Vec<f32> {
    let stride = seq.len() / taps;
    (0..taps).flat_map(|t| seq[t * stride + r.start * hidden..t * stride + r.end * hidden].iter().copied()).collect()
}

impl Head {
    pub fn spec(&self) -> &HeadSpec {
        match self {
            Head::Tagger { tagger, .. } => tagger.spec(),
            Head::Parser { parser, .. } => parser.spec(),
            Head::Lemma { lemmatizer, .. } => lemmatizer.spec(),
            Head::Sentence { boundary } => boundary.spec(),
            Head::Split { splitter } => splitter.spec(),
        }
    }

    /// The values a word each head reads, its own and its tagger's.
    pub fn hidden(&self) -> Vec<usize> {
        match self {
            Head::Tagger { tagger, .. } => vec![tagger.hidden()],
            Head::Parser { parser, .. } => vec![parser.hidden()],
            Head::Lemma { lemmatizer, pos } => {
                std::iter::once(lemmatizer.hidden()).chain(pos.as_ref().map(TokenTagger::hidden)).collect()
            }
            Head::Sentence { boundary } => vec![boundary.hidden()],
            Head::Split { splitter } => vec![splitter.hidden()],
        }
    }

    /// The poolings the head reads, its tagger's included.
    pub fn poolings(&self) -> Vec<WordPooling> {
        let own = self.spec().word_pooling;
        match self {
            Head::Lemma { pos: Some(t), .. } => vec![own, t.spec().word_pooling],
            _ => vec![own],
        }
    }

    /// The layer taps the head reads, its tagger's included.
    pub fn taps(&self) -> Vec<usize> {
        match self {
            Head::Tagger { tagger, .. } => vec![tagger.taps()],
            Head::Parser { parser, .. } => vec![parser.taps()],
            Head::Lemma { lemmatizer, pos } => {
                std::iter::once(lemmatizer.taps()).chain(pos.as_ref().map(TokenTagger::taps)).collect()
            }
            Head::Sentence { boundary } => vec![spec_taps(boundary.spec())],
            Head::Split { splitter } => vec![spec_taps(splitter.spec())],
        }
    }
}

/// Whole-view sequences, forms, and pooled features for whole-view passes.
struct Whole<'a> {
    seqs: Vec<Vec<usize>>,
    forms: Vec<Vec<&'a str>>,
    features: Vec<Vec<Vec<f32>>>,
}

/// One document's words as every head reads them: token sequences, forms, and pooled features.
pub struct Input<'a> {
    pub view: &'a View,
    pub tokens: &'a Tokens,
    pub seqs: Vec<Vec<usize>>,
    pub forms: Vec<Vec<&'a str>>,
    poolings: Vec<WordPooling>,
    features: Arc<Vec<Vec<Vec<f32>>>>,
    taps: usize,
    language: Option<&'a str>,
    whole: Option<Whole<'a>>,
}

impl<'a> Input<'a> {
    /// The words of `seqs` over `tokens`, `features[p]` their features under `poolings[p]`.
    pub fn new(
        view: &'a View,
        tokens: &'a Tokens,
        seqs: Vec<Vec<usize>>,
        poolings: Vec<WordPooling>,
        features: Arc<Vec<Vec<Vec<f32>>>>,
    ) -> Input<'a> {
        let forms = forms_of(view, tokens, &seqs);
        Input { view, tokens, seqs, forms, poolings, features, taps: 1, language: None, whole: None }
    }

    /// Sets the number of encoder layer taps present in `features`.
    pub fn with_taps(mut self, taps: usize) -> Input<'a> {
        self.taps = taps;
        self
    }

    /// Each sequence's features under `pooling`.
    pub fn features(&self, pooling: WordPooling) -> Result<&[Vec<f32>], String> {
        Ok(&self.features[pooling_of(&self.poolings, pooling)?])
    }

    pub fn tapped(&self, pooling: WordPooling, taps: usize) -> Result<std::borrow::Cow<'_, [Vec<f32>]>, String> {
        Ok(tail_taps(self.features(pooling)?, self.taps, taps))
    }
}

mod layers;
pub use layers::*;

pub(crate) fn wrong_head<T>() -> Result<T, String> {
    Err("the head is of another kind than its layer".into())
}

fn language_of<'a>(spec: &HeadSpec, input: &Input<'a>) -> Option<&'a str> {
    spec.language.as_ref().and(input.language)
}

/// Writes the layers `W` from one encoder pass, each from its own head.
pub struct MultiTask<W: Outputs> {
    words: Mutex<WordEncoder>,
    heads: Vec<Head>,
    poolings: Vec<WordPooling>,
    model: ModelRef,
    execution: Execution,
    width: usize,
    tier: Option<LicenseClass>,
    language: Option<String>,
    out: PhantomData<fn() -> W>,
}

impl<W: Outputs> MultiTask<W> {
    /// The heads of the bundle at `dir` for each output of `W`.
    pub fn open(dir: &Path, names: &[Option<&str>], o: &EncoderOptions) -> Result<MultiTask<W>, CoreError> {
        Self::open_with_language(dir, names, o, None)
    }

    /// As [`open`](Self::open), selecting a language for every head.
    pub fn open_language(
        dir: &Path,
        names: &[Option<&str>],
        o: &EncoderOptions,
        language: &str,
    ) -> Result<MultiTask<W>, CoreError> {
        Self::open_with_language(dir, names, o, Some(language.to_owned()))
    }

    fn open_with_language(
        dir: &Path,
        names: &[Option<&str>],
        o: &EncoderOptions,
        language: Option<String>,
    ) -> Result<MultiTask<W>, CoreError> {
        if names.len() != W::N {
            return Err(refuse(format!("{} head names for {} outputs", names.len(), W::N)));
        }
        let bundle = Bundle::open(dir)?;
        let heads = W::open(&bundle, names)?;
        for (i, h) in heads.iter().enumerate() {
            if heads[..i].iter().any(|o| o.spec().name == h.spec().name) {
                return Err(refuse(format!("head {:?} is named for two outputs", h.spec().name)));
            }
        }
        if let Some(selected) = language.as_deref() {
            for head in heads.iter().filter(|head| head.spec().language.is_some()) {
                let input = head.spec().language.as_ref().expect("filtered language input");
                if !input.languages.iter().any(|known| known == selected) {
                    return Err(refuse(format!(
                        "unknown language {selected:?} for head {:?}: one of {}",
                        head.spec().name,
                        input.languages.join(", ")
                    )));
                }
            }
        }
        let taps = heads.iter().flat_map(Head::taps).max().unwrap_or(1);
        let words = open_words(dir, o, taps)?;
        for h in &heads {
            if let Some(&n) = h.hidden().iter().find(|&&n| n != words.hidden()) {
                let name = &h.spec().name;
                return Err(refuse(format!(
                    "head {name:?} reads {n} values a word, the encoder gives {}",
                    words.hidden()
                )));
            }
        }
        let mut poolings: Vec<WordPooling> = Vec::new();
        for p in heads.iter().flat_map(Head::poolings) {
            if !poolings.contains(&p) {
                poolings.push(p);
            }
        }
        Ok(MultiTask {
            model: model_ref(&bundle),
            execution: words.execution().clone(),
            width: words.width(),
            tier: bundle.manifest.license_tier(),
            words: Mutex::new(words),
            heads,
            poolings,
            language,
            out: PhantomData,
        })
    }

    /// Each output's head, in output order.
    pub fn heads(&self) -> &[Head] {
        &self.heads
    }

    /// The bundle's licence tier (`Manifest::license_tier`): the strictest
    /// class of its encoder and of every head's training corpora.
    pub fn license_tier(&self) -> Option<LicenseClass> {
        self.tier
    }

    /// Rows the encoder has read so far.
    pub fn rows(&self) -> u64 {
        self.words.lock().unwrap_or_else(PoisonError::into_inner).rows()
    }

    /// Every output layer over the tokens of the view (`WordEncoder::reads`).
    pub fn layers(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<W, String> {
        let boundary = if sentences.is_none() && tokens.sentence.is_none() {
            self.heads.iter().find_map(|h| if let Head::Sentence { boundary } = h { Some(boundary) } else { None })
        } else {
            None
        };
        let Some(boundary) = boundary else {
            let seqs = HeadRunner::sequences(view, tokens, sentences)?;
            let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
            let (taps, features) =
                (words.taps(), words.token_features_pooled(view, tokens, &seqs, subwords, &self.poolings)?);
            drop(words);
            let input = Input {
                language: self.language.as_deref(),
                ..Input::new(view, tokens, seqs, self.poolings.clone(), Arc::new(features)).with_taps(taps)
            };
            return W::write(&self.heads, &input);
        };
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let whole_seqs = HeadRunner::sequences(view, &all, None)?;
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let (hidden, taps) = (words.hidden(), words.taps());
        let whole_features = words.token_features_pooled(view, &all, &whole_seqs, subwords, &self.poolings)?;
        drop(words);
        let whole_forms = forms_of(view, &all, &whole_seqs);
        let p = pooling_of(&self.poolings, boundary.spec().word_pooling)?;
        let bf = tail_taps(&whole_features[p], taps, spec_taps(boundary.spec()));
        let ranges = sentence_ranges(boundary, &bf, &whole_forms).map_err(|e| e.to_string())?;
        let seqs = ranges.iter().map(|r| whole_seqs[0][r.clone()].to_vec()).collect();
        let forms = ranges.iter().map(|r| whole_forms[0][r.clone()].to_vec()).collect();
        let features = Arc::new(
            whole_features
                .iter()
                .map(|wf| ranges.iter().map(|r| slice_tapped(&wf[0], taps, hidden, r)).collect())
                .collect(),
        );
        let whole = Some(Whole { seqs: whole_seqs, forms: whole_forms, features: whole_features });
        let input = Input {
            view,
            tokens,
            seqs,
            forms,
            poolings: self.poolings.clone(),
            features,
            taps,
            language: self.language.as_deref(),
            whole,
        };
        W::write(&self.heads, &input)
    }
}

impl<W: Outputs> Annotator for MultiTask<W> {
    const ID: &'static str = "turbo-tagger-multitask";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = W;

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<W, Error> {
        self.layers(view, tokens, sentences, subwords)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        for head in &self.heads {
            h.str(&head.spec().name);
        }
        if let Some(language) = &self.language {
            h.str(language);
        }
        h.u64(self.width as u64).finish()
    }
}
