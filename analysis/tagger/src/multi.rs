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
use turbo::manifest::{HeadKind, HeadSpec, LicenseClass, Scheme, WordPooling};
use turbo::status::Error as CoreError;
use turbo_analysis::annotate::Writes;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, Layer, ModelRef, Refs, Span, Spans, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{
    Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Strs, Subwords, Tokens, Words,
};

use crate::annotate::{
    HeadRunner, decode_with, entity_layer, find_head, forms_of, model_ref, open_words, pos_layer, refuse, tail_taps,
};
use crate::lemma::{lemma_layer, lemmas_with};
use crate::parse::{dep_layer, parse_with};
use crate::segment::{sentence_ranges, split_with, word_layer};
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
        Input { view, tokens, seqs, forms, poolings, features, taps: 1, whole: None }
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

    fn tapped(&self, pooling: WordPooling, taps: usize) -> Result<std::borrow::Cow<'_, [Vec<f32>]>, String> {
        Ok(tail_taps(self.features(pooling)?, self.taps, taps))
    }
}

/// A layer a head writes: its head kind, how it is opened, and how it decodes shared features.
pub trait HeadLayer: Layer {
    const KIND: HeadKind;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError>;
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String>;
}

fn wrong_head<T>() -> Result<T, String> {
    Err("the head is of another kind than its layer".into())
}

fn open_tags(bundle: &Bundle, spec: &HeadSpec, role: &str) -> Result<Head, CoreError> {
    if spec.scheme != Scheme::Tags {
        return Err(refuse(format!("head {:?}: a {role} head reads SCHEME_TAGS labels", spec.name)));
    }
    let tagger = TokenTagger::load(bundle, &spec.name)?;
    let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
    Ok(Head::Tagger { set: TagSet::new(&spec.name, &labels), tagger })
}

impl HeadLayer for PosTags {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        open_tags(bundle, spec, "part-of-speech")
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded = decode_with(tagger, &f, &input.forms).map_err(|e| e.to_string())?;
        pos_layer(input.tokens, &input.seqs, &decoded, set)
    }
}

impl HeadLayer for Entities {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        if spec.scheme == Scheme::Tags {
            return Err(refuse(format!(
                "head {:?}: an entity head reads SCHEME_BIO or SCHEME_BILOU labels",
                spec.name
            )));
        }
        let tagger = TokenTagger::load(bundle, &spec.name)?;
        let types: Vec<&str> = tagger.types().iter().map(String::as_str).collect();
        Ok(Head::Tagger { set: TagSet::new(&spec.name, &types), tagger })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded = decode_with(tagger, &f, &input.forms).map_err(|e| e.to_string())?;
        entity_layer(tagger, input.view, input.tokens, &input.seqs, &decoded, set)
    }
}

/// The morphological features layer from a SCHEME_TAGS head.
impl HeadLayer for MorphFeatures<Tokens> {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        open_tags(bundle, spec, "morphology")
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded = decode_with(tagger, &f, &input.forms).map_err(|e| e.to_string())?;
        let tags = pos_layer(input.tokens, &input.seqs, &decoded, set)?;
        let mut feats = Strs::default();
        for &t in &tags.tags {
            feats.push(set.label(t));
        }
        Ok(MorphFeatures::new(feats))
    }
}

impl HeadLayer for Dependencies<Tokens> {
    const KIND: HeadKind = HeadKind::BiaffineParser;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let parser = BiaffineParser::load(bundle, &spec.name)?;
        let labels: Vec<&str> = parser.labels().iter().map(String::as_str).collect();
        Ok(Head::Parser { relations: TagSet::new(&spec.name, &labels), parser })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Parser { parser, relations } = head else { return wrong_head() };
        let f = input.tapped(parser.spec().word_pooling, parser.taps())?;
        let parses = parse_with(parser, &f).map_err(|e| e.to_string())?;
        dep_layer(input.tokens, &input.seqs, &parses, relations)
    }
}

impl HeadLayer for Lemmas {
    const KIND: HeadKind = HeadKind::EditTreeLemmatizer;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let lemmatizer = EditTreeLemmatizer::load(bundle, &spec.name)?;
        let pos = spec.pos_input.as_ref().map(|p| TokenTagger::load(bundle, &p.head)).transpose()?;
        Ok(Head::Lemma { lemmatizer, pos })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Lemma { lemmatizer, pos } = head else { return wrong_head() };
        let own = lemmatizer.spec().word_pooling;
        let f = input.tapped(own, lemmatizer.taps())?;
        let pf = match pos {
            Some(t) if t.spec().word_pooling != own || t.taps() != lemmatizer.taps() => {
                Some(input.tapped(t.spec().word_pooling, t.taps())?)
            }
            _ => None,
        };
        let lemmas =
            lemmas_with(lemmatizer, pos.as_ref(), None, &f, pf.as_deref(), &input.forms).map_err(|e| e.to_string())?;
        lemma_layer(input.tokens, &input.seqs, lemmas)
    }
}

impl HeadLayer for Sentences {
    const KIND: HeadKind = HeadKind::SentenceBoundary;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        Ok(Head::Sentence { boundary: SentenceBoundary::load(bundle, &spec.name)? })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Sentence { boundary } = head else { return wrong_head() };
        let mut out = Sentences { spans: Spans::new(input.view.id()) };
        let push = |out: &mut Sentences, s: &[usize]| {
            let (a, b) = (s[0] as u32, s[s.len() - 1] as u32);
            out.spans.push(Span::new(input.tokens.spans.get(Ix::new(a)).start, input.tokens.spans.get(Ix::new(b)).end));
        };
        if let Some(whole) = &input.whole {
            let p = pooling_of(&input.poolings, boundary.spec().word_pooling)?;
            let wf = tail_taps(&whole.features[p], input.taps, spec_taps(boundary.spec()));
            for r in sentence_ranges(boundary, &wf, &whole.forms).map_err(|e| e.to_string())? {
                push(&mut out, &whole.seqs[0][r]);
            }
            return Ok(out);
        }
        for s in &input.seqs {
            push(&mut out, s);
        }
        Ok(out)
    }
}

impl HeadLayer for Words {
    const KIND: HeadKind = HeadKind::TokenSplit;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        Ok(Head::Split { splitter: TokenSplitter::load(bundle, &spec.name)? })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Split { splitter } = head else { return wrong_head() };
        let f = input.tapped(splitter.spec().word_pooling, spec_taps(splitter.spec()))?;
        let split = split_with(splitter, &f, &input.forms).map_err(|e| e.to_string())?;
        word_layer(input.view, input.tokens, &input.seqs, split)
    }
}

/// The layers a [`MultiTask`] writes: a tuple of [`HeadLayer`]s, one head
/// each.
pub trait Outputs: Writes + Sized {
    /// Each output's head, `names[i]` naming output `i`'s or, when None,
    /// the one head of the bundle that writes its layer.
    fn open(bundle: &Bundle, names: &[Option<&str>]) -> Result<Vec<Head>, CoreError>;
    fn write(heads: &[Head], input: &Input<'_>) -> Result<Self, String>;
}

macro_rules! outputs {
    ($($L:ident $i:tt),+) => {
        impl<$($L: HeadLayer),+> Outputs for ($($L,)+) {
            fn open(bundle: &Bundle, names: &[Option<&str>]) -> Result<Vec<Head>, CoreError> {
                Ok(vec![$($L::open(bundle, find_head(bundle, names[$i], $L::ID, $L::KIND)?)?),+])
            }
            fn write(heads: &[Head], input: &Input<'_>) -> Result<Self, String> {
                // Each head decodes on its own thread; every layer is its
                // head's alone, so the result does not depend on the order.
                std::thread::scope(|s| {
                    let running = ($(s.spawn(|| $L::write(&heads[$i], input)),)+);
                    Ok(($(running.$i.join().map_err(|_| format!("head {:?} panicked", heads[$i].spec().name))??,)+))
                })
            }
        }
    };
}

outputs!(A 0);
outputs!(A 0, B 1);
outputs!(A 0, B 1, C 2);
outputs!(A 0, B 1, C 2, D 3);
outputs!(A 0, B 1, C 2, D 3, E 4);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

/// Writes the layers `W` from one encoder pass, each from its own head.
pub struct MultiTask<W: Outputs> {
    words: Mutex<WordEncoder>,
    heads: Vec<Head>,
    poolings: Vec<WordPooling>,
    model: ModelRef,
    execution: Execution,
    width: usize,
    tier: Option<LicenseClass>,
    out: PhantomData<fn() -> W>,
}

impl<W: Outputs> MultiTask<W> {
    /// The heads of the bundle at `dir` for each output of `W`.
    pub fn open(dir: &Path, names: &[Option<&str>], o: &EncoderOptions) -> Result<MultiTask<W>, CoreError> {
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
            let input = Input::new(view, tokens, seqs, self.poolings.clone(), Arc::new(features)).with_taps(taps);
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
        let input = Input { view, tokens, seqs, forms, poolings: self.poolings.clone(), features, taps, whole };
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
        h.u64(self.width as u64).finish()
    }
}
