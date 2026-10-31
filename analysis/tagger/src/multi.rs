//! One encoder pass, many heads (#1517): [`MultiTask`] reads each window
//! of a document's words through the bundle's encoder once, pools the
//! states every head asks for from that one read
//! (`WordEncoder::token_features_pooled`), and each output layer is
//! decoded from them by its own head on the host. The layers it writes
//! are a tuple of types, as for any annotator: `MultiTask<(PosTags,
//! Dependencies<Tokens>, Lemmas)>` writes those three, each from the head
//! named for it or the one head of the bundle that writes its layer.
//! The heads decode at once, one thread each, once the encoder is done.
//!
//! Each layer is written exactly as the single-head annotator writes it
//! ([`crate::PosTagger`], [`crate::EntityTagger`],
//! [`crate::DependencyParser`], [`crate::Lemmatizer`] without a
//! dictionary, [`crate::SentenceTagger`], [`crate::WordTagger`]), from the
//! same features: one read and several reads of the same rows give the
//! same states. When [`Sentences`] is among the outputs and no prior
//! sentences are given, its [`SentenceBoundary`] head decodes boundaries
//! from the whole-view pass and cuts the other heads' sequences from those
//! same features.

use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::{BiaffineParser, EditTreeLemmatizer, SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, LicenseClass, Scheme, WordPooling};
use turbo::status::Error as CoreError;
use turbo_analysis::annotate::Writes;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Layer, ModelRef, Refs, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Subwords, Tokens, Words};

use crate::annotate::{HeadRunner, decode_with, entity_layer, find_head, forms_of, model_ref, pos_layer, refuse};
use crate::lemma::{lemma_layer, lemmas_with};
use crate::parse::{dep_layer, parse_with};
use crate::segment::{sentence_layer, sentence_ranges, split_with, word_layer};
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
    fn hidden(&self) -> Vec<usize> {
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
    fn poolings(&self) -> Vec<WordPooling> {
        let own = self.spec().word_pooling;
        match self {
            Head::Lemma { pos: Some(t), .. } => vec![own, t.spec().word_pooling],
            _ => vec![own],
        }
    }
}

/// One document's words as every head reads them: the sequences of
/// token indices, their text, and the features of each pooling read.
pub struct Input<'a> {
    pub view: &'a View,
    pub tokens: &'a Tokens,
    pub seqs: Vec<Vec<usize>>,
    pub forms: Vec<Vec<&'a str>>,
    poolings: Vec<WordPooling>,
    features: Vec<Vec<Vec<f32>>>,
    whole_seqs: Vec<Vec<usize>>,
    whole_forms: Vec<Vec<&'a str>>,
    whole_features: Vec<Vec<Vec<f32>>>,
}

impl Input<'_> {
    /// Each sequence's features under `pooling`.
    pub fn features(&self, pooling: WordPooling) -> Result<&[Vec<f32>], String> {
        let p = self.poolings.iter().position(|&q| q == pooling).ok_or("a pooling no head asked for")?;
        Ok(&self.features[p])
    }

    /// The whole-view sequence's features under `pooling`, for a sentence
    /// boundary head.
    pub fn whole_features(&self, pooling: WordPooling) -> Result<&[Vec<f32>], String> {
        let p = self.poolings.iter().position(|&q| q == pooling).ok_or("a pooling no head asked for")?;
        Ok(&self.whole_features[p])
    }
}

/// A layer a head writes: the head kind that writes it, how that head is
/// opened, and how the layer is made from the shared features.
pub trait HeadLayer: Layer {
    const KIND: HeadKind;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError>;
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String>;
}

fn wrong_head<T>() -> Result<T, String> {
    Err("the head is of another kind than its layer".into())
}

impl HeadLayer for PosTags {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        if spec.scheme != Scheme::Tags {
            return Err(refuse(format!("head {:?}: a part-of-speech head reads SCHEME_TAGS labels", spec.name)));
        }
        let tagger = TokenTagger::load(bundle, &spec.name)?;
        let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
        Ok(Head::Tagger { set: TagSet::new(&spec.name, &labels), tagger })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.features(tagger.spec().word_pooling)?;
        let decoded = decode_with(tagger, f, &input.forms).map_err(|e| e.to_string())?;
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
        let f = input.features(tagger.spec().word_pooling)?;
        let decoded = decode_with(tagger, f, &input.forms).map_err(|e| e.to_string())?;
        entity_layer(tagger, input.view, input.tokens, &input.seqs, &decoded, set)
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
        let f = input.features(parser.spec().word_pooling)?;
        let parses = parse_with(parser, f).map_err(|e| e.to_string())?;
        dep_layer(input.tokens, &input.seqs, &parses, relations)
    }
}

impl HeadLayer for Lemmas {
    const KIND: HeadKind = HeadKind::EditTreeLemmatizer;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let lemmatizer = EditTreeLemmatizer::load(bundle, &spec.name)?;
        let pos = match &spec.pos_input {
            Some(p) => Some(TokenTagger::load(bundle, &p.head)?),
            None => None,
        };
        Ok(Head::Lemma { lemmatizer, pos })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Lemma { lemmatizer, pos } = head else { return wrong_head() };
        let own = lemmatizer.spec().word_pooling;
        let f = input.features(own)?;
        let pf = match pos {
            Some(t) if t.spec().word_pooling != own => Some(input.features(t.spec().word_pooling)?),
            _ => None,
        };
        let lemmas = lemmas_with(lemmatizer, pos.as_ref(), None, f, pf, &input.forms).map_err(|e| e.to_string())?;
        lemma_layer(input.tokens, &input.seqs, lemmas)
    }
}

impl HeadLayer for Sentences {
    const KIND: HeadKind = HeadKind::SentenceBoundary;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let boundary = SentenceBoundary::load(bundle, &spec.name)?;
        Ok(Head::Sentence { boundary })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Sentence { boundary } = head else { return wrong_head() };
        let f = input.whole_features(boundary.spec().word_pooling)?;
        let ranges = sentence_ranges(boundary, f, &input.whole_forms).map_err(|e| e.to_string())?;
        let seq = input.whole_seqs.first().map_or(&[][..], Vec::as_slice);
        Ok(sentence_layer(input.view, input.tokens, seq, &ranges))
    }
}

impl HeadLayer for Words {
    const KIND: HeadKind = HeadKind::TokenSplit;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let splitter = TokenSplitter::load(bundle, &spec.name)?;
        Ok(Head::Split { splitter })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Split { splitter } = head else { return wrong_head() };
        let f = input.features(splitter.spec().word_pooling)?;
        let split = split_with(splitter, f, &input.forms).map_err(|e| e.to_string())?;
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
    /// The heads of the bundle at `dir` for each output of `W`, `names[i]`
    /// naming output `i`'s head or None for the one head that writes its
    /// layer. One head is not two outputs.
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
        let words = WordEncoder::open(dir, o)?;
        for h in &heads {
            if let Some(&n) = h.hidden().iter().find(|&&n| n != words.hidden()) {
                return Err(refuse(format!(
                    "head {:?} reads {n} values a word, the encoder gives {}",
                    h.spec().name,
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

    /// Every output layer over the tokens of the view, the pieces from
    /// `subwords` when the encoder reads them (`WordEncoder::reads`).
    pub fn layers(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<W, String> {
        let Some(boundary) = self.heads.iter().find_map(|h| match h {
            Head::Sentence { boundary } => Some(boundary),
            _ => None,
        }) else {
            let seqs = HeadRunner::sequences(view, tokens, sentences)?;
            let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
            let features = words.token_features_pooled(view, tokens, &seqs, subwords, &self.poolings)?;
            drop(words);
            let forms = forms_of(view, tokens, &seqs);
            let input = Input {
                view,
                tokens,
                seqs,
                forms,
                poolings: self.poolings.clone(),
                features,
                whole_seqs: Vec::new(),
                whole_forms: Vec::new(),
                whole_features: Vec::new(),
            };
            return W::write(&self.heads, &input);
        };
        let given = (sentences.is_some() || tokens.sentence.is_some())
            .then(|| HeadRunner::sequences(view, tokens, sentences))
            .transpose()?;
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let whole_seqs = HeadRunner::sequences(view, &all, None)?;
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let hidden = words.hidden();
        let whole_features = words.token_features_pooled(view, &all, &whole_seqs, subwords, &self.poolings)?;
        drop(words);
        let whole_forms = forms_of(view, &all, &whole_seqs);
        let ranges = match given {
            Some(seqs) => seqs.iter().map(|s| s[0]..s[s.len() - 1] + 1).collect(),
            None => {
                let p = self
                    .poolings
                    .iter()
                    .position(|&q| q == boundary.spec().word_pooling)
                    .ok_or("a pooling no head asked for")?;
                sentence_ranges(boundary, &whole_features[p], &whole_forms).map_err(|e| e.to_string())?
            }
        };
        let seqs = ranges.iter().map(|r| whole_seqs[0][r.clone()].to_vec()).collect();
        let forms = ranges.iter().map(|r| whole_forms[0][r.clone()].to_vec()).collect();
        let features = whole_features
            .iter()
            .map(|wf| ranges.iter().map(|r| wf[0][r.start * hidden..r.end * hidden].to_vec()).collect())
            .collect();
        let input = Input {
            view,
            tokens,
            seqs,
            forms,
            poolings: self.poolings.clone(),
            features,
            whole_seqs,
            whole_forms,
            whole_features,
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
        h.u64(self.width as u64).finish()
    }
}
