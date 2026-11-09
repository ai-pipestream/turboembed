//! Annotators over a bundle's segmenting heads: [`SentenceTagger`]
//! writes the sentence layer from a HEAD_SENTENCE_BOUNDARY head, reading
//! the whole view's tokens as one sequence of words (in windows when it
//! is longer than a row, words.rs); [`WordTagger`] writes the syntactic
//! words layer from a HEAD_TOKEN_SPLIT head (`_`, substring splits and
//! fusional splits). [`SpanTagger`] (span.rs) shares their encoder path.
//!
//! Every sentence runs from its first word's start to its last word's
//! end, so its offsets are the tokens' own.

use std::ops::Range;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::boundary::sentences;
use turbo::head::tagger::Work;
use turbo::head::{SentenceBoundary, TokenSplitter};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, Layer as _, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Sentences, Subwords, Tokens, Words};

pub use crate::span::SpanTagger;

use crate::annotate::{HeadRunner, find_head, forms_of, model_ref, open_words, refuse};
use crate::multi::Input;
use crate::words::WordEncoder;

/// The encoder a segmenting head reads, and what its annotator names.
pub(crate) struct Reader {
    words: Mutex<WordEncoder>,
    pub(crate) model: ModelRef,
    pub(crate) execution: Execution,
    width: usize,
}

impl Reader {
    pub(crate) fn open(
        dir: &Path,
        bundle: &Bundle,
        spec: &HeadSpec,
        hidden: usize,
        o: &EncoderOptions,
    ) -> Result<Reader, CoreError> {
        let words = open_words(dir, o, spec.layer_mix.map_or(1, |m| m.layers as usize))?;
        if words.hidden() != hidden {
            return Err(refuse(format!("the head reads {hidden} values a word, the encoder gives {}", words.hidden())));
        }
        Ok(Reader {
            execution: words.execution().clone(),
            width: words.width(),
            model: model_ref(bundle),
            words: Mutex::new(words),
        })
    }

    pub(crate) fn features(
        &self,
        view: &View,
        tokens: &Tokens,
        seqs: &[Vec<usize>],
        subwords: Option<&Subwords>,
        spec: &HeadSpec,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        words.token_features(view, tokens, seqs, subwords, spec.word_pooling)
    }

    pub(crate) fn rows(&self) -> u64 {
        self.words.lock().unwrap_or_else(PoisonError::into_inner).rows()
    }

    pub(crate) fn config(&self, spec: &HeadSpec) -> u64 {
        Fnv1a::new().str(&spec.name).u64(self.width as u64).finish()
    }
}

/// Writes the sentence layer from a HEAD_SENTENCE_BOUNDARY head.
pub struct SentenceTagger {
    head: SentenceBoundary,
    read: Reader,
}

impl SentenceTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the sentence layer's.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<SentenceTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Sentences::ID, HeadKind::SentenceBoundary)?;
        let head = SentenceBoundary::load(&bundle, &spec.name)?;
        let read = Reader::open(dir, &bundle, spec, head.hidden(), o)?;
        Ok(SentenceTagger { head, read })
    }

    pub fn head(&self) -> &SentenceBoundary {
        &self.head
    }

    /// Rows the encoder has read so far.
    pub fn rows(&self) -> u64 {
        self.read.rows()
    }

    /// The sentences of the view's tokens: each from its first token's
    /// start to its last token's end. The tokens are one sequence,
    /// whatever sentence column they carry.
    pub fn sentences(&self, view: &View, tokens: &Tokens, subwords: Option<&Subwords>) -> Result<Sentences, String> {
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let seqs = HeadRunner::sequences(view, &all, None)?;
        let Some(seq) = seqs.first() else {
            return Ok(Sentences { spans: Spans::new(view.id()) });
        };
        let features = self.read.features(view, &all, &seqs, subwords, self.head.spec())?;
        let forms = forms_of(view, &all, &seqs);
        let ranges = sentence_ranges(&self.head, &features, &forms).map_err(|e| e.to_string())?;
        Ok(sentence_layer(view, tokens, seq, &ranges))
    }
}

impl Annotator for SentenceTagger {
    const ID: &'static str = "turbo-tagger-sentences";
    type Reads = (View, Tokens, Option<Subwords>);
    type Writes = (Sentences,);

    fn annotate(&self, (view, tokens, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.sentences(view, tokens, subwords)
            .map(|s| (s,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.read.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.read.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.read.config(self.head.spec())
    }
}

/// The sentence word ranges `head` decodes from the whole-view sequence.
pub(crate) fn sentence_ranges(
    head: &SentenceBoundary,
    features: &[Vec<f32>],
    forms: &[Vec<&str>],
) -> Result<Vec<Range<usize>>, CoreError> {
    let Some(f) = features.first() else { return Ok(Vec::new()) };
    let words_text = forms.first().map_or(&[][..], Vec::as_slice);
    let n = words_text.len();
    let words: Vec<u32> = (0..n as u32).collect();
    let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), Work::default());
    let reads = head.reads_forms().then_some(words_text);
    head.scores(f, &words, reads, &mut scores, &mut work)?;
    head.decode(&scores, n, &mut ends)?;
    sentences(&ends)
}

/// The sentence layer from the decoded word ranges of `seq`.
fn sentence_layer(view: &View, tokens: &Tokens, seq: &[usize], ranges: &[Range<usize>]) -> Sentences {
    let mut out = Sentences { spans: Spans::new(view.id()) };
    for r in ranges {
        let start = tokens.spans.get(Ix::new(seq[r.start] as u32)).start;
        let end = tokens.spans.get(Ix::new(seq[r.end - 1] as u32)).end;
        out.spans.push(Span::new(start, end));
    }
    out
}

/// The sentence layer `boundary` decodes from `input`, whose one
/// sequence is the whole view's tokens (a whole-view read, as
/// [`SentenceTagger`] makes it): each sentence from its first token's
/// start to its last token's end.
pub fn decode_sentences(boundary: &SentenceBoundary, input: &Input<'_>) -> Result<Sentences, String> {
    let Some(seq) = input.seqs.first() else {
        return Ok(Sentences { spans: Spans::new(input.view.id()) });
    };
    if input.seqs.len() != 1 {
        return Err(format!("a sentence head reads one whole-view sequence, not {}", input.seqs.len()));
    }
    let taps = boundary.spec().layer_mix.map_or(1, |m| m.layers as usize);
    let features = input.tapped(boundary.spec().word_pooling, taps)?;
    let ranges = sentence_ranges(boundary, &features, &input.forms).map_err(|e| e.to_string())?;
    Ok(sentence_layer(input.view, input.tokens, seq, &ranges))
}

/// Writes the syntactic words layer from a HEAD_TOKEN_SPLIT head.
pub struct WordTagger {
    head: TokenSplitter,
    read: Reader,
}

impl WordTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the words layer's.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<WordTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Words::ID, HeadKind::TokenSplit)?;
        let head = TokenSplitter::load(&bundle, &spec.name)?;
        let read = Reader::open(dir, &bundle, spec, head.hidden(), o)?;
        Ok(WordTagger { head, read })
    }

    pub fn head(&self) -> &TokenSplitter {
        &self.head
    }

    /// Rows the encoder has read so far.
    pub fn rows(&self) -> u64 {
        self.read.rows()
    }

    /// Each token's syntactic words over the view's tokens: a 1-to-1 word
    /// when `_` keeps the token whole, byte subspans when the split words
    /// tile the token's bytes, or explicit forms for fusional contractions.
    pub fn words(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Words, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let features = self.read.features(view, tokens, &seqs, subwords, self.head.spec())?;
        let forms = forms_of(view, tokens, &seqs);
        let split = split_with(&self.head, &features, &forms, None).map_err(|e| e.to_string())?;
        word_layer(view, tokens, &seqs, split)
    }
}

impl Annotator for WordTagger {
    const ID: &'static str = "turbo-tagger-words";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Words,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.words(view, tokens, sentences, subwords)
            .map(|w| (w,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.read.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.read.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.read.config(self.head.spec())
    }
}

/// Each sequence's decoded word lists from `splitter`.
pub(crate) fn split_with(
    splitter: &TokenSplitter,
    features: &[Vec<f32>],
    forms: &[Vec<&str>],
    language: Option<&str>,
) -> Result<Vec<Vec<Vec<String>>>, CoreError> {
    let taps = splitter.taps();
    let h = splitter.hidden();
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), Work::default());
    let mut out = Vec::with_capacity(features.len());
    for (f, words) in features.iter().zip(forms) {
        let n = words.len();
        let word_stride = n.checked_mul(h).filter(|&s| s == 0 || f.len().is_multiple_of(s));
        let Some(ws) = word_stride else {
            return Err(refuse(format!("{} forms for {} feature values", words.len(), f.len())));
        };
        let have = f.len().checked_div(ws).unwrap_or(taps);
        let f = if have > taps && ws > 0 { &f[(have - taps) * ws..] } else { f.as_slice() };
        let ids: Vec<u32> = (0..n as u32).collect();
        let reads = splitter.reads_forms().then_some(words.as_slice());
        splitter.scores_language(f, &ids, reads, language, &mut scores, &mut work)?;
        splitter.decode(&scores, words, &mut labels)?;
        out.push(splitter.words(&labels, words)?);
    }
    Ok(out)
}

/// The syntactic words layer from each sequence's decoded words.
pub(crate) fn word_layer(
    view: &View,
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    split: Vec<Vec<Vec<String>>>,
) -> Result<Words, String> {
    let mut of: Vec<Option<Vec<String>>> = vec![None; tokens.spans.len()];
    for (s, words) in seqs.iter().zip(split) {
        for (&i, ws) in s.iter().zip(words) {
            of[i] = Some(ws);
        }
    }
    let split = of.into_iter().collect::<Option<Vec<_>>>().ok_or("a token lies outside every sentence")?;
    Words::from_splits(view.text(), tokens, &split)
}
