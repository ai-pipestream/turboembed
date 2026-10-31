//! Annotators over a bundle's segmenting heads: [`SentenceTagger`]
//! writes the sentence layer from a HEAD_SENTENCE_BOUNDARY head, reading
//! the whole view's tokens as one sequence of words (in windows when it
//! is longer than a row, words.rs); [`WordTagger`] writes the syntactic
//! words layer from a HEAD_TOKEN_SPLIT head (`_`, substring splits and
//! fusional splits); [`SpanTagger`] writes typed, possibly nested spans
//! from a HEAD_SPAN_CLASSIFIER head into the entity layer, a sentence at
//! a time as the token taggers read them.
//!
//! Every sentence and span runs from its first word's start to its last
//! word's end, so its offsets are the tokens' own.

use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::boundary::sentences;
use turbo::head::span::SpanWork;
use turbo::head::tagger::Work;
use turbo::head::{SentenceBoundary, SpanClassifier, TokenSplitter};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::Error as CoreError;
use turbo_analysis::{
    Annotator, Error, Execution, Fnv1a, Ix, IxRange, Layer as _, ModelRef, Refs, Span, Spans, TagSet, View,
};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tag, Tokens, Words};

use crate::annotate::{HeadRunner, find_head, forms_of, model_ref, refuse};
use crate::words::WordEncoder;

/// The encoder a segmenting head reads, and what its annotator names.
pub(crate) struct Reader {
    words: Mutex<WordEncoder>,
    pub(crate) model: ModelRef,
    pub(crate) execution: Execution,
    width: usize,
}

impl Reader {
    pub(crate) fn open(dir: &Path, bundle: &Bundle, hidden: usize, o: &EncoderOptions) -> Result<Reader, CoreError> {
        let words = WordEncoder::open(dir, o)?;
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
        let read = Reader::open(dir, &bundle, head.hidden(), o)?;
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
        let read = Reader::open(dir, &bundle, head.hidden(), o)?;
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
        let split = split_with(&self.head, &features, &forms).map_err(|e| e.to_string())?;
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
) -> Result<Vec<Vec<Vec<String>>>, CoreError> {
    let hidden = splitter.hidden();
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), Work::default());
    let mut out = Vec::with_capacity(features.len());
    for (f, words) in features.iter().zip(forms) {
        let n = f.len() / hidden;
        if words.len() != n {
            return Err(refuse(format!("{} forms for {n} words", words.len())));
        }
        let ids: Vec<u32> = (0..n as u32).collect();
        let reads = splitter.reads_forms().then_some(words.as_slice());
        splitter.scores(f, &ids, reads, &mut scores, &mut work)?;
        splitter.decode(&scores, words, &mut labels)?;
        out.push(splitter.words(&labels, words)?);
    }
    Ok(out)
}

/// Byte sub-spans when `ws` is two or more non-empty pieces that tile
/// `surface` from start to end.
fn subspans_of(surface: &str, span: Span, ws: &[String]) -> Option<Vec<Span>> {
    if ws.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(ws.len());
    let mut off = 0usize;
    for w in ws {
        if w.is_empty() || !surface[off..].starts_with(w.as_str()) {
            return None;
        }
        let next = off + w.len();
        out.push(Span::new(span.start + off as u32, span.start + next as u32));
        off = next;
    }
    (off == surface.len()).then_some(out)
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
    let text = view.text();
    let mut out = Words::new();
    for (i, entry) in of.into_iter().enumerate() {
        let ws = entry.ok_or("a token lies outside every sentence")?;
        let t = Ix::<Tokens>::new(i as u32);
        let span = tokens.spans.get(t);
        let surface = &text[span.range()];
        let range = IxRange::new(i as u32, i as u32 + 1);
        if ws.len() == 1 && ws[0] == surface {
            out.push_token(t);
        } else if let Some(subs) = subspans_of(surface, span, &ws) {
            for sub in subs {
                out.push_word(range, None, Some(sub));
            }
        } else {
            for w in &ws {
                out.push_word(range, Some(w), None);
            }
        }
    }
    out.check_against(text, tokens)?;
    Ok(out)
}

/// Writes typed spans from a HEAD_SPAN_CLASSIFIER head into the entity
/// layer, which holds nested and overlapping spans; each span's score is
/// its label's softmax probability among the span's own scores.
pub struct SpanTagger {
    head: SpanClassifier,
    kinds: Arc<TagSet>,
    /// Each label's kind index; None for `O`.
    kind_of: Vec<Option<u16>>,
    read: Reader,
}

impl SpanTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the entity layer's.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<SpanTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Entities::ID, HeadKind::SpanClassifier)?;
        let head = SpanClassifier::load(&bundle, &spec.name)?;
        let read = Reader::open(dir, &bundle, head.hidden(), o)?;
        let mut kind_of = Vec::new();
        let mut types: Vec<&str> = Vec::new();
        for l in head.labels() {
            kind_of.push((l != "O").then(|| {
                types.push(l);
                (types.len() - 1) as u16
            }));
        }
        let kinds = TagSet::new(&spec.name, &types);
        Ok(SpanTagger { head, kinds, kind_of, read })
    }

    pub fn head(&self) -> &SpanClassifier {
        &self.head
    }

    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// The spans the head keeps in each sequence of the view's tokens
    /// (`HeadRunner::sequences`), in order of sequence, first word, last.
    pub fn entities(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Entities, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let features = self.read.features(view, tokens, &seqs, subwords, self.head.spec())?;
        let forms = forms_of(view, tokens, &seqs);
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let (mut scores, mut kept, mut w, mut sw) = (Vec::new(), Vec::new(), Work::default(), SpanWork::default());
        let labels = self.head.labels().len();
        let max = self.head.spec().span.map_or(0, |s| s.max_words as usize);
        for ((seq, f), forms) in seqs.iter().zip(&features).zip(&forms) {
            let n = seq.len();
            let words: Vec<u32> = (0..n as u32).collect();
            let reads = self.head.reads_forms().then_some(forms.as_slice());
            self.head.scores(f, &words, reads, &mut scores, &mut w, &mut sw).map_err(|e| e.to_string())?;
            self.head.decode(&scores, n, &mut kept).map_err(|e| e.to_string())?;
            for s in &kept {
                let (i, j) = (s.first as usize, s.last as usize);
                // The span's row: the spans of every earlier first word,
                // then this word's spans up to its last.
                let row = (0..i).map(|a| max.min(n - a)).sum::<usize>() + (j - i);
                let r = &scores[row * labels..][..labels];
                let m = r.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
                let z: f64 = r.iter().map(|&v| (v as f64 - m).exp()).sum();
                let kind = self.kind_of[s.label as usize].ok_or("a kept span labelled O")?;
                let (a, b) = (seq[i] as u32, seq[j] as u32);
                out.spans.push(Span::new(tokens.spans.get(Ix::new(a)).start, tokens.spans.get(Ix::new(b)).end));
                out.tokens.push(IxRange::new(a, b + 1));
                out.kind.push(Tag(kind));
                out.score.push((r[s.label as usize] as f64 - m).exp() / z);
            }
        }
        out.check()?;
        Ok(out)
    }
}

impl Annotator for SpanTagger {
    const ID: &'static str = "turbo-tagger-spans";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.entities(view, tokens, sentences, subwords)
            .map(|e| (e,))
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
