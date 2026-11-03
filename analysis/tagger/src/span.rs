//! Typed, possibly nested spans from a HEAD_SPAN_CLASSIFIER head:
//! [`SpanHead`] decodes the entity layer from a sequence's word features,
//! however they were read, and [`SpanTagger`] is the annotator that reads
//! them through the bundle's encoder itself, a sentence at a time as the
//! token taggers read them.
//!
//! Every span runs from its first word's start to its last word's end, so
//! its offsets are the tokens' own.

use std::path::Path;
use std::sync::Arc;

use turbo::bundle::Bundle;
use turbo::head::SpanClassifier;
use turbo::head::span::SpanWork;
use turbo::head::tagger::Work;
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Ix, IxRange, Layer as _, ModelRef, Refs, Span, Spans, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tag, Tokens};

use crate::annotate::{HeadRunner, find_head, forms_of};
use crate::segment::Reader;

/// A span classifier head and the kinds its entity layer names.
pub struct SpanHead {
    head: SpanClassifier,
    kinds: Arc<TagSet>,
    /// Each label's kind index; None for `O`.
    kind_of: Vec<Option<u16>>,
}

impl SpanHead {
    /// The span classifier head `spec` of `bundle`.
    pub fn load(bundle: &Bundle, spec: &HeadSpec) -> Result<SpanHead, CoreError> {
        let head = SpanClassifier::load(bundle, &spec.name)?;
        let mut kind_of = Vec::new();
        let mut types: Vec<&str> = Vec::new();
        for l in head.labels() {
            kind_of.push((l != "O").then(|| {
                types.push(l);
                (types.len() - 1) as u16
            }));
        }
        let kinds = TagSet::new(&spec.name, &types);
        Ok(SpanHead { head, kinds, kind_of })
    }

    pub fn head(&self) -> &SpanClassifier {
        &self.head
    }

    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// The spans the head keeps in each sequence of `seqs` over the view's
    /// tokens, `features[i]` sequence `i`'s word features under the head's
    /// pooling and taps, in order of sequence, first word, last; each
    /// span's score is its label's softmax probability among the span's
    /// own scores.
    pub fn decode(
        &self,
        view: &View,
        tokens: &Tokens,
        seqs: &[Vec<usize>],
        features: &[Vec<f32>],
    ) -> Result<Entities, String> {
        if features.len() != seqs.len() {
            return Err(format!("{} feature rows for {} sequences", features.len(), seqs.len()));
        }
        let forms = forms_of(view, tokens, seqs);
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
        for ((seq, f), forms) in seqs.iter().zip(features).zip(&forms) {
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

/// Writes typed spans from a HEAD_SPAN_CLASSIFIER head into the entity
/// layer, which holds nested and overlapping spans.
pub struct SpanTagger {
    head: SpanHead,
    read: Reader,
}

impl SpanTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the entity layer's.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<SpanTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Entities::ID, HeadKind::SpanClassifier)?;
        let head = SpanHead::load(&bundle, spec)?;
        let read = Reader::open(dir, &bundle, spec, head.head.hidden(), o)?;
        Ok(SpanTagger { head, read })
    }

    pub fn head(&self) -> &SpanClassifier {
        &self.head.head
    }

    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.head.kinds
    }

    /// Rows the encoder has read so far.
    pub fn rows(&self) -> u64 {
        self.read.rows()
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
        let features = self.read.features(view, tokens, &seqs, subwords, self.head.head.spec())?;
        self.head.decode(view, tokens, &seqs, &features)
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
        self.read.config(self.head.head.spec())
    }
}
