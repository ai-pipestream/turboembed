//! The heads that segment and span: the sentence boundary head
//! (HEAD_SENTENCE_BOUNDARY), which finds the sentences of a whole text's
//! tokens read as one sequence; the token split head (HEAD_TOKEN_SPLIT),
//! which gives each token its syntactic words; and the entity head, a
//! token tagger over BIO or BILOU labels or a span classifier
//! (HEAD_SPAN_CLASSIFIER). Each decodes as the analysis annotators decode
//! it (`turbo-tagger`, segment.rs and span.rs), from the same encoder
//! read.

use std::ops::Range;
use std::sync::{Arc, PoisonError};

use turbo_analysis::TagSet;

use super::{BundleHeads, Entity, Loaded, Missing, kind_of, tagset};
use crate::bundle::Bundle;
use crate::head::boundary::sentences;
use crate::head::run::{decode_tagger, tail_taps};
use crate::head::span::SpanWork;
use crate::head::tagger::Work;
use crate::head::{SentenceBoundary, SpanClassifier, TokenSplitter, TokenTagger};
use crate::manifest::{HeadKind, HeadSpec, Scheme};
use crate::status::{BUNDLE_INVALID, Error, Result};

/// The bundle's entity head.
pub(crate) enum EntityHead {
    /// A token tagger over BIO or BILOU labels: an entity is a run of
    /// tags, its probability the mean of its words'.
    Tagger(TokenTagger),
    /// A span classifier: an entity is a kept span, its probability its
    /// label's softmax among the span's own scores. `kind_of` is each
    /// label's kind index, None for `O`.
    Spans { head: SpanClassifier, kind_of: Vec<Option<u32>> },
}

impl EntityHead {
    /// The entity head `spec` of `bundle` and the kinds it names, or why
    /// the analyzer does not run it.
    pub(super) fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Loaded<(EntityHead, Arc<TagSet>)>> {
        match spec.kind {
            HeadKind::TokenTagger => {
                if spec.scheme == Scheme::Tags {
                    return Err(Error::new(
                        BUNDLE_INVALID,
                        format!("head {:?}: an entity head reads SCHEME_BIO or SCHEME_BILOU labels", spec.name),
                    ));
                }
                let t = TokenTagger::load(bundle, &spec.name)?;
                let set = tagset(&spec.name, t.types());
                Ok(Ok((EntityHead::Tagger(t), set)))
            }
            HeadKind::SpanClassifier => {
                let head = SpanClassifier::load(bundle, &spec.name)?;
                let mut types: Vec<String> = Vec::new();
                let kind_of = head
                    .labels()
                    .iter()
                    .map(|l| {
                        (l != "O").then(|| {
                            types.push(l.clone());
                            types.len() as u32 - 1
                        })
                    })
                    .collect();
                let set = tagset(&spec.name, &types);
                Ok(Ok((EntityHead::Spans { head, kind_of }, set)))
            }
            kind => Ok(Err(Missing::Kind { head: spec.name.clone(), kind })),
        }
    }

    pub(super) fn spec(&self) -> &HeadSpec {
        match self {
            EntityHead::Tagger(t) => t.spec(),
            EntityHead::Spans { head, .. } => head.spec(),
        }
    }

    pub(super) fn hidden(&self) -> usize {
        match self {
            EntityHead::Tagger(t) => t.hidden(),
            EntityHead::Spans { head, .. } => head.hidden(),
        }
    }

    pub(super) fn taps(&self) -> usize {
        match self {
            EntityHead::Tagger(t) => t.taps(),
            EntityHead::Spans { head, .. } => head.taps(),
        }
    }

    /// The entities of each of `sequences`, `features[i]` sequence `i`'s
    /// word features at the head's pooling and taps, in order of first
    /// word then last.
    pub(super) fn decode(&self, features: &[Vec<f32>], sequences: &[Vec<&str>]) -> Result<Vec<Vec<Entity>>> {
        let mut all = Vec::with_capacity(sequences.len());
        match self {
            EntityHead::Tagger(t) => {
                for d in &decode_tagger(t, features, sequences)? {
                    let mut found = Vec::new();
                    for w in t.spans(&d.tags)? {
                        let p = &d.prob[w.start as usize..w.end as usize];
                        let prob = p.iter().sum::<f64>() / p.len() as f64;
                        found.push(Entity { start: w.start, end: w.end, kind: w.kind, prob });
                    }
                    all.push(found);
                }
            }
            EntityHead::Spans { head, kind_of } => {
                let (mut scores, mut kept, mut w, mut sw) =
                    (Vec::new(), Vec::new(), Work::default(), SpanWork::default());
                let labels = head.labels().len();
                let max = head.spec().span.map_or(0, |s| s.max_words as usize);
                for (f, forms) in features.iter().zip(sequences) {
                    let n = forms.len();
                    let words: Vec<u32> = (0..n as u32).collect();
                    let reads = head.reads_forms().then_some(forms.as_slice());
                    head.scores(f, &words, reads, &mut scores, &mut w, &mut sw)?;
                    head.decode(&scores, n, &mut kept)?;
                    let mut found = Vec::with_capacity(kept.len());
                    for s in &kept {
                        let (i, j) = (s.first as usize, s.last as usize);
                        // The span's row: the spans of every earlier first
                        // word, then this word's spans up to its last.
                        let row = (0..i).map(|a| max.min(n - a)).sum::<usize>() + (j - i);
                        let r = &scores[row * labels..][..labels];
                        let m = r.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
                        let z: f64 = r.iter().map(|&v| (v as f64 - m).exp()).sum();
                        let kind = kind_of[s.label as usize]
                            .ok_or_else(|| Error::new(BUNDLE_INVALID, "a kept span labelled O"))?;
                        let prob = (r[s.label as usize] as f64 - m).exp() / z;
                        found.push(Entity { start: s.first, end: s.last + 1, kind, prob });
                    }
                    all.push(found);
                }
            }
        }
        Ok(all)
    }
}

/// The sentence boundary head writing `spec`'s layer, refused when it is
/// of another kind: the sentences of every task come from it, so no rule
/// splitter stands in for a head the analyzer cannot run.
pub(super) fn open_sentences(bundle: &Bundle, spec: &HeadSpec) -> Result<SentenceBoundary> {
    if let Err(m) = kind_of(spec, HeadKind::SentenceBoundary) {
        return Err(super::unsupported("sentence detection", "sentences", Some(&m)));
    }
    SentenceBoundary::load(bundle, &spec.name)
}

/// The token split head writing `spec`'s layer, or why the analyzer does
/// not run it.
pub(super) fn open_split(bundle: &Bundle, spec: &HeadSpec) -> Result<Loaded<TokenSplitter>> {
    match kind_of(spec, HeadKind::TokenSplit) {
        Err(m) => Ok(Err(m)),
        Ok(()) => TokenSplitter::load(bundle, &spec.name).map(Ok),
    }
}

/// Each token's syntactic words in each of `sequences`, from its word
/// features at the split head's pooling and taps.
pub(super) fn split_sequences(
    splitter: &TokenSplitter,
    features: &[Vec<f32>],
    sequences: &[Vec<&str>],
) -> Result<Vec<Vec<Vec<String>>>> {
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), Work::default());
    let mut out = Vec::with_capacity(sequences.len());
    for (f, forms) in features.iter().zip(sequences) {
        let ids: Vec<u32> = (0..forms.len() as u32).collect();
        let reads = splitter.reads_forms().then_some(forms.as_slice());
        splitter.scores(f, &ids, reads, &mut scores, &mut work)?;
        splitter.decode(&scores, forms, &mut labels)?;
        out.push(splitter.words(&labels, forms)?);
    }
    Ok(out)
}

impl BundleHeads {
    /// Whether the entity head is a span classifier.
    pub(crate) fn entities_by_spans(&self) -> bool {
        matches!(self.entities, Ok((EntityHead::Spans { .. }, _)))
    }

    /// Whether the bundle finds sentences with a head rather than rules.
    pub(crate) fn has_sentences(&self) -> bool {
        self.sentences.is_some()
    }

    /// The sentences the bundle's sentence boundary head finds in a whole
    /// text's `tokens`, read as one sequence (in windows of whole words
    /// when it is longer than a row), as token ranges in order; None when
    /// the bundle has no such head.
    pub(crate) fn sentences(&self, tokens: &[&str]) -> Result<Option<Vec<Range<usize>>>> {
        let Some(b) = &self.sentences else { return Ok(None) };
        if tokens.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let seq = [tokens.to_vec()];
        let (features, have) = {
            let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
            (words.features(&seq, &[b.spec().word_pooling])?, words.taps())
        };
        let f = tail_taps(&features[0], have, b.taps());
        let n = tokens.len();
        let ids: Vec<u32> = (0..n as u32).collect();
        let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), Work::default());
        b.scores(&f[0], &ids, b.reads_forms().then_some(tokens), &mut scores, &mut work)?;
        b.decode(&scores, n, &mut ends)?;
        sentences(&ends).map(Some)
    }
}
