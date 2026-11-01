//! Annotators over a bundle's token tagger heads (`manifest.heads`):
//! [`PosTagger`] writes a tag per token, [`EntityTagger`] writes the
//! spans a BIO or BILOU head marks. Each token is a word; the words are
//! read a sentence at a time (`Tokens.sentence`, else the sentence layer,
//! else the whole view) through [`WordEncoder`], then scored and decoded
//! by the core's `TokenTagger` on the host (`turbo::head::run`).

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::TokenTagger;
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Execution, Ix, IxRange, Layer as _, ModelRef, Span, Spans, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, PosTags, Sentences, Subwords, Tag, Tokens};

use crate::words::WordEncoder;

/// One head and the encoder that feeds it.
pub struct HeadRunner {
    words: Mutex<WordEncoder>,
    tagger: TokenTagger,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

pub use turbo::head::run::{Decoded, same_layer};

pub(crate) fn refuse(m: String) -> CoreError {
    CoreError::new(INVALID_ARGUMENT, m)
}

/// The head of `bundle` named `name`, or, with no name, the one head that
/// writes `layer`; refused unless it is of `kind`.
pub fn find_head<'a>(
    bundle: &'a Bundle,
    name: Option<&str>,
    layer: &str,
    kind: HeadKind,
) -> Result<&'a HeadSpec, CoreError> {
    let heads = &bundle.manifest.heads;
    let spec: &HeadSpec = match name {
        Some(n) => heads.iter().find(|h| h.name == n).ok_or_else(|| refuse(format!("no head {n:?}")))?,
        None => {
            let mut found = heads.iter().filter(|h| same_layer(&h.layer, layer));
            match (found.next(), found.next()) {
                (Some(h), None) => h,
                (None, _) => return Err(refuse(format!("no head writes {layer}"))),
                _ => return Err(refuse(format!("several heads write {layer}; name one"))),
            }
        }
    };
    if spec.kind != kind {
        return Err(refuse(format!("head {:?} is not a {kind:?} head", spec.name)));
    }
    Ok(spec)
}

/// The model a bundle's annotators name.
pub(crate) fn model_ref(bundle: &Bundle) -> ModelRef {
    let m = &bundle.manifest.model;
    ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() }
}

/// Opens `WordEncoder` configured for `taps` encoder layers.
pub fn open_words(dir: &Path, o: &EncoderOptions, taps: usize) -> Result<WordEncoder, CoreError> {
    let mut eo = o.clone();
    if eo.layers <= 1 {
        eo.layers = taps as u32;
    } else if eo.layers as usize != taps {
        return Err(refuse(format!("encoder options ask for {} layers, the head reads {taps}", eo.layers)));
    }
    WordEncoder::open(dir, &eo)
}

pub(crate) use turbo::head::run::tail_taps;

impl HeadRunner {
    /// The head of the bundle at `dir` named `name`, or, with no name, the
    /// one head that writes `layer`.
    pub fn open(dir: &Path, name: Option<&str>, layer: &str, o: &EncoderOptions) -> Result<HeadRunner, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, layer, HeadKind::TokenTagger)?;
        let tagger = TokenTagger::load(&bundle, &spec.name)?;
        let words = open_words(dir, o, tagger.taps())?;
        if words.hidden() != tagger.hidden() {
            return Err(refuse(format!(
                "the head reads {} values a word, the encoder gives {}",
                tagger.hidden(),
                words.hidden()
            )));
        }
        Ok(HeadRunner {
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            model: model_ref(&bundle),
            tagger,
        })
    }

    pub fn tagger(&self) -> &TokenTagger {
        &self.tagger
    }

    /// Each sequence of words decoded.
    pub fn decode(&self, sequences: &[Vec<&str>]) -> Result<Vec<Decoded>, CoreError> {
        let pooling = self.tagger.spec().word_pooling;
        let features = self.words.lock().unwrap_or_else(PoisonError::into_inner).features(sequences, pooling)?;
        decode_with(&self.tagger, &features, sequences)
    }

    /// Each sequence of `tokens` decoded, its pieces from `subwords` when
    /// the encoder reads them (`WordEncoder::reads`).
    pub fn decode_tokens(
        &self,
        view: &View,
        tokens: &Tokens,
        seqs: &[Vec<usize>],
        subwords: Option<&Subwords>,
    ) -> Result<Vec<Decoded>, String> {
        let pooling = self.tagger.spec().word_pooling;
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let features = words.token_features(view, tokens, seqs, subwords, pooling)?;
        drop(words);
        let forms = forms_of(view, tokens, seqs);
        decode_with(&self.tagger, &features, &forms).map_err(|e| e.to_string())
    }

    /// The token indices of each sequence of `tokens`, in order: by the
    /// token layer's own sentence column, else by `sentences`, else one.
    /// Every token is in exactly one sequence: a token across a sentence
    /// boundary, or outside every sentence, is refused, as are sentences
    /// off the view's UTF-8 boundaries.
    pub fn sequences(view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Vec<Vec<usize>>, String> {
        if tokens.spans.view() != view.id() || sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err("the tokens or sentences are in another view".into());
        }
        tokens.check_contiguous_in(view)?;
        let n = tokens.spans.len();
        if let Some(of) = &tokens.sentence {
            if of.len() != n {
                return Err("a sentence for every token".into());
            }
            let mut out: Vec<Vec<usize>> = Vec::new();
            for (i, s) in of.iter().enumerate() {
                if i == 0 || *s != of[i - 1] {
                    out.push(Vec::new());
                }
                out.last_mut().expect("pushed").push(i);
            }
            return Ok(out);
        }
        let Some(sentences) = sentences else {
            return Ok(if n == 0 { Vec::new() } else { vec![(0..n).collect()] });
        };
        sentences.check().map_err(|e| format!("invalid sentences: {e}"))?;
        if sentences.spans.iter().any(|(_, s)| view.text().get(s.range()).is_none()) {
            return Err("a sentence is off the view's UTF-8 boundaries".into());
        }
        let (starts, ends) = tokens.spans.columns();
        let mut out = Vec::new();
        let mut owned = 0;
        for (_, s) in sentences.spans.iter() {
            // The tokens wholly inside the sentence; tokens are in order
            // and do not overlap, so they are a run.
            let first = starts.partition_point(|&t| t < s.start);
            let last = ends.partition_point(|&e| e <= s.end);
            if first < last {
                out.push((first..last).collect::<Vec<_>>());
                owned += last - first;
            }
        }
        if owned != n {
            return Err("a token crosses a sentence boundary or lies outside every sentence".into());
        }
        Ok(out)
    }

    /// The tag of every token of the view, by `head`'s labels.
    pub fn pos_tags(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
        tagset: &Arc<TagSet>,
    ) -> Result<PosTags, String> {
        let seqs = Self::sequences(view, tokens, sentences)?;
        let decoded = self.decode_tokens(view, tokens, &seqs, subwords)?;
        pos_layer(tokens, &seqs, &decoded, tagset)
    }

    /// The entities `head` marks: each span from its first word's start
    /// to its last word's end, scored by the mean of its words'
    /// probabilities.
    pub fn entities(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
        kinds: &Arc<TagSet>,
    ) -> Result<Entities, String> {
        let seqs = Self::sequences(view, tokens, sentences)?;
        let decoded = self.decode_tokens(view, tokens, &seqs, subwords)?;
        entity_layer(&self.tagger, view, tokens, &seqs, &decoded, kinds)
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// Pieces a row holds beside its special tokens.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The scheme, for the annotators to check theirs.
    pub fn scheme(&self) -> Scheme {
        self.tagger.spec().scheme
    }
}

pub(crate) use turbo::head::run::decode_tagger as decode_with;

/// The part-of-speech layer from each sequence's decoded tags.
pub(crate) fn pos_layer(
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    decoded: &[Decoded],
    tagset: &Arc<TagSet>,
) -> Result<PosTags, String> {
    let n = tokens.spans.len();
    let (mut tags, mut prob) = (vec![Tag(0); n], vec![0.0; n]);
    let mut seen = vec![false; n];
    for (s, d) in seqs.iter().zip(decoded) {
        for (k, &i) in s.iter().enumerate() {
            (tags[i], prob[i], seen[i]) = (Tag(d.tags[k] as u16), d.prob[k], true);
        }
    }
    if seen.iter().any(|s| !s) {
        return Err("a token lies outside every sentence".into());
    }
    Ok(PosTags { tagset: tagset.clone(), tags, prob: Some(prob) })
}

/// The entity layer from each sequence's decoded labels: each span from
/// its first word's start to its last word's end, scored by the mean of
/// its words' probabilities.
pub(crate) fn entity_layer(
    tagger: &TokenTagger,
    view: &View,
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    decoded: &[Decoded],
    kinds: &Arc<TagSet>,
) -> Result<Entities, String> {
    let mut out = Entities {
        spans: Spans::new(view.id()),
        tokens: Vec::new(),
        kind: Vec::new(),
        kinds: kinds.clone(),
        score: Vec::new(),
    };
    for (s, d) in seqs.iter().zip(decoded) {
        for w in tagger.spans(&d.tags).map_err(|e| e.to_string())? {
            let (a, b) = (s[w.start as usize], s[w.end as usize - 1]);
            let start = tokens.spans.get(Ix::new(a as u32)).start;
            let end = tokens.spans.get(Ix::new(b as u32)).end;
            out.spans.push(Span::new(start, end));
            out.tokens.push(IxRange::new(a as u32, b as u32 + 1));
            out.kind.push(Tag(w.kind as u16));
            let p = &d.prob[w.start as usize..w.end as usize];
            out.score.push(p.iter().sum::<f64>() / p.len() as f64);
        }
    }
    Ok(out)
}

/// Each sequence's words' text.
pub(crate) fn forms_of<'v>(view: &'v View, tokens: &Tokens, seqs: &[Vec<usize>]) -> Vec<Vec<&'v str>> {
    let text = view.text();
    seqs.iter().map(|s| s.iter().map(|&i| &text[tokens.spans.get(Ix::new(i as u32)).range()]).collect()).collect()
}
