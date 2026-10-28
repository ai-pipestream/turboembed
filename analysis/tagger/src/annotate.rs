//! Annotators over a bundle's token tagger heads (`manifest.heads`):
//! [`PosTagger`] writes a tag per token, [`EntityTagger`] writes the
//! spans a BIO or BILOU head marks. Each token is a word; the words are
//! read a sentence at a time (`Tokens.sentence`, else the sentence layer,
//! else the whole view) through [`WordEncoder`], then scored and decoded
//! by the core's `TokenTagger` on the host.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::TokenTagger;
use turbo::head::tagger::Work;
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Execution, Ix, IxRange, Layer as _, ModelRef, Span, Spans, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tag, Tokens};

use crate::words::WordEncoder;

/// One head and the encoder that feeds it.
pub struct HeadRunner {
    words: Mutex<WordEncoder>,
    tagger: TokenTagger,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

/// What a sequence decoded to: each word's label and the softmax
/// probability of that label among the word's own scores.
pub struct Decoded {
    pub tags: Vec<u32>,
    pub prob: Vec<f64>,
}

/// Whether a head's layer id names the layer `id`. A head's id is always
/// `turboembed:<name>`; until the analysis layer ids move to that
/// namespace too (#1534), the names after the namespaces are compared.
pub fn same_layer(head: &str, id: &str) -> bool {
    let name = |s: &str| s.rsplit_once(':').map_or(s, |(_, n)| n).to_owned();
    name(head) == name(id)
}

fn refuse(m: String) -> CoreError {
    CoreError::new(INVALID_ARGUMENT, m)
}

impl HeadRunner {
    /// The head of the bundle at `dir` named `name`, or, with no name, the
    /// one head that writes `layer`.
    pub fn open(dir: &Path, name: Option<&str>, layer: &str, o: &EncoderOptions) -> Result<HeadRunner, CoreError> {
        let bundle = Bundle::open(dir)?;
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
        if spec.kind != HeadKind::TokenTagger {
            return Err(refuse(format!("head {:?} is not a token tagger", spec.name)));
        }
        let tagger = TokenTagger::load(&bundle, &spec.name)?;
        let words = WordEncoder::open(dir, o)?;
        if words.hidden() != tagger.hidden() {
            return Err(refuse(format!(
                "the head reads {} values a word, the encoder gives {}",
                tagger.hidden(),
                words.hidden()
            )));
        }
        let m = &bundle.manifest.model;
        Ok(HeadRunner {
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            model: ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() },
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
        let (mut scores, mut work) = (Vec::new(), Work::default());
        let labels = self.tagger.labels().len();
        let mut out = Vec::with_capacity(sequences.len());
        for (s, f) in sequences.iter().zip(&features) {
            let n = s.len();
            let words: Vec<u32> = (0..n as u32).collect();
            self.tagger.scores(f, &words, &mut scores, &mut work)?;
            let mut tags = Vec::new();
            self.tagger.decode(&scores, n, &mut tags)?;
            let prob = tags
                .iter()
                .enumerate()
                .map(|(w, &t)| {
                    let row = &scores[w * labels..][..labels];
                    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
                    let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
                    (row[t as usize] as f64 - m).exp() / z
                })
                .collect();
            out.push(Decoded { tags, prob });
        }
        Ok(out)
    }

    /// The token indices of each sequence of `tokens`, in order: by the
    /// token layer's own sentence column, else by `sentences`, else one.
    pub fn sequences(view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Vec<Vec<usize>>, String> {
        if tokens.spans.view() != view.id() || sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err("the tokens or sentences are in another view".into());
        }
        tokens.check_contiguous_in(view)?;
        let n = tokens.spans.len();
        if let Some(of) = &tokens.sentence {
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
        let (starts, ends) = tokens.spans.columns();
        let mut out = Vec::new();
        for (_, s) in sentences.spans.iter() {
            let first = ends.partition_point(|&e| e <= s.start);
            let last = starts.partition_point(|&t| t < s.end);
            if first < last {
                out.push((first..last).collect());
            }
        }
        Ok(out)
    }

    fn texts<'a>(view: &'a View, tokens: &Tokens, seqs: &[Vec<usize>]) -> Vec<Vec<&'a str>> {
        seqs.iter()
            .map(|s| s.iter().map(|&i| &view.text()[tokens.spans.get(Ix::new(i as u32)).range()]).collect())
            .collect()
    }

    /// The tag of every token of the view, by `head`'s labels.
    pub fn pos_tags(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        tagset: &Arc<TagSet>,
    ) -> Result<PosTags, String> {
        let seqs = Self::sequences(view, tokens, sentences)?;
        let decoded = self.decode(&Self::texts(view, tokens, &seqs)).map_err(|e| e.to_string())?;
        let n = tokens.spans.len();
        let (mut tags, mut prob) = (vec![Tag(0); n], vec![0.0; n]);
        let mut seen = vec![false; n];
        for (s, d) in seqs.iter().zip(&decoded) {
            for (k, &i) in s.iter().enumerate() {
                (tags[i], prob[i], seen[i]) = (Tag(d.tags[k] as u16), d.prob[k], true);
            }
        }
        if seen.iter().any(|s| !s) {
            return Err("a token lies outside every sentence".into());
        }
        Ok(PosTags { tagset: tagset.clone(), tags, prob: Some(prob) })
    }

    /// The entities `head` marks: each span from its first word's start
    /// to its last word's end, scored by the mean of its words'
    /// probabilities.
    pub fn entities(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        kinds: &Arc<TagSet>,
    ) -> Result<Entities, String> {
        let seqs = Self::sequences(view, tokens, sentences)?;
        let decoded = self.decode(&Self::texts(view, tokens, &seqs)).map_err(|e| e.to_string())?;
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: kinds.clone(),
            score: Vec::new(),
        };
        for (s, d) in seqs.iter().zip(&decoded) {
            for w in self.tagger.spans(&d.tags).map_err(|e| e.to_string())? {
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
