//! Annotators over a bundle's `HEAD_TOKEN_SPLIT` (`turboembed:words`) and
//! language-span `HEAD_TOKEN_TAGGER` (`turbo:language-spans`) heads (#1726).

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::tagger::Work;
use turbo::head::{TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, IxRange, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{LanguageSpans, Sentences, Subwords, Tokens, UNDETERMINED, Words};
use turbo_unicode::Script;

use crate::annotate::{Decoded, HeadRunner, decode_with, find_head, forms_of, model_ref, refuse};
use crate::multi::{Head, HeadLayer, Input, wrong_head};
use crate::words::WordEncoder;

/// Writes the syntactic [`Words`] layer (`turboembed:words`) from a
/// `HEAD_TOKEN_SPLIT` head (#1516, #1726).
pub struct WordSplitter {
    words: Mutex<WordEncoder>,
    splitter: TokenSplitter,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

/// Alias for [`WordSplitter`].
pub type WordSplitTagger = WordSplitter;

impl WordSplitter {
    /// Opens the `HEAD_TOKEN_SPLIT` head named `name` in the bundle at `dir`,
    /// or with no name the one head that writes [`Words::ID`].
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<WordSplitter, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Words::ID, HeadKind::TokenSplit)?;
        let splitter = TokenSplitter::load(&bundle, &spec.name)?;
        let words = WordEncoder::open(dir, o)?;
        if words.hidden() != splitter.hidden() {
            return Err(refuse(format!(
                "the head reads {} values a word, the encoder gives {}",
                splitter.hidden(),
                words.hidden()
            )));
        }
        Ok(WordSplitter {
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            model: model_ref(&bundle),
            splitter,
        })
    }

    pub fn splitter(&self) -> &TokenSplitter {
        &self.splitter
    }

    /// Splits each token in each sequence into one or more syntactic word forms.
    pub fn split(&self, sequences: &[Vec<&str>]) -> Result<Vec<Vec<Vec<String>>>, CoreError> {
        let pooling = self.splitter.spec().word_pooling;
        let features = self.words.lock().unwrap_or_else(PoisonError::into_inner).features(sequences, pooling)?;
        split_with(&self.splitter, &features, sequences)
    }

    /// Builds [`Words`] over `tokens` of `view`.
    pub fn words(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Words, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let pooling = self.splitter.spec().word_pooling;
        let features = self
            .words
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .token_features(view, tokens, &seqs, subwords, pooling)?;
        let forms = forms_of(view, tokens, &seqs);
        let split = split_with(&self.splitter, &features, &forms).map_err(|e| e.to_string())?;
        words_layer(view, tokens, &seqs, split)
    }
}

pub(crate) fn split_with(
    splitter: &TokenSplitter,
    features: &[Vec<f32>],
    forms: &[Vec<&str>],
) -> Result<Vec<Vec<Vec<String>>>, CoreError> {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut work = Work::default();
    let hidden = splitter.hidden();
    let mut out = Vec::with_capacity(features.len());
    for (f, s) in features.iter().zip(forms) {
        let n = f.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        splitter.scores(f, &words, Some(s), &mut scores, &mut work)?;
        splitter.decode(&scores, s, &mut labels)?;
        out.push(splitter.words(&labels, s)?);
    }
    Ok(out)
}

pub(crate) fn words_layer(
    view: &View,
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    split: Vec<Vec<Vec<String>>>,
) -> Result<Words, String> {
    let n = tokens.spans.len();
    let mut parts_by_tok: Vec<Option<Vec<String>>> = vec![None; n];
    for (seq, seq_parts) in seqs.iter().zip(split) {
        for (&t, parts) in seq.iter().zip(seq_parts) {
            parts_by_tok[t] = Some(parts);
        }
    }
    let mut out = Words::new();
    for (t, entry) in parts_by_tok.into_iter().enumerate() {
        let idx = t as u32;
        match entry {
            Some(parts) if parts.len() >= 2 => {
                let range = IxRange::new(idx, idx + 1);
                for p in parts {
                    out.push_word(range, Some(&p), None);
                }
            }
            _ => {
                out.push_token(Ix::new(idx));
            }
        }
    }
    out.check_against(view.text(), tokens)?;
    Ok(out)
}

impl Annotator for WordSplitter {
    const ID: &'static str = "turbo-tagger-words";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Words,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.words(view, tokens, sentences, subwords)
            .map(|w| (w,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::new().str(&self.splitter.spec().name).u64(self.width as u64).finish()
    }
}

impl HeadLayer for Words {
    const KIND: HeadKind = HeadKind::TokenSplit;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let splitter = TokenSplitter::load(bundle, &spec.name)?;
        Ok(Head::Splitter { splitter })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Splitter { splitter } = head else { return wrong_head() };
        let f = input.features(splitter.spec().word_pooling)?;
        let split = split_with(splitter, f, &input.forms).map_err(|e| e.to_string())?;
        words_layer(input.view, input.tokens, &input.seqs, split)
    }
}

/// Writes the [`LanguageSpans`] layer (`turbo:language-spans`) from a
/// `HEAD_TOKEN_TAGGER` head (#1726).
pub struct LanguageTagger {
    run: HeadRunner,
    languages: Arc<TagSet>,
}

impl LanguageTagger {
    /// Opens the `HEAD_TOKEN_TAGGER` head named `name` in the bundle at `dir`,
    /// or with no name the one head that writes [`LanguageSpans::ID`].
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<LanguageTagger, CoreError> {
        let run = HeadRunner::open(dir, name, LanguageSpans::ID, o)?;
        let languages = language_tagset(run.tagger());
        Ok(LanguageTagger { run, languages })
    }

    pub fn languages(&self) -> &Arc<TagSet> {
        &self.languages
    }

    /// Builds [`LanguageSpans`] over `tokens` of `view`.
    pub fn language_spans(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<LanguageSpans, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let decoded = self.run.decode_tokens(view, tokens, &seqs, subwords)?;
        language_layer(self.run.tagger(), view, tokens, &seqs, &decoded, &self.languages)
    }
}

pub(crate) fn language_tagset(tagger: &TokenTagger) -> Arc<TagSet> {
    let spec = tagger.spec();
    let mut labels: Vec<&str> = if spec.scheme == Scheme::Tags {
        spec.labels.iter().map(String::as_str).filter(|&l| l != "O").collect()
    } else {
        tagger.types().iter().map(String::as_str).collect()
    };
    if !labels.contains(&UNDETERMINED) {
        labels.push(UNDETERMINED);
    }
    TagSet::new(&spec.name, &labels)
}

pub(crate) fn language_layer(
    tagger: &TokenTagger,
    view: &View,
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    decoded: &[Decoded],
    languages: &Arc<TagSet>,
) -> Result<LanguageSpans, String> {
    let mut raw: Vec<(u32, u32, Tag, f64)> = Vec::new();
    let spec = tagger.spec();
    if spec.scheme == Scheme::Tags {
        for (seq, dec) in seqs.iter().zip(decoded) {
            let mut cur: Option<(usize, usize, Tag, f64, usize)> = None;
            for (&tok_idx, (&tag_id, &prob)) in seq.iter().zip(dec.tags.iter().zip(&dec.prob)) {
                let label = spec.labels.get(tag_id as usize).map(String::as_str).unwrap_or("O");
                let tag = if label == "O" { None } else { languages.tag(label) };
                match (cur.take(), tag) {
                    (Some((s, _e, prev, sum, n)), Some(t)) if prev == t => {
                        cur = Some((s, tok_idx, prev, sum + prob, n + 1));
                    }
                    (Some((s, e, prev, sum, n)), next) => {
                        let start = tokens.spans.get(Ix::new(s as u32)).start;
                        let end = tokens.spans.get(Ix::new(e as u32)).end;
                        raw.push((start, end, prev, sum / n as f64));
                        if let Some(t) = next {
                            cur = Some((tok_idx, tok_idx, t, prob, 1));
                        }
                    }
                    (None, Some(t)) => {
                        cur = Some((tok_idx, tok_idx, t, prob, 1));
                    }
                    (None, None) => {}
                }
            }
            if let Some((s, e, prev, sum, n)) = cur {
                let start = tokens.spans.get(Ix::new(s as u32)).start;
                let end = tokens.spans.get(Ix::new(e as u32)).end;
                raw.push((start, end, prev, sum / n as f64));
            }
        }
    } else {
        for (seq, dec) in seqs.iter().zip(decoded) {
            for w in tagger.spans(&dec.tags).map_err(|e| e.to_string())? {
                let (a, b) = (seq[w.start as usize], seq[w.end as usize - 1]);
                let start = tokens.spans.get(Ix::new(a as u32)).start;
                let end = tokens.spans.get(Ix::new(b as u32)).end;
                let p = &dec.prob[w.start as usize..w.end as usize];
                let conf = p.iter().sum::<f64>() / p.len() as f64;
                raw.push((start, end, Tag(w.kind as u16), conf));
            }
        }
    }
    raw.sort_by_key(|&(start, end, _, _)| (start, end));
    let mut out = LanguageSpans::new(view.id(), languages.clone());
    let text = view.text();
    for (start, end, tag, conf) in raw {
        let span = Span::new(start, end);
        let script = text
            .get(span.range())
            .and_then(|s| {
                s.chars()
                    .map(turbo_unicode::script::script)
                    .find(|&sc| !matches!(sc, Script::Common | Script::Inherited | Script::Unknown))
            })
            .unwrap_or(Script::Common);
        out.push(span, tag, script, conf.clamp(0.0, 1.0));
    }
    out.check()?;
    Ok(out)
}

impl Annotator for LanguageTagger {
    const ID: &'static str = "turbo-tagger-language";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (LanguageSpans,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.language_spans(view, tokens, sentences, subwords)
            .map(|l| (l,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.run.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.run.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::new().str(&self.run.tagger().spec().name).u64(self.run.width() as u64).finish()
    }
}

impl HeadLayer for LanguageSpans {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let tagger = TokenTagger::load(bundle, &spec.name)?;
        let set = language_tagset(&tagger);
        Ok(Head::Tagger { set, tagger })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.features(tagger.spec().word_pooling)?;
        let decoded = decode_with(tagger, f, &input.forms).map_err(|e| e.to_string())?;
        language_layer(tagger, input.view, input.tokens, &input.seqs, &decoded, set)
    }
}
