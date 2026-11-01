//! Lemma and dependency heads on a UD treebank: their specs, examples
//! read through the encoder, prediction through the core's heads, and
//! the baselines their receipts score against.
//!
//! Both score as word labels, so the receipt's tag metrics and bootstrap
//! apply unchanged: a lemma head's label is the lemma, a parser's is
//! `<head> <relation>` (`deps_label`), whose accuracy is LAS. UAS is
//! scored beside it (`uas`).

use std::collections::BTreeMap;

use turbo::head::{BiaffineParser, EditTreeLemmatizer};
use turbo::manifest::{Adapter, Biaffine, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_tagger::WordEncoder;
use turbo_trainer::{LemmaExample, ParseExample};

use crate::corpus::{CorpusRecord, Counts, FileRecord, Sentence, Task};
use crate::run::PREDICT_CHUNK;
use crate::ud::{Treebank, UdSentence, UdSplits};
use crate::{Error, Result};

/// What a UD head learns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdHead {
    Lemma,
    Deps,
}

impl UdHead {
    pub fn suffix(self) -> &'static str {
        match self {
            UdHead::Lemma => "lemma",
            UdHead::Deps => "deps",
        }
    }

    /// Each word's label: its lemma, or [`deps_label`].
    pub fn labels(self, s: &UdSentence) -> Vec<String> {
        match self {
            UdHead::Lemma => s.lemmas.clone(),
            UdHead::Deps => s.heads.iter().zip(&s.rels).map(|(&h, r)| deps_label(h, r)).collect(),
        }
    }

    /// The sentences of `part`, labelled as this head scores them.
    pub fn sentences(self, part: &[UdSentence]) -> Vec<Sentence> {
        part.iter().map(|s| Sentence { words: s.words.clone(), labels: self.labels(s) }).collect()
    }
}

/// A word's head (0 the root) and relation as one label.
pub fn deps_label(head: u32, rel: &str) -> String {
    format!("{head} {rel}")
}

/// The corpus record of `bank` read for `head`.
pub fn record(
    bank: &Treebank,
    head: UdHead,
    splits: &UdSplits,
    files: Vec<FileRecord>,
    recorded: bool,
) -> CorpusRecord {
    let counts = |p: &[UdSentence]| Counts { sentences: p.len(), words: p.iter().map(|s| s.words.len()).sum() };
    CorpusRecord {
        id: format!("{}-{}", bank.id, head.suffix()),
        name: format!("{}, {}", bank.name, head.suffix()),
        task: Task::Tags,
        licence: bank.licence.into(),
        spdx: bank.spdx.into(),
        terms: bank.terms.into(),
        attribution: bank.attribution.into(),
        recorded,
        corpus_sha256: String::new(),
        files,
        split: format!("the release's own train, dev and test files at {} ({})", bank.tag, bank.commit),
        train: counts(&splits.train),
        dev: counts(&splits.dev),
        test: counts(&splits.test),
        skipped: 0,
        recipe_only: bank.recipe_only,
    }
}

/// The head's tensors under their bundle names, with the shapes the
/// core's lemmatizer or parser reads, as `crate::safetensors::write`
/// takes them.
pub fn named(
    head: UdHead,
    spec: &HeadSpec,
    hidden: usize,
    values: Vec<(String, Vec<f32>)>,
) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let shapes: BTreeMap<String, Vec<u64>> = match head {
        UdHead::Lemma => turbo::head::lemmatizer::tensors(spec, hidden),
        UdHead::Deps => turbo::head::parser::tensors(spec, hidden),
    }
    .into_iter()
    .collect();
    values
        .into_iter()
        .map(|(suffix, v)| {
            let shape = shapes.get(&suffix).cloned().unwrap_or_else(|| vec![v.len() as u64]);
            (format!("{}.{suffix}", spec.tensor_prefix), shape, v)
        })
        .collect()
}

/// The relations of `train`, sorted.
pub fn relations(train: &[UdSentence]) -> Vec<String> {
    let mut all: Vec<String> = train.iter().flat_map(|s| s.rels.iter().cloned()).collect();
    all.sort();
    all.dedup();
    all
}

/// The shape of a UD head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdOptions {
    pub pooling: WordPooling,
    pub bilstm: Option<(u32, u32)>,
    /// A parser's arc and relation projection widths.
    pub biaffine: Biaffine,
}

/// The head `head` trains over `labels` (the mined trees, or the
/// relations), carrying the class of the corpus's licence.
///
/// # Errors
///
/// A head the core would refuse, or a licence the zoo refuses.
pub fn spec(head: UdHead, labels: Vec<String>, corpus: &CorpusRecord, o: &UdOptions) -> Result<HeadSpec> {
    let (name, layer, kind, decode, biaffine) = match head {
        UdHead::Lemma => ("lemma", "turboembed:lemmas", HeadKind::EditTreeLemmatizer, Decode::EditTree, None),
        UdHead::Deps => ("deps", "turboembed:dependencies", HeadKind::BiaffineParser, Decode::Mst, Some(o.biaffine)),
    };
    let spec = HeadSpec {
        name: name.into(),
        layer: layer.into(),
        kind,
        labels,
        scheme: Scheme::Tags,
        word_pooling: o.pooling,
        adapter: o.bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        biaffine,
        pos_input: None,
        hash_embed: None,
        layer_mix: None,
        license_class: Some(corpus.class()?),
        span: None,
        relation: None,
        artifact: "weights-f32".into(),
        tensor_prefix: format!("heads.{name}"),
    };
    spec.validate("head").map_err(|e| Error(e.message))?;
    Ok(spec)
}

/// Each sentence's word features, read `PREDICT_CHUNK` sentences at a
/// time.
///
/// # Errors
///
/// What the encoder fails on.
pub fn features(enc: &mut WordEncoder, part: &[UdSentence], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(part.len());
    for chunk in part.chunks(PREDICT_CHUNK) {
        let words: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
        out.extend(enc.features(&words, pooling).map_err(|e| Error(e.message))?);
    }
    Ok(out)
}

/// Lemma examples of `part` with its features.
pub fn lemma_examples(part: &[UdSentence], features: Vec<Vec<f32>>) -> Vec<LemmaExample> {
    part.iter()
        .zip(features)
        .map(|(s, features)| LemmaExample {
            features,
            tags: Vec::new(),
            forms: s.words.clone(),
            lemmas: s.lemmas.clone(),
        })
        .collect()
}

/// Parse examples of `part` with its features. A sentence with a
/// relation `rels` lacks is left out, and counted.
pub fn parse_examples(part: &[UdSentence], features: Vec<Vec<f32>>, rels: &[String]) -> (Vec<ParseExample>, usize) {
    let index: BTreeMap<&str, u32> = rels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let mut left_out = 0;
    let mut out = Vec::with_capacity(part.len());
    for (s, features) in part.iter().zip(features) {
        match s.rels.iter().map(|r| index.get(r.as_str()).copied()).collect::<Option<Vec<u32>>>() {
            Some(ids) => out.push(ParseExample { features, heads: s.heads.clone(), rels: ids }),
            None => left_out += 1,
        }
    }
    (out, left_out)
}

/// The lemmas `head` gives each sentence, decoded as at inference.
///
/// # Errors
///
/// What the head fails on.
pub fn predict_lemmas(
    head: &EditTreeLemmatizer,
    part: &[UdSentence],
    features: &[Vec<f32>],
) -> Result<Vec<Vec<String>>> {
    let (mut scores, mut trees, mut w) = (Vec::new(), Vec::new(), Default::default());
    let mut out = Vec::with_capacity(part.len());
    for (s, f) in part.iter().zip(features) {
        let words: Vec<u32> = (0..s.words.len() as u32).collect();
        let forms: Vec<&str> = s.words.iter().map(String::as_str).collect();
        head.scores(f, &words, None, &mut scores, &mut w).map_err(|e| Error(e.message))?;
        head.decode(&scores, &forms, &mut trees).map_err(|e| Error(e.message))?;
        out.push(head.lemmas(&trees, &forms).map_err(|e| Error(e.message))?);
    }
    Ok(out)
}

/// The `<head> <relation>` labels `parser` gives each sentence, decoded
/// as at inference.
///
/// # Errors
///
/// What the parser fails on.
pub fn predict_parses(parser: &BiaffineParser, part: &[UdSentence], features: &[Vec<f32>]) -> Result<Vec<Vec<String>>> {
    let (mut arcs, mut heads, mut scores, mut rels, mut w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Default::default());
    let mut out = Vec::with_capacity(part.len());
    for (s, f) in part.iter().zip(features) {
        let n = s.words.len();
        let words: Vec<u32> = (0..n as u32).collect();
        parser.arcs(f, &words, &mut arcs, &mut w).map_err(|e| Error(e.message))?;
        parser.decode(&arcs, n, &mut heads).map_err(|e| Error(e.message))?;
        parser.relations(&heads, &mut scores, &w).map_err(|e| Error(e.message))?;
        parser.relation_labels(&scores, n, &mut rels).map_err(|e| Error(e.message))?;
        out.push(heads.iter().zip(&rels).map(|(&h, &r)| deps_label(h, &parser.labels()[r as usize])).collect());
    }
    Ok(out)
}

/// The commonest value of each key in `pairs`, ties to the smaller value.
fn majority<'a>(pairs: impl Iterator<Item = (&'a str, &'a str)>) -> BTreeMap<&'a str, &'a str> {
    let mut counts: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for (k, v) in pairs {
        *counts.entry(k).or_default().entry(v).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(k, vs)| (k, vs.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0))).map_or("", |(v, _)| v)))
        .collect()
}

/// The lemma baseline: each word's commonest training lemma, the word
/// itself when training never saw it.
pub fn lemma_baseline(train: &[UdSentence], test: &[UdSentence]) -> Vec<Vec<String>> {
    let m = majority(train.iter().flat_map(|s| s.words.iter().zip(&s.lemmas).map(|(w, l)| (w.as_str(), l.as_str()))));
    test.iter()
        .map(|s| s.words.iter().map(|w| m.get(w.as_str()).map_or(w.clone(), |l| (*l).to_owned())).collect())
        .collect()
}

/// The dependency baseline: each word takes its commonest training
/// attachment, the signed distance to its head (or the root) with the
/// relation together, the commonest overall when training never saw the
/// word. A distance that leaves the sentence or lands on the word itself
/// is moved to the nearest word that is neither.
pub fn deps_baseline(train: &[UdSentence], test: &[UdSentence]) -> Vec<Vec<String>> {
    let attach = |i: usize, h: u32, r: &str| match h {
        0 => format!("root {r}"),
        h => format!("{} {r}", i64::from(h) - (i as i64 + 1)),
    };
    let pairs: Vec<(&str, String)> = train
        .iter()
        .flat_map(|s| s.words.iter().enumerate().map(move |(i, w)| (w.as_str(), attach(i, s.heads[i], &s.rels[i]))))
        .collect();
    let by_word = majority(pairs.iter().map(|(w, a)| (*w, a.as_str())));
    let overall = majority(pairs.iter().map(|(_, a)| ("", a.as_str())))
        .get("")
        .map_or("root root".to_owned(), |a| (*a).to_owned());
    test.iter()
        .map(|s| {
            let n = s.words.len() as i64;
            s.words
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let a = by_word.get(w.as_str()).copied().unwrap_or(&overall);
                    let (d, r) = a.split_once(' ').unwrap_or(("root", a));
                    let at = i as i64 + 1;
                    let head = match d.parse::<i64>() {
                        Err(_) => 0,
                        Ok(d) => match (at + d).clamp(1, n) {
                            h if h != at => h,
                            _ if at > 1 => at - 1,
                            _ if n > 1 => at + 1,
                            _ => 0,
                        },
                    };
                    deps_label(head as u32, r)
                })
                .collect()
        })
        .collect()
}

/// Unlabelled attachment of `predicted` against `gold`, both
/// [`deps_label`]s: the share of words whose head is right.
pub fn uas(gold: &[Sentence], predicted: &[Vec<String>]) -> f64 {
    let head = |l: &str| l.split_once(' ').map(|(h, _)| h.to_owned());
    let (mut right, mut total) = (0usize, 0usize);
    for (g, p) in gold.iter().zip(predicted) {
        total += g.labels.len();
        right += g.labels.iter().zip(p).filter(|(a, b)| head(a) == head(b)).count();
    }
    if total == 0 { 0.0 } else { right as f64 / total as f64 }
}
