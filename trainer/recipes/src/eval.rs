//! Scoring predicted labels against gold, the majority baseline every
//! model is reported beside, and a paired bootstrap of the two.
//!
//! Tags score by word accuracy, also over the words training never saw.
//! BIO scores by exact span F1 (micro), per type, and per coarse group
//! where the corpus has one; spans are read the conlleval way, so an
//! `I-X` with no open `X` span starts one rather than being dropped.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use turbo_trainer::rng::Rng;

use crate::corpus::{Sentence, Task};

/// A span of words `start..end` of one type.
pub type Span = (usize, usize, String);

/// The spans of a BIO label sequence, read leniently.
pub fn spans(labels: &[String]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut open: Option<(usize, &str)> = None;
    for (i, l) in labels.iter().enumerate() {
        let (begin, kind) = match (l.strip_prefix("B-"), l.strip_prefix("I-")) {
            (Some(k), _) => (true, Some(k)),
            (_, Some(k)) => (false, Some(k)),
            _ => (false, None),
        };
        let continues = !begin && kind.is_some() && open.is_some_and(|(_, k)| Some(k) == kind);
        if !continues {
            if let Some((s, k)) = open.take() {
                out.push((s, i, k.to_owned()));
            }
            open = kind.map(|k| (i, k));
        }
    }
    if let Some((s, k)) = open {
        out.push((s, labels.len(), k.to_owned()));
    }
    out
}

/// Counts of one type, or of all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanScore {
    pub gold: usize,
    pub found: usize,
    pub correct: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl SpanScore {
    pub(crate) fn add(&mut self, gold: usize, found: usize, correct: usize) {
        self.gold += gold;
        self.found += found;
        self.correct += correct;
    }

    pub(crate) fn finish(mut self) -> Self {
        let r = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        self.precision = r(self.correct, self.found);
        self.recall = r(self.correct, self.gold);
        self.f1 = r(2 * self.correct, self.gold + self.found);
        self
    }
}

/// Precision, recall, and F1 for a single sub-metric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prf {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// A split's score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Metrics {
    Tags {
        words: usize,
        correct: usize,
        accuracy: f64,
        /// Words not in the training split, and how many of those were right.
        unseen_words: usize,
        unseen_correct: usize,
        /// Words whose gold tag training never gave, which no model trained
        /// on it can get right.
        unseen_labels: usize,
    },
    Spans {
        all: SpanScore,
        per_type: BTreeMap<String, SpanScore>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        coarse: BTreeMap<String, SpanScore>,
    },
    Coref {
        documents: usize,
        mentions: usize,
        gold_clusters: usize,
        predicted_clusters: usize,
        muc: Prf,
        b_cubed: Prf,
        ceaf_e: Prf,
        conll_f1: f64,
    },
    /// Relations between entities: exact (head, tail, label) triples,
    /// micro over every label but `none`, and per label.
    Relations { all: SpanScore, per_type: BTreeMap<String, SpanScore> },
}

impl Metrics {
    /// The headline number: accuracy, micro span F1, or CoNLL F1.
    pub fn score(&self) -> f64 {
        match self {
            Metrics::Tags { accuracy, .. } => *accuracy,
            Metrics::Spans { all, .. } | Metrics::Relations { all, .. } => all.f1,
            Metrics::Coref { conll_f1, .. } => *conll_f1,
        }
    }

    /// The name of the headline metric (`accuracy`, `span_f1`, or `conll_f1`).
    pub fn metric_name(&self) -> &'static str {
        match self {
            Metrics::Tags { .. } => "accuracy",
            Metrics::Spans { .. } | Metrics::Relations { .. } => "span_f1",
            Metrics::Coref { .. } => "conll_f1",
        }
    }
}

/// What training saw: its words and its labels.
#[derive(Debug, Clone, Default)]
pub struct Seen {
    pub words: BTreeSet<String>,
    pub labels: BTreeSet<String>,
}

impl Seen {
    pub fn of(train: &[Sentence]) -> Seen {
        let mut s = Seen::default();
        for t in train {
            s.words.extend(t.words.iter().cloned());
            s.labels.extend(t.labels.iter().cloned());
        }
        s
    }
}

/// Per sentence: (correct, total) for tags, (correct, found, gold) for spans.
fn tallies(task: Task, gold: &[Sentence], predicted: &[Vec<String>]) -> Vec<[usize; 3]> {
    gold.iter()
        .zip(predicted)
        .map(|(g, p)| match task {
            Task::Tags | Task::Coref => [g.labels.iter().zip(p).filter(|(a, b)| a == b).count(), g.labels.len(), 0],
            Task::Bio | Task::Relations => {
                let (want, got) = (spans(&g.labels), spans(p));
                [got.iter().filter(|s| want.contains(s)).count(), got.len(), want.len()]
            }
        })
        .collect()
}

fn headline(task: Task, t: &[[usize; 3]], pick: impl Fn(usize) -> usize) -> f64 {
    let mut sum = [0usize; 3];
    for i in 0..t.len() {
        let row = t[pick(i)];
        sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
    }
    let r = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    match task {
        Task::Tags | Task::Coref => r(sum[0], sum[1]),
        Task::Bio | Task::Relations => r(2 * sum[0], sum[1] + sum[2]),
    }
}

/// Scores `predicted` (one label sequence a sentence) against `gold`.
/// `coarse` maps a fine type to its group, for a corpus that has groups.
///
/// # Panics
///
/// When `predicted` and `gold` differ in sentences or in words; a caller
/// predicts exactly the words it is given.
pub fn evaluate(
    task: Task,
    gold: &[Sentence],
    predicted: &[Vec<String>],
    seen: &Seen,
    coarse: Option<fn(&str) -> &str>,
) -> Metrics {
    assert_eq!(gold.len(), predicted.len(), "one prediction a sentence");
    for (g, p) in gold.iter().zip(predicted) {
        assert_eq!(g.labels.len(), p.len(), "one label a word");
    }
    match task {
        Task::Tags | Task::Coref => {
            let (mut words, mut correct, mut uw, mut uc, mut ul) = (0, 0, 0, 0, 0);
            for (g, p) in gold.iter().zip(predicted) {
                for ((w, a), b) in g.words.iter().zip(&g.labels).zip(p) {
                    words += 1;
                    correct += usize::from(a == b);
                    if !seen.words.contains(w) {
                        uw += 1;
                        uc += usize::from(a == b);
                    }
                    ul += usize::from(!seen.labels.contains(a));
                }
            }
            let accuracy = if words == 0 { 0.0 } else { correct as f64 / words as f64 };
            Metrics::Tags { words, correct, accuracy, unseen_words: uw, unseen_correct: uc, unseen_labels: ul }
        }
        Task::Bio | Task::Relations => {
            let (mut all, mut per_type, mut groups) =
                (SpanScore::default(), BTreeMap::<String, SpanScore>::new(), BTreeMap::<String, SpanScore>::new());
            for (g, p) in gold.iter().zip(predicted) {
                let (want, got) = (spans(&g.labels), spans(p));
                for s in &want {
                    per_type.entry(s.2.clone()).or_default().add(1, 0, usize::from(got.contains(s)));
                }
                for s in &got {
                    per_type.entry(s.2.clone()).or_default().add(0, 1, 0);
                }
                all.add(want.len(), got.len(), got.iter().filter(|s| want.contains(s)).count());
                if let Some(c) = coarse {
                    let up =
                        |v: &[Span]| -> Vec<Span> { v.iter().map(|(a, b, k)| (*a, *b, c(k).to_owned())).collect() };
                    let (want, got) = (up(&want), up(&got));
                    for s in &want {
                        groups.entry(s.2.clone()).or_default().add(1, 0, usize::from(got.contains(s)));
                    }
                    for s in &got {
                        groups.entry(s.2.clone()).or_default().add(0, 1, 0);
                    }
                }
            }
            let done = |m: BTreeMap<String, SpanScore>| m.into_iter().map(|(k, v)| (k, v.finish())).collect();
            Metrics::Spans { all: all.finish(), per_type: done(per_type), coarse: done(groups) }
        }
    }
}

/// The majority baseline: each word's most frequent training label (the
/// lesser label on a tie), and the most frequent label overall for a word
/// training never saw.
#[derive(Debug, Clone)]
pub struct Majority {
    by_word: BTreeMap<String, String>,
    default: String,
}

impl Majority {
    pub fn train(train: &[Sentence]) -> Majority {
        let mut counts: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
        let mut overall: BTreeMap<&str, usize> = BTreeMap::new();
        for s in train {
            for (w, l) in s.words.iter().zip(&s.labels) {
                *counts.entry(w).or_default().entry(l).or_default() += 1;
                *overall.entry(l).or_default() += 1;
            }
        }
        // BTreeMap iterates labels in order, so `max_by` keeps the last of
        // the most frequent; reversing the order keeps the lesser label.
        let best = |m: &BTreeMap<&str, usize>| {
            m.iter().rev().max_by_key(|(_, n)| **n).map(|(l, _)| (*l).to_owned()).unwrap_or_default()
        };
        Majority { by_word: counts.iter().map(|(w, m)| ((*w).to_owned(), best(m))).collect(), default: best(&overall) }
    }

    pub fn predict(&self, words: &[String]) -> Vec<String> {
        words.iter().map(|w| self.by_word.get(w).unwrap_or(&self.default).clone()).collect()
    }
}

/// A paired bootstrap of a model against the baseline over sentences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    pub model: f64,
    pub baseline: f64,
    /// model - baseline on the whole split.
    pub delta: f64,
    /// The 2.5th and 97.5th percentiles of the delta over the resamples.
    pub low: f64,
    pub high: f64,
    pub resamples: usize,
    pub seed: u64,
}

/// Resamples the sentences `resamples` times with replacement, seeded,
/// and scores model and baseline on the same resample each time.
pub fn bootstrap(
    task: Task,
    gold: &[Sentence],
    model: &[Vec<String>],
    baseline: &[Vec<String>],
    resamples: usize,
    seed: u64,
) -> Comparison {
    let (m, b) = (tallies(task, gold, model), tallies(task, gold, baseline));
    let (ms, bs) = (headline(task, &m, |i| i), headline(task, &b, |i| i));
    let mut rng = Rng::new(seed);
    let mut deltas: Vec<f64> = (0..resamples)
        .map(|_| {
            let pick: Vec<usize> = (0..gold.len()).map(|_| rng.below(gold.len())).collect();
            headline(task, &m, |i| pick[i]) - headline(task, &b, |i| pick[i])
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let at = |q: f64| if deltas.is_empty() { 0.0 } else { deltas[((deltas.len() - 1) as f64 * q).round() as usize] };
    Comparison { model: ms, baseline: bs, delta: ms - bs, low: at(0.025), high: at(0.975), resamples, seed }
}
