//! A rule-based NLI stand-in and document builders for the claim
//! verifier tests. The scorer is not a model: it decides from word sets,
//! so the tests and the eval measure the verifier's pruning and edge
//! logic, not entailment quality.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;

use turbo_analysis::{Document, Edges, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{ClaimRelationKind, Entities, Sentences, Tokens, Vectors};
use turbo_doccat::{NliScorer, ZeroShotError};

const NEGATIONS: &[&str] = &["not", "never", "no"];

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}

/// Contradiction when exactly one side is negated and the rest of the
/// words overlap by half; entailment when every hypothesis word is in
/// the premise; neutral otherwise.
pub struct RuleNli {
    pub outcomes: Vec<String>,
    pub max_pair_tokens: u32,
    pub seen: Mutex<Vec<(String, String)>>,
}

impl RuleNli {
    pub fn three() -> RuleNli {
        RuleNli::with(&["contradiction", "entailment", "neutral"])
    }

    pub fn with(outcomes: &[&str]) -> RuleNli {
        RuleNli {
            outcomes: outcomes.iter().map(|s| s.to_string()).collect(),
            max_pair_tokens: 512,
            seen: Mutex::default(),
        }
    }

    pub fn seen(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn decide(premise: &str, hypothesis: &str) -> &'static str {
        let (p, h) = (words(premise), words(hypothesis));
        let neg = |s: &HashSet<String>| s.iter().any(|w| NEGATIONS.contains(&w.as_str()));
        let strip = |s: &HashSet<String>| -> HashSet<String> {
            s.iter().filter(|w| !NEGATIONS.contains(&w.as_str())).cloned().collect()
        };
        let (ps, hs) = (strip(&p), strip(&h));
        let jaccard = ps.intersection(&hs).count() as f32 / ps.union(&hs).count().max(1) as f32;
        if neg(&p) != neg(&h) && jaccard >= 0.5 {
            "contradiction"
        } else if h.is_subset(&p) {
            "entailment"
        } else {
            "neutral"
        }
    }
}

impl NliScorer for RuleNli {
    fn outcome_labels(&self) -> &[String] {
        &self.outcomes
    }

    fn model_ref(&self) -> ModelRef {
        ModelRef { id: "rule-nli".into(), revision: "1".into(), sha256: "0".into() }
    }

    fn max_pair_tokens(&self) -> u32 {
        self.max_pair_tokens
    }

    fn pair_special_tokens(&self) -> usize {
        3
    }

    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, ZeroShotError> {
        self.seen.lock().unwrap().extend(pairs.iter().map(|(p, h)| (p.to_string(), h.to_string())));
        Ok(pairs
            .iter()
            .map(|(p, h)| {
                let decided = Self::decide(p, h);
                let hit = |o: &String| match (decided, o.as_str()) {
                    (d, o) if d == o => 4.0,
                    ("contradiction" | "neutral", "not_entailment") => 4.0,
                    _ => 0.0,
                };
                self.outcomes.iter().map(hit).collect()
            })
            .collect())
    }
}

/// A document of `sentences` joined by single spaces, tokenized on
/// spaces.
pub fn doc(sentences: &[&str]) -> Document {
    let text = sentences.join(" ");
    let (mut sents, mut tokens) = (Spans::new(ViewId(0)), Spans::new(ViewId(0)));
    let mut at = 0u32;
    for s in sentences {
        sents.push(Span::new(at, at + s.len() as u32));
        for w in s.split(' ') {
            tokens.push(Span::new(at, at + w.len() as u32));
            at += w.len() as u32 + 1;
        }
    }
    let mut d = Document::new(text.as_str());
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sents }, Provenance::by("test:sentences")).unwrap();
    d.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens")).unwrap();
    d
}

/// Marks every occurrence of each of `names` as an entity.
pub fn add_entities(d: &mut Document, names: &[&str]) {
    let text = d.text(ViewId(0)).to_string();
    let mut found: Vec<Span> = names
        .iter()
        .flat_map(|n| text.match_indices(n).map(|(i, _)| Span::new(i as u32, (i + n.len()) as u32)))
        .collect();
    found.sort_by_key(|s| (s.start, s.end));
    let n = found.len();
    let mut spans = Spans::new(ViewId(0));
    for s in found {
        spans.push(s);
    }
    let kinds = TagSet::new("test:kinds", &["ORG"]);
    let kind = vec![kinds.tag("ORG").unwrap(); n];
    let tokens = vec![IxRange { start: Ix::new(0), end: Ix::new(0) }; n];
    let e = Entities { spans, tokens, kind, kinds, score: vec![1.0; n] };
    d.add(&Key::<Entities>::DEFAULT, e, Provenance::by("test:entities")).unwrap();
}

/// One vector per sentence.
pub fn add_vectors(d: &mut Document, rows: &[&[f32]]) {
    let dim = rows[0].len() as u32;
    let data = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let model = ModelRef { id: "test-vec".into(), revision: "1".into(), sha256: "0".into() };
    let v = Vectors::<Sentences>::new(dim, data, Vec::new(), model);
    d.add(&Key::<Vectors<Sentences>>::DEFAULT, v, Provenance::by("test:vectors")).unwrap();
}

/// The edges as `(from, to, kind)`, sorted.
pub fn rows(edges: &Edges<Sentences, Sentences>) -> Vec<(usize, usize, ClaimRelationKind)> {
    let mut out: Vec<_> =
        edges.iter().map(|(_, e)| (e.from.get(), e.to.get(), ClaimRelationKind::from_tag(e.kind).unwrap())).collect();
    out.sort();
    out
}

/// Counts of the kinds in `rows`.
pub fn kinds(rows: &[(usize, usize, ClaimRelationKind)]) -> BTreeMap<ClaimRelationKind, usize> {
    let mut m = BTreeMap::new();
    for r in rows {
        *m.entry(r.2).or_insert(0) += 1;
    }
    m
}
