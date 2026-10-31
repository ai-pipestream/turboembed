//! The relation head on a relation corpus (`redocred`): its spec, its
//! training examples from the encoder's word features, prediction through
//! the core's classifier, and scoring against gold by exact relation
//! triples, micro and per label, beside the majority baseline (`none` for
//! every pair, so it finds nothing) and with a paired bootstrap over
//! documents.

use std::collections::BTreeMap;

use turbo::head::RelationClassifier;
use turbo::head::relation::RelationWork;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, RelationShape, Scheme, WordPooling};
use turbo_tagger::WordEncoder;
use turbo_trainer::RelationExample;
use turbo_trainer::relation::GoldRelation;
use turbo_trainer::rng::Rng;

use crate::eval::{Comparison, Metrics, SpanScore};
use crate::redocred::{Document, ENTITY_TYPES, HEAD, LAYER, RelationCorpus, Triple};
use crate::{Error, Result};

/// Documents read by the encoder at a time.
pub const CHUNK: usize = 64;

/// The shape of the relation head to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationOptions {
    pub pooling: WordPooling,
    /// Width of the head and tail projections.
    pub mlp: u32,
    /// Width of the entity type embedding.
    pub type_dim: u32,
    /// BiLSTM adapter (hidden, layers), or none.
    pub bilstm: Option<(u32, u32)>,
}

impl Default for RelationOptions {
    fn default() -> Self {
        RelationOptions { pooling: WordPooling::First, mlp: 256, type_dim: 32, bilstm: None }
    }
}

/// The relation head for `corpus`, over its labels and entity types,
/// carrying the class of its licence.
///
/// # Errors
///
/// A head the core would refuse, or a licence the zoo does not know or
/// refuses.
pub fn spec(corpus: &RelationCorpus, o: &RelationOptions) -> Result<HeadSpec> {
    let spec = HeadSpec {
        name: HEAD.into(),
        layer: LAYER.into(),
        kind: HeadKind::RelationClassifier,
        labels: corpus.labels(),
        scheme: Scheme::Tags,
        word_pooling: o.pooling,
        adapter: o.bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::Relations,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        coref: None,
        relation: Some(RelationShape {
            mlp: o.mlp,
            entity_labels: ENTITY_TYPES.iter().map(|t| (*t).to_owned()).collect(),
            type_dim: o.type_dim,
        }),
        license_class: Some(corpus.record.class()?),
        artifact: "weights-f32".into(),
        tensor_prefix: format!("heads.{HEAD}"),
    };
    spec.validate("head").map_err(|e| Error(e.message))?;
    Ok(spec)
}

/// Each document's training example: its word features, mentions, entity
/// type indices and the gold relations whose label `labels` holds. The
/// relations left out, of labels training never gave, are counted.
///
/// # Errors
///
/// An entity type outside [`ENTITY_TYPES`], or what the encoder fails on.
pub fn examples(
    enc: &mut WordEncoder,
    docs: &[Document],
    labels: &[String],
    pooling: WordPooling,
) -> Result<(Vec<RelationExample>, usize)> {
    let index: BTreeMap<&str, u32> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let (mut out, mut left_out) = (Vec::with_capacity(docs.len()), 0);
    for chunk in docs.chunks(CHUNK) {
        let words: Vec<Vec<&str>> = chunk.iter().map(|d| d.words.iter().map(String::as_str).collect()).collect();
        let features = enc.features(&words, pooling).map_err(|e| Error(e.message))?;
        for (features, d) in features.into_iter().zip(chunk) {
            let types = d
                .types
                .iter()
                .map(|t| ENTITY_TYPES.iter().position(|k| k == t).map(|i| i as u32))
                .collect::<Option<Vec<u32>>>()
                .ok_or_else(|| Error(format!("{}: an entity type outside {ENTITY_TYPES:?}", d.title)))?;
            let mut relations = Vec::with_capacity(d.relations.len());
            for (head, tail, r) in &d.relations {
                match index.get(r.as_str()) {
                    Some(&label) => relations.push(GoldRelation { head: *head, tail: *tail, label }),
                    None => left_out += 1,
                }
            }
            out.push(RelationExample { features, mentions: d.mentions.clone(), types, relations });
        }
    }
    Ok((out, left_out))
}

/// The relations `head` keeps in each example, decoded as at inference.
///
/// # Errors
///
/// What the head refuses.
pub fn predict(head: &RelationClassifier, examples: &[RelationExample]) -> Result<Vec<Vec<Triple>>> {
    let hidden = head.hidden();
    let (mut scores, mut kept, mut w) = (Vec::new(), Vec::new(), RelationWork::default());
    let mut out = Vec::with_capacity(examples.len());
    for e in examples {
        let words: Vec<u32> = (0..(e.features.len() / hidden.max(1)) as u32).collect();
        head.scores(&e.features, &words, &e.mentions, &e.types, &mut scores, &mut w).map_err(|e| Error(e.message))?;
        head.decode(&scores, e.types.len(), &mut kept).map_err(|e| Error(e.message))?;
        out.push(kept.iter().map(|r| (r.head, r.tail, head.labels()[r.label as usize].clone())).collect());
    }
    Ok(out)
}

/// The majority baseline's relations: the most frequent label of a pair
/// in training is `none`, so it keeps none.
pub fn baseline(test: &[Document]) -> Vec<Vec<Triple>> {
    vec![Vec::new(); test.len()]
}

/// Scores `predicted` (the relations of each document) against `gold`:
/// a relation counts when its entities, direction and label are a gold
/// relation's. Gold relations of labels training never gave count as
/// missed.
///
/// # Panics
///
/// When `predicted` and `gold` differ in documents.
pub fn evaluate(gold: &[Document], predicted: &[Vec<Triple>]) -> Metrics {
    assert_eq!(gold.len(), predicted.len(), "one prediction a document");
    let (mut all, mut per_type) = (SpanScore::default(), BTreeMap::<String, SpanScore>::new());
    for (g, p) in gold.iter().zip(predicted) {
        for r in &g.relations {
            per_type.entry(r.2.clone()).or_default().add(1, 0, usize::from(p.contains(r)));
        }
        for r in p {
            per_type.entry(r.2.clone()).or_default().add(0, 1, 0);
        }
        all.add(g.relations.len(), p.len(), p.iter().filter(|r| g.relations.contains(r)).count());
    }
    Metrics::Relations { all: all.finish(), per_type: per_type.into_iter().map(|(k, v)| (k, v.finish())).collect() }
}

/// Per document: (right, found, gold).
fn tallies(gold: &[Document], predicted: &[Vec<Triple>]) -> Vec<[usize; 3]> {
    gold.iter()
        .zip(predicted)
        .map(|(g, p)| [p.iter().filter(|r| g.relations.contains(r)).count(), p.len(), g.relations.len()])
        .collect()
}

fn f1(t: &[[usize; 3]], pick: impl Fn(usize) -> usize) -> f64 {
    let mut sum = [0usize; 3];
    for i in 0..t.len() {
        sum.iter_mut().zip(t[pick(i)]).for_each(|(s, v)| *s += v);
    }
    if sum[1] + sum[2] == 0 { 0.0 } else { 2.0 * sum[0] as f64 / (sum[1] + sum[2]) as f64 }
}

/// Resamples the documents `resamples` times with replacement, seeded,
/// and scores model and baseline by micro relation F1 on the same
/// resample each time.
pub fn bootstrap(
    gold: &[Document],
    model: &[Vec<Triple>],
    baseline: &[Vec<Triple>],
    resamples: usize,
    seed: u64,
) -> Comparison {
    let (m, b) = (tallies(gold, model), tallies(gold, baseline));
    let (ms, bs) = (f1(&m, |i| i), f1(&b, |i| i));
    let mut rng = Rng::new(seed);
    let mut deltas: Vec<f64> = (0..resamples)
        .map(|_| {
            let pick: Vec<usize> = (0..gold.len()).map(|_| rng.below(gold.len())).collect();
            f1(&m, |i| pick[i]) - f1(&b, |i| pick[i])
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let at = |q: f64| if deltas.is_empty() { 0.0 } else { deltas[((deltas.len() - 1) as f64 * q).round() as usize] };
    Comparison { model: ms, baseline: bs, delta: ms - bs, low: at(0.025), high: at(0.975), resamples, seed }
}
