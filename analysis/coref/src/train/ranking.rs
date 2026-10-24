//! Mention ranking (`CorefTrainer.trainRanking`), after Durrett and
//! Klein (EMNLP 2013): for each anaphor, the options are its candidates
//! and starting a new chain, and AdaGrad climbs the log probability of
//! the gold options under a softmax over option scores. The weights are
//! kept as a GIS model whose `link` parameters are the option scores.

use std::collections::HashMap;

use turbo_jvm::random::JavaRandom;
use turbo_jvm::strict_math;
use turbo_train::{JavaString, WritableModel, write_gis_writable};

use super::{Gold, TrainingDocument, now_ms};
use crate::annotator::CorefAnnotator;
use crate::features::{DIFFERENCE, NEW_CHAIN, PRODUCT};
use crate::model::{APART, CorefModel, LINK};

/// The epochs of the default ranking settings.
pub const DEFAULT_EPOCHS: usize = 10;
/// The AdaGrad step size of the default ranking settings.
pub const DEFAULT_LEARNING_RATE: f64 = 0.05;
/// The L2 weight of the default ranking settings.
pub const DEFAULT_L2: f64 = 0.001;
/// The seed of the shuffle that orders anaphors within an epoch.
const SHUFFLE_SEED: i64 = 17;
/// The AdaGrad smoothing term.
const ADAGRAD_EPSILON: f64 = 1e-8;

/// One anaphor's options: feature ids per option, the anaphor's span
/// vector, each option's antecedent vector (`None` for the new chain),
/// and which options are gold.
#[derive(Clone)]
struct Instance {
    options: Vec<Vec<usize>>,
    anaphor: Option<Vec<f32>>,
    antecedents: Vec<Option<Vec<f32>>>,
    gold: Vec<bool>,
}

/// One weight per binary feature and three dense blocks.
#[derive(Clone)]
struct Weights {
    sparse: Vec<f64>,
    product: Vec<f64>,
    difference: Vec<f64>,
    new_chain: Vec<f64>,
}

impl Weights {
    fn new(features: usize, dimension: usize) -> Weights {
        Weights {
            sparse: vec![0.0; features],
            product: vec![0.0; dimension],
            difference: vec![0.0; dimension],
            new_chain: vec![0.0; dimension],
        }
    }
}

fn ids(names: &[String], feature_ids: &mut HashMap<String, usize>, order: &mut Vec<String>) -> Vec<usize> {
    names
        .iter()
        .map(|n| {
            *feature_ids.entry(n.clone()).or_insert_with(|| {
                order.push(n.clone());
                order.len() - 1
            })
        })
        .collect()
}

fn instances(
    doc: &TrainingDocument<'_>,
    annotator: &CorefAnnotator,
    feature_ids: &mut HashMap<String, usize>,
    order: &mut Vec<String>,
) -> Result<Vec<Instance>, String> {
    let Some(prepared) = doc.input.prepare()? else { return Ok(Vec::new()) };
    let mut resolver = annotator.resolver(&prepared)?;
    resolver.resolve_precise();
    let gold = Gold::of(&doc.gold, &resolver)?;
    let mut out = Vec::new();
    for j in 0..resolver.mentions.len() {
        if gold.skips(&mut resolver, j) {
            continue;
        }
        let candidates = resolver.ranker_candidates(j);
        let mut gold_options: Vec<bool> =
            candidates.iter().map(|&i| gold.chains[j].is_some() && gold.chains[i] == gold.chains[j]).collect();
        let nearest = candidates.iter().zip(&gold_options).filter(|(_, g)| **g).map(|(&i, _)| i).max();
        let any_gold = nearest.is_some();
        if !any_gold && gold.unlearnable(&resolver, j) {
            continue;
        }
        let mut options = Vec::with_capacity(candidates.len() + 1);
        let mut antecedents = Vec::with_capacity(candidates.len() + 1);
        for &i in &candidates {
            let pair = resolver.features(i, j)?;
            // Inference expands the vectors into real-valued features and
            // refuses a non-finite one; training refuses it too.
            pair.expand()?;
            options.push(ids(&pair.names, feature_ids, order));
            antecedents.push(pair.antecedent);
        }
        let new_chain = resolver.new_chain_features(j)?;
        options.push(ids(&new_chain.names, feature_ids, order));
        antecedents.push(None);
        gold_options.push(!any_gold);
        out.push(Instance { options, anaphor: new_chain.anaphor, antecedents, gold: gold_options });
        if let Some(a) = nearest {
            resolver.clusters.union(a, j);
        }
    }
    Ok(out)
}

fn dense(w: &Weights, u: &[f32], v: Option<&[f32]>) -> f64 {
    let mut score = 0.0;
    for d in 0..u.len() {
        match v {
            Some(v) => {
                score += w.product[d] * u[d] as f64 * v[d] as f64 + w.difference[d] * (u[d] - v[d]).abs() as f64;
            }
            None => score += w.new_chain[d] * u[d] as f64,
        }
    }
    score
}

fn update(weights: &mut [f64], squares: &mut [f64], f: usize, gradient: f64, rate: f64, l2: f64) {
    let g = gradient - l2 * weights[f];
    squares[f] += g * g;
    weights[f] += rate * g / (squares[f].sqrt() + ADAGRAD_EPSILON);
}

/// One AdaGrad step up the log probability of the gold options.
fn step(x: &Instance, w: &mut Weights, sq: &mut Weights, gradient: &mut [f64], rate: f64, l2: f64) {
    let scores: Vec<f64> = x
        .options
        .iter()
        .enumerate()
        .map(|(o, features)| {
            let sparse: f64 = features.iter().map(|&f| w.sparse[f]).sum();
            sparse + x.anaphor.as_deref().map_or(0.0, |u| dense(w, u, x.antecedents[o].as_deref()))
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let gold_max = scores.iter().zip(&x.gold).filter(|(_, g)| **g).map(|(&s, _)| s).fold(f64::NEG_INFINITY, f64::max);
    let all: f64 = scores.iter().map(|s| strict_math::exp(s - max)).sum();
    let gold: f64 = scores.iter().zip(&x.gold).filter(|(_, g)| **g).map(|(s, _)| strict_math::exp(s - gold_max)).sum();
    let dimension = x.anaphor.as_ref().map_or(0, Vec::len);
    let (mut pg, mut dg, mut ng) = (vec![0.0; dimension], vec![0.0; dimension], vec![0.0; dimension]);
    let mut touched = Vec::new();
    let mut queued = vec![false; gradient.len()];
    for (o, features) in x.options.iter().enumerate() {
        let expected = strict_math::exp(scores[o] - max) / all;
        let gold_expected = if x.gold[o] { strict_math::exp(scores[o] - gold_max) / gold } else { 0.0 };
        let delta = gold_expected - expected;
        if delta == 0.0 {
            continue;
        }
        for &f in features {
            if !queued[f] {
                queued[f] = true;
                touched.push(f);
            }
            gradient[f] += delta;
        }
        if let Some(u) = &x.anaphor {
            for d in 0..dimension {
                match &x.antecedents[o] {
                    Some(v) => {
                        pg[d] += delta * u[d] as f64 * v[d] as f64;
                        dg[d] += delta * (u[d] - v[d]).abs() as f64;
                    }
                    None => ng[d] += delta * u[d] as f64,
                }
            }
        }
    }
    for f in touched {
        update(&mut w.sparse, &mut sq.sparse, f, gradient[f], rate, l2);
        gradient[f] = 0.0;
    }
    for d in 0..dimension {
        update(&mut w.product, &mut sq.product, d, pg[d], rate, l2);
        update(&mut w.difference, &mut sq.difference, d, dg[d], rate, l2);
        update(&mut w.new_chain, &mut sq.new_chain, d, ng[d], rate, l2);
    }
}

/// The ranker as a GIS model: every predicate has one `link` parameter.
struct RankerModel {
    outcomes: [JavaString; 2],
    predicates: Vec<JavaString>,
    parameters: Vec<f64>,
}

impl WritableModel for RankerModel {
    fn outcome_labels(&self) -> &[JavaString] {
        &self.outcomes
    }

    fn pred_labels(&self) -> &[JavaString] {
        &self.predicates
    }

    fn pattern(&self, _: usize) -> &[u32] {
        &[0]
    }

    fn parameters(&self, p: usize) -> &[f64] {
        std::slice::from_ref(&self.parameters[p])
    }
}

/// Trains a ranking model: `epochs` passes in an order shuffled by
/// `java.util.Random(17)`, AdaGrad with `learning_rate`, L2 weight `l2`.
pub fn train_ranking(
    language: &str,
    documents: &[TrainingDocument<'_>],
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    annotator: &CorefAnnotator,
) -> Result<CorefModel, String> {
    if turbo_unicode::java::is_blank(language) {
        return Err("languageCode must not be null or blank".into());
    }
    if epochs == 0 {
        return Err("epochs must be positive: 0".into());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(format!("learningRate must be positive and finite: {learning_rate}"));
    }
    if !l2.is_finite() || l2 < 0.0 {
        return Err(format!("l2 must be finite and not negative: {l2}"));
    }
    let mut feature_ids = HashMap::new();
    let mut names = Vec::new();
    let mut all = Vec::new();
    let dimension = annotator.token_vector_dimension();
    for doc in documents {
        all.extend(instances(doc, annotator, &mut feature_ids, &mut names)?);
    }
    if all.is_empty() {
        return Err("training data produced no ranking instances".into());
    }
    let mut weights = Weights::new(names.len(), dimension);
    let mut squares = Weights::new(names.len(), dimension);
    let mut gradient = vec![0.0; names.len()];
    let mut random = JavaRandom::new(SHUFFLE_SEED);
    for _ in 0..epochs {
        random.shuffle(&mut all);
        for x in &all {
            step(x, &mut weights, &mut squares, &mut gradient, learning_rate, l2);
        }
    }
    let mut predicates: Vec<JavaString> = names.iter().map(|n| JavaString::new(n)).collect();
    let mut parameters = weights.sparse.clone();
    for d in 0..dimension {
        for (prefix, w) in
            [(PRODUCT, &weights.product), (DIFFERENCE, &weights.difference), (NEW_CHAIN, &weights.new_chain)]
        {
            predicates.push(JavaString::new(&format!("{prefix}{d}")));
            parameters.push(w[d]);
        }
    }
    let model = RankerModel { outcomes: [JavaString::new(LINK), JavaString::new(APART)], predicates, parameters };
    let entry = write_gis_writable(&model).map_err(|e| e.to_string())?;
    CorefModel::package(language, entry, true, dimension, None, now_ms())
}
