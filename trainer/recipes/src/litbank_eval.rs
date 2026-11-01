//! LitBank coreference head spec, encoder feature extraction, ranker
//! prediction, singleton baseline, CoNLL (MUC, B-cubed, CEAF-e) scoring,
//! and seeded paired bootstrap (#1527).

use std::collections::{BTreeMap, BTreeSet};

use turbo::head::CorefRanker;
use turbo::head::coref::chains as decode_chains;
use turbo::head::tagger::Work;
use turbo::manifest::{
    Adapter, Bilstm, COREF_LABELS, CorefShape, Decode, HeadKind, HeadSpec, Scheme, SpanShape, WordPooling,
};
use turbo_tagger::WordEncoder;
pub use turbo_trainer::CorefExample;
use turbo_trainer::rng::Rng;

use crate::corpus::CorpusRecord;
use crate::eval::{Comparison, Metrics, Prf};
use crate::litbank::CorefDoc;
use crate::run::PREDICT_CHUNK;
use crate::{Error, Result};

/// Alias for [`CorefExample`].
pub type CorefDocument = CorefExample;

/// One coreference cluster: a list of `(first_word, last_word)` spans.
pub type Cluster = Vec<(u32, u32)>;

/// All coreference clusters of one document.
pub type DocClusters = Vec<Cluster>;

/// Default maximum words in a mention span (`span.max_words`).
pub const DEFAULT_MAX_WORDS: u32 = 64;
/// Default span width embedding dimension (`span.width_dim`).
pub const DEFAULT_WIDTH_DIM: u32 = 16;
/// Default mention projection width (`coref.dim`).
pub const DEFAULT_MENTION_DIM: u32 = 128;
/// Default antecedent window size (`coref.antecedents`).
pub const DEFAULT_ANTECEDENTS: u32 = 128;

/// Architecture options for a [`HeadKind::CorefRanker`] head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorefOptions {
    pub pooling: WordPooling,
    pub bilstm: Option<(u32, u32)>,
    pub span: SpanShape,
    pub coref: CorefShape,
}

impl Default for CorefOptions {
    fn default() -> Self {
        Self {
            pooling: WordPooling::First,
            bilstm: None,
            span: SpanShape { max_words: DEFAULT_MAX_WORDS, width_dim: DEFAULT_WIDTH_DIM },
            coref: CorefShape { dim: DEFAULT_MENTION_DIM, antecedents: DEFAULT_ANTECEDENTS },
        }
    }
}

/// Builds and validates the `coref` [`HeadSpec`] on `turboembed:coref`,
/// carrying the licence class of `corpus` (`CorpusRecord::class`).
///
/// # Errors
///
/// Refuses a head spec rejected by `HeadSpec::validate` or an unknown/refused
/// corpus licence.
pub fn spec(corpus: &CorpusRecord, o: &CorefOptions) -> Result<HeadSpec> {
    let spec = HeadSpec {
        name: "coref".into(),
        layer: "turboembed:coref".into(),
        kind: HeadKind::CorefRanker,
        labels: COREF_LABELS.iter().map(|&s| s.to_owned()).collect(),
        scheme: Scheme::Tags,
        word_pooling: o.pooling,
        adapter: o.bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::Antecedent,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: Some(o.span),
        coref: Some(o.coref),
        relation: None,
        layer_mix: None,
        license_class: Some(corpus.class()?),
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.coref".into(),
    };
    spec.validate("head").map_err(|e| Error(e.message))?;
    Ok(spec)
}

/// Names the ranker's tensors with the shapes `turbo::head::coref::tensors`
/// expects, ready for `crate::safetensors::write_with`.
pub fn named(spec: &HeadSpec, hidden: usize, values: Vec<(String, Vec<f32>)>) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let shapes: BTreeMap<String, Vec<u64>> = turbo::head::coref::tensors(spec, hidden).into_iter().collect();
    values
        .into_iter()
        .map(|(suffix, v)| {
            let shape = shapes.get(&suffix).cloned().unwrap_or_else(|| vec![v.len() as u64]);
            (format!("{}.{suffix}", spec.tensor_prefix), shape, v)
        })
        .collect()
}

/// Extracts per-word encoder features `[words, hidden]` for each document in
/// `docs`, reading through `WordEncoder::features` just as `CorefTagger` does
/// at inference.
pub fn features(enc: &mut WordEncoder, docs: &[CorefDoc], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(docs.len());
    for chunk in docs.chunks(PREDICT_CHUNK) {
        let words: Vec<Vec<&str>> = chunk.iter().map(|d| d.words.iter().map(String::as_str).collect()).collect();
        out.extend(enc.features(&words, pooling).map_err(|e| Error(e.message))?);
    }
    Ok(out)
}

/// Builds [`CorefExample`]s from `docs` and their per-word `features`,
/// filtering out mentions wider than `max_words` (returning the number of
/// mentions dropped).
pub fn examples(docs: &[CorefDoc], features: Vec<Vec<f32>>, max_words: u32) -> (Vec<CorefExample>, usize) {
    let mut out = Vec::with_capacity(docs.len());
    let mut dropped = 0usize;
    for (doc, features) in docs.iter().zip(features) {
        let mut mentions = Vec::with_capacity(doc.mentions.len());
        let mut chains = Vec::with_capacity(doc.chains.len());
        for (&(first, last), &chain) in doc.mentions.iter().zip(&doc.chains) {
            if last - first < max_words {
                mentions.push((first, last));
                chains.push(chain);
            } else {
                dropped += 1;
            }
        }
        out.push(CorefExample { features, mentions, chains });
    }
    (out, dropped)
}

/// Predicts coreference clusters for each document in `docs` using `head`
/// over the document's gold mention spans, matching `CorefTagger::resolve`:
/// mentions up to `head.max_words()` are scored and linked by highest-scoring
/// antecedent, while any wider mention stands in its own singleton cluster.
pub fn predict(head: &CorefRanker, docs: &[CorefDoc], features: &[Vec<f32>]) -> Result<Vec<DocClusters>> {
    let max_words = head.spec().span.expect("a coref ranker has a span shape").max_words;
    let (mut g, mut scores, mut antecedents, mut w) = (Vec::new(), Vec::new(), Vec::new(), Work::default());
    let mut out = Vec::with_capacity(docs.len());

    for (doc, f) in docs.iter().zip(features) {
        let words: Vec<u32> = (0..doc.words.len() as u32).collect();
        let mut scored = Vec::with_capacity(doc.mentions.len());
        let mut wide = Vec::new();
        for &(first, last) in &doc.mentions {
            if last - first < max_words {
                scored.push((first, last));
            } else {
                wide.push((first, last));
            }
        }
        head.mentions(f, &words, &scored, &mut g, &mut w).map_err(|e| Error(e.message))?;
        head.scores(&g, &mut scores).map_err(|e| Error(e.message))?;
        head.decode(&scores, &mut antecedents).map_err(|e| Error(e.message))?;
        let chain_ids = decode_chains(&antecedents).map_err(|e| Error(e.message))?;

        let mut by_chain: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
        for (span, cid) in scored.into_iter().zip(chain_ids) {
            by_chain.entry(cid).or_default().push(span);
        }
        let mut clusters: Vec<Vec<(u32, u32)>> = by_chain.into_values().collect();
        for span in wide {
            clusters.push(vec![span]);
        }
        for c in &mut clusters {
            c.sort();
        }
        clusters.sort_by_key(|c| c[0]);
        out.push(clusters);
    }
    Ok(out)
}

/// Deterministic singleton baseline: every gold mention in `docs` forms its
/// own singleton cluster (every mention chooses the dummy root `None`).
pub fn baseline(docs: &[CorefDoc]) -> Vec<DocClusters> {
    docs.iter().map(|d| d.mentions.iter().map(|&span| vec![span]).collect()).collect()
}

/// Per-document sufficient statistics for micro-averaged CoNLL coreference
/// metrics (MUC, B-cubed, CEAF-e).
#[derive(Debug, Clone, Copy, Default)]
struct DocTally {
    mentions: usize,
    gold_clusters: usize,
    pred_clusters: usize,
    muc_r_num: usize,
    muc_r_den: usize,
    muc_p_num: usize,
    muc_p_den: usize,
    b3_r_num: f64,
    b3_r_den: usize,
    b3_p_num: f64,
    b3_p_den: usize,
    ceaf_e_num: f64,
}

impl DocTally {
    fn add(&mut self, other: &DocTally) {
        self.mentions += other.mentions;
        self.gold_clusters += other.gold_clusters;
        self.pred_clusters += other.pred_clusters;
        self.muc_r_num += other.muc_r_num;
        self.muc_r_den += other.muc_r_den;
        self.muc_p_num += other.muc_p_num;
        self.muc_p_den += other.muc_p_den;
        self.b3_r_num += other.b3_r_num;
        self.b3_r_den += other.b3_r_den;
        self.b3_p_num += other.b3_p_num;
        self.b3_p_den += other.b3_p_den;
        self.ceaf_e_num += other.ceaf_e_num;
    }

    fn finish(&self, documents: usize) -> Metrics {
        let muc = prf(self.muc_r_num as f64, self.muc_r_den as f64, self.muc_p_num as f64, self.muc_p_den as f64);
        let b_cubed = prf(self.b3_r_num, self.b3_r_den as f64, self.b3_p_num, self.b3_p_den as f64);
        let ceaf_e = prf(self.ceaf_e_num, self.gold_clusters as f64, self.ceaf_e_num, self.pred_clusters as f64);
        let conll_f1 = (muc.f1 + b_cubed.f1 + ceaf_e.f1) / 3.0;
        Metrics::Coref {
            documents,
            mentions: self.mentions,
            gold_clusters: self.gold_clusters,
            predicted_clusters: self.pred_clusters,
            muc,
            b_cubed,
            ceaf_e,
            conll_f1,
        }
    }
}

fn prf(r_num: f64, r_den: f64, p_num: f64, p_den: f64) -> Prf {
    if r_den == 0.0 && p_den == 0.0 {
        return Prf { precision: 1.0, recall: 1.0, f1: 1.0 };
    }
    let recall = if r_den == 0.0 { 0.0 } else { r_num / r_den };
    let precision = if p_den == 0.0 { 0.0 } else { p_num / p_den };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    Prf { precision, recall, f1 }
}

fn doc_tally(gold: &[Vec<(u32, u32)>], pred: &[Vec<(u32, u32)>]) -> DocTally {
    let key: Vec<BTreeSet<(u32, u32)>> =
        gold.iter().map(|c| c.iter().copied().collect::<BTreeSet<_>>()).filter(|c| !c.is_empty()).collect();
    let resp: Vec<BTreeSet<(u32, u32)>> =
        pred.iter().map(|c| c.iter().copied().collect::<BTreeSet<_>>()).filter(|c| !c.is_empty()).collect();

    let (muc_r_num, muc_r_den) = muc_side(&key, &resp);
    let (muc_p_num, muc_p_den) = muc_side(&resp, &key);
    let (b3_r_num, b3_r_den) = b_cubed_side(&key, &resp);
    let (b3_p_num, b3_p_den) = b_cubed_side(&resp, &key);

    let mut sim = vec![0.0; key.len() * resp.len()];
    for (i, k) in key.iter().enumerate() {
        for (j, r) in resp.iter().enumerate() {
            let common = k.intersection(r).count();
            if common > 0 {
                sim[i * resp.len() + j] = (2 * common) as f64 / (k.len() + r.len()) as f64;
            }
        }
    }
    let ceaf_e_num = maximum_assignment(key.len(), resp.len(), &sim);

    DocTally {
        mentions: b3_r_den,
        gold_clusters: key.len(),
        pred_clusters: resp.len(),
        muc_r_num,
        muc_r_den,
        muc_p_num,
        muc_p_den,
        b3_r_num,
        b3_r_den,
        b3_p_num,
        b3_p_den,
        ceaf_e_num,
    }
}

fn muc_side(a: &[BTreeSet<(u32, u32)>], b: &[BTreeSet<(u32, u32)>]) -> (usize, usize) {
    let mut num = 0usize;
    let mut den = 0usize;
    for ca in a {
        let mut covered = 0usize;
        let mut parts = 0usize;
        for cb in b {
            let common = ca.intersection(cb).count();
            if common > 0 {
                parts += 1;
                covered += common;
            }
        }
        parts += ca.len() - covered;
        num += ca.len() - parts;
        den += ca.len() - 1;
    }
    (num, den)
}

fn b_cubed_side(a: &[BTreeSet<(u32, u32)>], b: &[BTreeSet<(u32, u32)>]) -> (f64, usize) {
    let mut num = 0.0;
    let mut den = 0usize;
    for ca in a {
        den += ca.len();
        for cb in b {
            let common = ca.intersection(cb).count();
            if common > 0 {
                num += (common * common) as f64 / ca.len() as f64;
            }
        }
    }
    (num, den)
}

/// Kuhn-Munkres (Hungarian) maximum weight bipartite matching over an
/// `rows x cols` non-negative score matrix.
fn maximum_assignment(rows: usize, cols: usize, scores: &[f64]) -> f64 {
    let n = rows.max(cols);
    if n == 0 {
        return 0.0;
    }
    let mut max_score = 0.0_f64;
    for &s in scores {
        if s > max_score {
            max_score = s;
        }
    }
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let s = if i0 <= rows && j <= cols { scores[(i0 - 1) * cols + (j - 1)] } else { 0.0 };
                    let cost = max_score - s;
                    let cur = cost - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut total = 0.0;
    for j in 1..=cols {
        let i = p[j];
        if i >= 1 && i <= rows {
            total += scores[(i - 1) * cols + (j - 1)];
        }
    }
    total
}

/// Scores `predicted` coreference clusters against `gold` documents using
/// micro-averaged MUC, B-cubed, CEAF-e, and their mean CoNLL F1.
pub fn evaluate_coref(gold: &[CorefDoc], predicted: &[DocClusters]) -> Result<Metrics> {
    if gold.len() != predicted.len() {
        return Err(Error(format!(
            "coref evaluation expected {} document predictions, got {}",
            gold.len(),
            predicted.len()
        )));
    }
    let mut sum = DocTally::default();
    for (g, p) in gold.iter().zip(predicted) {
        sum.add(&doc_tally(&g.clusters, p));
    }
    Ok(sum.finish(gold.len()))
}

/// Seeded paired bootstrap of `model` against `baseline` over `gold`
/// documents, resampling documents with replacement `resamples` times.
pub fn bootstrap_coref(
    gold: &[CorefDoc],
    model: &[DocClusters],
    baseline: &[DocClusters],
    resamples: usize,
    seed: u64,
) -> Result<Comparison> {
    if gold.len() != model.len() || gold.len() != baseline.len() {
        return Err(Error("coref bootstrap requires equal document counts".into()));
    }
    let m_tallies: Vec<DocTally> = gold.iter().zip(model).map(|(g, p)| doc_tally(&g.clusters, p)).collect();
    let b_tallies: Vec<DocTally> = gold.iter().zip(baseline).map(|(g, p)| doc_tally(&g.clusters, p)).collect();

    let score_pick = |tallies: &[DocTally], pick: &dyn Fn(usize) -> usize| -> f64 {
        let mut sum = DocTally::default();
        for i in 0..tallies.len() {
            sum.add(&tallies[pick(i)]);
        }
        sum.finish(tallies.len()).score()
    };

    let ms = score_pick(&m_tallies, &|i| i);
    let bs = score_pick(&b_tallies, &|i| i);
    let mut rng = Rng::new(seed);
    let n = gold.len();
    let mut deltas: Vec<f64> = (0..resamples)
        .map(|_| {
            if n == 0 {
                return 0.0;
            }
            let pick: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            score_pick(&m_tallies, &|i| pick[i]) - score_pick(&b_tallies, &|i| pick[i])
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let at = |q: f64| {
        if deltas.is_empty() { 0.0 } else { deltas[((deltas.len() - 1) as f64 * q).round() as usize] }
    };
    Ok(Comparison { model: ms, baseline: bs, delta: ms - bs, low: at(0.025), high: at(0.975), resamples, seed })
}
