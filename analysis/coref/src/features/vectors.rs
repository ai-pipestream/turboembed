//! The vector features: head word similarity, contextual span vectors,
//! and the expansion of an option into model names and values.

use super::{DIFFERENCE, Features, NEW_CHAIN, PRODUCT};
use crate::sieve::Resolver;

/// The value of a similarity feature when a vector is missing.
const UNKNOWN_SIMILARITY: &str = "unknown";

/// The cosine of two vectors, `-2` when either is all zeros.
pub(crate) fn cosine(u: &[f32], v: &[f32]) -> f64 {
    let dot = turbo_analysis::math::dot_f64(u, v);
    let uu = turbo_analysis::math::sum_sq_f64(u);
    let vv = turbo_analysis::math::sum_sq_f64(v);
    if uu == 0.0 || vv == 0.0 { -2.0 } else { dot / (uu * vv).sqrt() }
}

/// The componentwise mean of some vectors, summed in `f64`.
fn mean<'v>(vectors: impl ExactSizeIterator<Item = &'v [f32]>, dimension: usize) -> Vec<f32> {
    let n = vectors.len();
    let mut sum = vec![0.0f64; dimension];
    for v in vectors {
        for d in 0..dimension {
            sum[d] += v[d] as f64;
        }
    }
    sum.iter().map(|s| (s / n as f64) as f32).collect()
}

/// A contextual cosine's tenth-sized bucket, `neg` or `unknown`.
fn context_bucket(cosine: f64) -> String {
    if cosine < -1.0 {
        UNKNOWN_SIMILARITY.into()
    } else if cosine < 0.0 {
        "neg".into()
    } else {
        ((cosine * 10.0) as i32).min(9).to_string()
    }
}

/// A word-vector cosine's range, or `unknown`.
fn similarity_bucket(cosine: f64) -> &'static str {
    if cosine < -1.0 {
        UNKNOWN_SIMILARITY
    } else if cosine < 0.3 {
        "low"
    } else if cosine < 0.5 {
        "mid"
    } else if cosine < 0.7 {
        "high"
    } else if cosine < 0.9 {
        "higher"
    } else {
        "same"
    }
}

impl Resolver<'_> {
    /// The validated vector of a word, cached; `Err` when the provider
    /// breaks its contract.
    fn word_vector(&mut self, word: &str) -> Result<Option<Vec<f32>>, String> {
        if let Some(cached) = self.vector_cache.get(word) {
            return Ok(cached.clone());
        }
        let vectors = self.vectors.expect("similarity features need word vectors");
        let dimension = vectors.dimension();
        let found = match vectors.vector(word) {
            None => None,
            Some(v) => {
                if v.len() != dimension {
                    return Err(format!(
                        "word vectors returned dimension {} for {word}, expected {dimension}",
                        v.len()
                    ));
                }
                if let Some(d) = v.iter().position(|x| !x.is_finite()) {
                    return Err(format!("word vectors returned a non-finite value for {word} at dimension {d}"));
                }
                Some(v.to_vec())
            }
        };
        self.vector_cache.insert(word.to_owned(), found.clone());
        Ok(found)
    }

    fn word_cosine(&mut self, a: &str, b: &str) -> Result<f64, String> {
        match (self.word_vector(a)?, self.word_vector(b)?) {
            (Some(u), Some(v)) => Ok(cosine(&u, &v)),
            _ => Ok(-2.0),
        }
    }

    /// `sim`, its kinds conjunction, and `clusterSim`: the best head
    /// similarity over the candidate cluster's heads.
    pub(super) fn similarity_features(
        &mut self,
        i: usize,
        j: usize,
        kinds: &str,
        f: &mut Vec<String>,
    ) -> Result<(), String> {
        let (ana, ant) = (self.mentions[j].head.clone(), self.mentions[i].head.clone());
        let similarity = format!("sim={}", similarity_bucket(self.word_cosine(&ana, &ant)?));
        f.push(similarity.clone());
        f.push(format!("{kinds}|{similarity}"));
        let heads: Vec<String> = self.clusters.heads(i).iter().cloned().collect();
        let mut best = UNKNOWN_SIMILARITY;
        let mut best_cosine = -2.0;
        for head in heads {
            let c = self.word_cosine(&ana, &head)?;
            if c > best_cosine {
                best_cosine = c;
                best = similarity_bucket(c);
            }
        }
        f.push(format!("clusterSim={best}"));
        Ok(())
    }

    /// The mean token vector of mention `m`, computed once.
    pub(super) fn span_vector(&mut self, m: usize) -> Vec<f32> {
        if self.span_vectors[m].is_none() {
            let tokens = self.token_vectors.as_ref().expect("span vectors need token vectors");
            let mention = &self.mentions[m];
            let span = &tokens[mention.first..=mention.last];
            let v = mean(span.iter().map(Vec::as_slice), tokens[0].len());
            self.span_vectors[m] = Some(v);
        }
        self.span_vectors[m].clone().expect("computed above")
    }

    fn cluster_vector(&mut self, i: usize) -> Vec<f32> {
        let members = self.clusters.members(i).to_vec();
        let spans: Vec<Vec<f32>> = members.iter().map(|&m| self.span_vector(m)).collect();
        let dimension = spans[0].len();
        mean(spans.iter().map(Vec::as_slice), dimension)
    }

    /// `ctx`, its kinds conjunction, and `ctxCluster` from the span vectors.
    pub(super) fn context_features(
        &mut self,
        i: usize,
        j: usize,
        kinds: &str,
        mut f: Vec<String>,
    ) -> Result<Features, String> {
        let u = self.span_vector(j);
        let v = self.span_vector(i);
        let context = format!("ctx={}", context_bucket(cosine(&u, &v)));
        f.push(context.clone());
        f.push(format!("{kinds}|{context}"));
        let cluster = self.cluster_vector(i);
        f.push(format!("ctxCluster={}", context_bucket(cosine(&u, &cluster))));
        Ok(Features { names: f, anaphor: Some(u), antecedent: Some(v) })
    }
}

impl Features {
    /// The dimension of the real-valued features, zero without vectors.
    pub fn dimension(&self) -> usize {
        self.anaphor.as_ref().map_or(0, Vec::len)
    }

    /// The names and values a model evaluates: the binary features, then
    /// per dimension the product and difference of a pair, or the
    /// anaphor's value for the new-chain option. Values are `None` when
    /// every feature counts one; `Err` when a value overflows `f32`.
    pub fn expand(&self) -> Result<(Vec<String>, Option<Vec<f32>>), String> {
        let dimension = self.dimension();
        if dimension == 0 {
            return Ok((self.names.clone(), None));
        }
        let u = self.anaphor.as_ref().expect("dimension is positive");
        let mut names = self.names.clone();
        let mut values = vec![1.0f32; names.len()];
        for d in 0..dimension {
            match &self.antecedent {
                Some(v) => {
                    names.push(format!("{PRODUCT}{d}"));
                    values.push(dense(u[d] as f64 * v[d] as f64, PRODUCT, d)?);
                    names.push(format!("{DIFFERENCE}{d}"));
                    values.push(dense((u[d] as f64 - v[d] as f64).abs(), DIFFERENCE, d)?);
                }
                None => {
                    names.push(format!("{NEW_CHAIN}{d}"));
                    values.push(u[d]);
                }
            }
        }
        Ok((names, Some(values)))
    }
}

fn dense(value: f64, feature: &str, d: usize) -> Result<f32, String> {
    let v = value as f32;
    if v.is_finite() { Ok(v) } else { Err(format!("dense feature {feature}{d} is not finite")) }
}
