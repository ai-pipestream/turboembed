//! A coreference ranker head (`heads[]` of kind HEAD_COREF_RANKER): for
//! each mention of a document, its antecedent among the mentions before
//! it, or none (Lee et al. 2017, with the mentions given, as a span
//! classifier head finds them, rather than pruned from every span).
//!
//! A mention from word i to word j is read as a span classifier reads a
//! span (span.rs): its first and last word vectors and a learned
//! embedding of its width, joined; one linear layer and tanh make its
//! vector g, `coref.dim` wide. A pair scores
//!
//! ```text
//! s(a, b) = g_a . (P g_b) + distance[bucket(a - b)]
//! ```
//!
//! for mention a and an earlier mention b, the distance counted in
//! mentions; the dummy antecedent (none) scores 0. Mention a is scored
//! against the `coref.antecedents` mentions before it, nearest first.

use crate::bundle::Bundle;
use crate::manifest::{HeadKind, HeadSpec};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::tagger::load_tensors;
use super::words::{self, Take, WordReader, Work};

/// The distance buckets, by the smallest distance each holds: 1, 2, 3,
/// 4, 5 to 7, 8 to 15, 16 to 31, 32 to 63, and 64 or more.
pub const BUCKETS: [u32; 9] = [1, 2, 3, 4, 5, 8, 16, 32, 64];

/// The bucket of a distance of `d` mentions, `d` at least 1.
pub fn bucket(d: u32) -> usize {
    BUCKETS.iter().rposition(|&b| d >= b).unwrap_or(0)
}

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let (mut out, width) = words::tensors(spec, hidden);
    let (max, e) = spec.span.map_or((0, 0), |s| (s.max_words as u64, s.width_dim as u64));
    let d = spec.coref.map_or(0, |c| c.dim as u64);
    out.push(("span.width".into(), vec![max, e]));
    // Saturates where `check_width` refuses, so the size check after it
    // refuses the shape too.
    out.push(("mention.weight".into(), vec![d, width.saturating_mul(2).saturating_add(e)]));
    out.push(("mention.bias".into(), vec![d]));
    out.push(("pair.weight".into(), vec![d, d]));
    out.push(("distance".into(), vec![BUCKETS.len() as u64]));
    out
}

/// Refuses an encoder `hidden` wide whose mention input, `2 * width +
/// width_dim`, does not fit in a `u64`, before any shape is computed.
pub fn check_width(spec: &HeadSpec, hidden: usize) -> Result<()> {
    let hashed = spec.hash_embed.map_or(0, |h| 4 * h.dim as u64);
    let e = spec.span.map_or(0, |s| s.width_dim as u64);
    let fits = (hidden as u64).checked_add(hashed).is_some_and(|_| {
        let (_, width) = words::tensors(spec, hidden);
        width.checked_mul(2).and_then(|w| w.checked_add(e)).is_some()
    });
    if fits { Ok(()) } else { Err(invalid(format!("head {:?}: an encoder {hidden} wide is too wide", spec.name))) }
}

/// A loaded coreference ranker. Read-only after loading, so one head
/// serves many threads, each with its own [`Work`].
#[derive(Debug)]
pub struct CorefRanker {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    width_emb: Vec<f32>,
    mention_w: Vec<f32>,
    mention_b: Vec<f32>,
    pair: Vec<f32>,
    distance: Vec<f32>,
    max_words: usize,
    dim: usize,
    antecedents: usize,
}

impl CorefRanker {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<CorefRanker> {
        let l = load_tensors(bundle, name, HeadKind::CorefRanker, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::CorefRanker {
            return Err(invalid(format!("head {:?} is not a coreference ranker", spec.name)));
        }
        spec.validate("head")?;
        let (Some(span), Some(coref)) = (spec.span, spec.coref) else {
            return Err(invalid(format!("head {:?}: no span or coref", spec.name)));
        };
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        check_width(spec, hidden)?;
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let head = CorefRanker {
            width_emb: t.take("span.width")?,
            mention_w: t.take("mention.weight")?,
            mention_b: t.take("mention.bias")?,
            pair: t.take("pair.weight")?,
            distance: t.take("distance")?,
            spec: spec.clone(),
            hidden,
            reader,
            max_words: span.max_words as usize,
            dim: coref.dim as usize,
            antecedents: coref.antecedents as usize,
        };
        let all = [&head.width_emb, &head.mention_w, &head.mention_b, &head.pair, &head.distance];
        if !all.iter().all(|v| v.iter().all(|x| x.is_finite())) {
            return Err(Error::new(INVALID_ARGUMENT, format!("head {:?}: a weight is not finite", spec.name)));
        }
        Ok(head)
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The width of a mention vector, `coref.dim`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The most antecedents a mention is scored against.
    pub fn antecedents(&self) -> usize {
        self.antecedents
    }

    /// The vectors [mentions, dim] of `mentions` into `out`, each its
    /// first and last word, from one sequence's subword states and word
    /// map as a token tagger reads them (`TokenTagger::scores_forms`).
    /// The mentions are in order of first word then last, none twice,
    /// each in the sequence and at most `span.max_words` words long.
    /// Returns the number of words.
    pub fn mentions(
        &self,
        states: &[f32],
        words: &[u32],
        mentions: &[(u32, u32)],
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<usize> {
        out.clear();
        let n = self.reader.read(states, words, None, w)?;
        check_mentions(mentions, n, self.max_words)?;
        let (width, e, d) = (self.reader.width(), self.width_emb.len() / self.max_words, self.dim);
        let cols = 2 * width + e;
        let mut x = vec![0f32; cols];
        out.reserve(mentions.len() * d);
        for &(i, j) in mentions {
            let (i, j) = (i as usize, j as usize);
            x[..width].copy_from_slice(&w.pooled[i * width..][..width]);
            x[width..2 * width].copy_from_slice(&w.pooled[j * width..][..width]);
            x[2 * width..].copy_from_slice(&self.width_emb[(j - i) * e..][..e]);
            for o in 0..d {
                let row = &self.mention_w[o * cols..][..cols];
                out.push((self.mention_b[o] + row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>()).tanh());
            }
        }
        Ok(n)
    }

    /// The pair scores [mentions, antecedents] of mention vectors `g`
    /// [mentions, dim] into `out`: row a holds mention a's scores against
    /// mentions a - 1, a - 2, and so on, and `f32::NEG_INFINITY` past the
    /// first mention.
    pub fn scores(&self, g: &[f32], out: &mut Vec<f32>) -> Result<()> {
        let (d, k) = (self.dim, self.antecedents);
        if !g.len().is_multiple_of(d) {
            return Err(Error::new(INVALID_SHAPE, format!("{} values are not mention vectors of {d}", g.len())));
        }
        let m = g.len() / d;
        // P g_b for every mention b, once.
        let mut pg = vec![0f32; m * d];
        for b in 0..m {
            let gb = &g[b * d..][..d];
            for (o, v) in pg[b * d..][..d].iter_mut().enumerate() {
                *v = self.pair[o * d..][..d].iter().zip(gb).map(|(p, x)| p * x).sum();
            }
        }
        out.clear();
        out.resize(m * k, f32::NEG_INFINITY);
        for a in 0..m {
            let ga = &g[a * d..][..d];
            for c in 0..k.min(a) {
                let b = a - 1 - c;
                let s = ga.iter().zip(&pg[b * d..][..d]).map(|(x, y)| x * y).sum::<f32>();
                out[a * k + c] = s + self.distance[bucket((a - b) as u32)];
            }
        }
        Ok(())
    }

    /// Each mention's antecedent from its `scores` [mentions,
    /// antecedents]: the mention with the highest score above 0, the
    /// dummy's, or None. A tie goes to none, then to the nearer mention.
    pub fn decode(&self, scores: &[f32], out: &mut Vec<Option<u32>>) -> Result<()> {
        let k = self.antecedents;
        if !scores.len().is_multiple_of(k) {
            return Err(Error::new(INVALID_SHAPE, format!("{} scores are not rows of {k}", scores.len())));
        }
        if scores.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err(Error::new(INVALID_ARGUMENT, "a pair score is not finite"));
        }
        out.clear();
        for (a, row) in scores.chunks_exact(k).enumerate() {
            let mut best: (f32, Option<u32>) = (0.0, None);
            for (c, &s) in row.iter().enumerate().take(a) {
                if s > best.0 {
                    best = (s, Some((a - 1 - c) as u32));
                }
            }
            out.push(best.1);
        }
        Ok(())
    }
}

/// Mentions in order of first word then last, none twice, each within
/// `n` words and at most `max` words long.
pub fn check_mentions(mentions: &[(u32, u32)], n: usize, max: usize) -> Result<()> {
    for (k, &(i, j)) in mentions.iter().enumerate() {
        if i > j || j as usize >= n || (j - i) as usize >= max || (k > 0 && mentions[k - 1] >= (i, j)) {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("mention {k} ({i}, {j}) of {n} words: out of order, repeated, outside, or past {max} words"),
            ));
        }
    }
    Ok(())
}

/// Each mention's chain from its antecedent links: the index of the
/// chain's first mention. Refused unless each antecedent comes before its
/// mention.
pub fn chains(antecedents: &[Option<u32>]) -> Result<Vec<u32>> {
    let mut out: Vec<u32> = Vec::with_capacity(antecedents.len());
    for (a, &b) in antecedents.iter().enumerate() {
        match b {
            None => out.push(a as u32),
            Some(b) if (b as usize) < a => out.push(out[b as usize]),
            Some(b) => return Err(invalid(format!("mention {a}'s antecedent {b} is not before it"))),
        }
    }
    Ok(out)
}
