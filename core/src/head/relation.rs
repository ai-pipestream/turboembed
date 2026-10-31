//! A relation classifier head (`heads[]` of kind
//! HEAD_RELATION_CLASSIFIER): a relation or `none` for every ordered pair
//! of the mentions given with a sequence, found by an entity head (a
//! token tagger or a span classifier) on the same encoder pass.
//!
//! The steps, for one sequence of n words and its m mentions:
//! 1. the word vectors, as the token tagger reads them (words.rs);
//! 2. each mention as its first and last word vectors and, with
//!    `relation.entity_labels`, its entity type's learned embedding
//!    (`relation.type` [types, type_dim]), joined;
//! 3. two one-layer MLPs with LeakyReLU(0.1), as the parser's: the
//!    mention's view as a relation's head (`head.*`) and as its tail
//!    (`tail.*`), each `relation.mlp` wide;
//! 4. scores s_l(a, b) = [head_a; 1]^T W_l [tail_b; 1], W [labels,
//!    mlp + 1, mlp + 1], for every pair of mentions a != b, in order of
//!    a, then b.
//!
//! A pair whose best label (a tie to the lower index) is `none` is no
//! relation; DECODE_RELATIONS keeps every other pair.

use crate::bundle::Bundle;
use crate::manifest::{HeadKind, HeadSpec, NO_RELATION};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::Linear;
use super::parser::LEAKY_SLOPE;
use super::tagger::load_tensors;
use super::words::{self, Take, WordReader};

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let (mut out, width) = words::tensors(spec, hidden);
    let labels = spec.labels.len() as u64;
    let (mlp, types, dim) =
        spec.relation.as_ref().map_or((0, 0, 0), |r| (r.mlp as u64, r.entity_labels.len() as u64, r.type_dim as u64));
    if types > 0 {
        out.push(("relation.type".into(), vec![types, dim]));
    }
    for name in ["head", "tail"] {
        out.push((format!("{name}.weight"), vec![mlp, 2 * width + dim]));
        out.push((format!("{name}.bias"), vec![mlp]));
    }
    out.push(("rel.weight".into(), vec![labels, mlp + 1, mlp + 1]));
    out
}

/// A mention of a sequence: words `first..=last` and its entity type, an
/// index into `relation.entity_labels` (0 when the head reads no type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mention {
    pub first: u32,
    pub last: u32,
    pub entity: u32,
}

/// One kept relation: mention `head` to mention `tail` (indices into the
/// mentions given), its label, and its label's score less `none`'s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relation {
    pub head: u32,
    pub tail: u32,
    pub label: u32,
    pub margin: f32,
}

/// A loaded relation classifier. Read-only after loading, so one head
/// serves many threads, each with its own [`RelationWork`].
#[derive(Debug)]
pub struct RelationClassifier {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    types: Vec<f32>,
    type_dim: usize,
    head: Linear,
    tail: Linear,
    /// W [labels, mlp + 1, mlp + 1] as a layer of labels x (mlp + 1)
    /// outputs: a tail's [tail; 1] to W_l [tail; 1] for every label.
    rel: Linear,
    none: usize,
}

/// What a relation classifier computes in, kept across calls.
#[derive(Default)]
pub struct RelationWork {
    words: words::Work,
    mentions: Vec<f32>,
    heads: Vec<f32>,
    tails: Vec<f32>,
    projected: Vec<f32>,
}

fn leaky(v: &mut [f32]) {
    v.iter_mut().filter(|x| **x < 0.0).for_each(|x| *x *= LEAKY_SLOPE);
}

impl RelationClassifier {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<RelationClassifier> {
        let l = load_tensors(bundle, name, HeadKind::RelationClassifier, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::RelationClassifier {
            return Err(invalid(format!("head {:?} is not a relation classifier", spec.name)));
        }
        spec.validate("head")?;
        let shape = spec.relation.as_ref().ok_or_else(|| invalid(format!("head {:?}: no relation", spec.name)))?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let (mlp, type_dim) = (shape.mlp as usize, shape.type_dim as usize);
        let cols = 2 * reader.width() + type_dim;
        let types = if shape.entity_labels.is_empty() { Vec::new() } else { t.take("relation.type")? };
        let rel = t.take("rel.weight")?;
        if !types.iter().chain(&rel).all(|v| v.is_finite()) {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("head {:?}: a type or rel weight is not finite", spec.name),
            ));
        }
        let mlp_of =
            |name: &str| Linear::new(cols, mlp, t.take(&format!("{name}.weight"))?, t.take(&format!("{name}.bias"))?);
        let (head, tail) = (mlp_of("head")?, mlp_of("tail")?);
        head.validate("head")?;
        tail.validate("tail")?;
        let (k, labels) = (mlp + 1, spec.labels.len());
        let rel = Linear::new(k, labels * k, rel, vec![0.0; labels * k])?;
        let none = spec.labels.iter().position(|l| l == NO_RELATION).ok_or_else(|| invalid("no label \"none\""))?;
        Ok(RelationClassifier { spec: spec.clone(), hidden, reader, types, type_dim, head, tail, rel, none })
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    /// The relations, by label index.
    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    /// The entity types a mention may carry, by index.
    pub fn entity_labels(&self) -> &[String] {
        self.spec.relation.as_ref().map_or(&[], |r| &r.entity_labels)
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The number of ordered pairs `m` mentions make, None past usize.
    pub fn pair_count(m: usize) -> Option<usize> {
        m.checked_mul(m.saturating_sub(1))
    }

    /// The label scores [pairs, labels] of one sequence's `mentions` into
    /// `out`, pairs in order of head mention then tail, from its subword
    /// states and word map as a token tagger reads them. Returns the
    /// number of words.
    ///
    /// Refused: a mention past the words, backwards, or of an entity type
    /// the head lacks (with no types, any but 0), or more mentions than
    /// the scores could hold (INVALID_ARGUMENT); what the word reader
    /// refuses; a score not finite even when summed in f64.
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        mentions: &[Mention],
        out: &mut Vec<f32>,
        w: &mut RelationWork,
    ) -> Result<usize> {
        out.clear();
        let labels = self.spec.labels.len();
        let types = self.entity_labels().len();
        let pairs = Self::pair_count(mentions.len()).and_then(|p| p.checked_mul(labels));
        let pairs = pairs.ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("{} mentions", mentions.len())))?;
        let n = self.reader.read(states, words, None, &mut w.words)?;
        for (i, m) in mentions.iter().enumerate() {
            let typed = if types == 0 { m.entity == 0 } else { (m.entity as usize) < types };
            if m.first > m.last || m.last as usize >= n || !typed {
                return Err(Error::new(INVALID_ARGUMENT, format!("mention {i} {m:?} of a sequence of {n} words")));
            }
        }
        let m = mentions.len();
        if m < 2 {
            return Ok(n);
        }
        let (width, dim) = (self.reader.width(), self.type_dim);
        let cols = 2 * width + dim;
        w.mentions.clear();
        for x in mentions {
            w.mentions.extend_from_slice(&w.words.pooled[x.first as usize * width..][..width]);
            w.mentions.extend_from_slice(&w.words.pooled[x.last as usize * width..][..width]);
            if dim > 0 {
                w.mentions.extend_from_slice(&self.types[x.entity as usize * dim..][..dim]);
            }
        }
        debug_assert_eq!(w.mentions.len(), m * cols);
        let mlp = self.head.out_dim;
        for (lin, dst) in [(&self.head, &mut w.heads), (&self.tail, &mut w.tails)] {
            dst.clear();
            dst.resize(m * mlp, 0.0);
            lin.apply_rows(&w.mentions, m, dst);
            leaky(dst);
        }
        // Every tail's [tail; 1] through all the labels' W_l in one packed
        // product, then each head's [head; 1] against it, summed in f64.
        let k = mlp + 1;
        let mut ones = Vec::with_capacity(m * k);
        for t in w.tails.chunks_exact(mlp) {
            ones.extend_from_slice(t);
            ones.push(1.0);
        }
        w.projected.clear();
        w.projected.resize(m * labels * k, 0.0);
        self.rel.apply_rows(&ones, m, &mut w.projected);
        out.reserve(pairs);
        for a in 0..m {
            let x = &w.heads[a * mlp..][..mlp];
            for b in (0..m).filter(|&b| b != a) {
                let z = &w.projected[b * labels * k..][..labels * k];
                out.extend(z.chunks_exact(k).map(|zl| {
                    (x.iter().zip(zl).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>() + zl[mlp] as f64) as f32
                }));
            }
        }
        // A projection past f32's range is summed again in f64 throughout,
        // and a score not finite even so is refused.
        if !w.projected.iter().chain(out.iter()).all(|v| v.is_finite()) {
            self.scores_f64(m, out, w);
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("head {:?}: a relation score is not finite", self.spec.name),
            ));
        }
        Ok(n)
    }

    /// The scores of [`RelationClassifier::scores`] summed in f64 alone.
    fn scores_f64(&self, m: usize, out: &mut [f32], w: &RelationWork) {
        let (mlp, labels, wr) = (self.head.out_dim, self.spec.labels.len(), self.rel.weight());
        let k = mlp + 1;
        let one = |v: &[f32], i: usize| if i < mlp { v[i] as f64 } else { 1.0 };
        let mut row = 0;
        for a in 0..m {
            let x = &w.heads[a * mlp..][..mlp];
            for b in (0..m).filter(|&b| b != a) {
                let y = &w.tails[b * mlp..][..mlp];
                for l in 0..labels {
                    let s: f64 = (0..k)
                        .map(|i| one(x, i) * (0..k).map(|j| wr[(l * k + i) * k + j] as f64 * one(y, j)).sum::<f64>())
                        .sum();
                    out[row * labels + l] = s as f32;
                }
                row += 1;
            }
        }
    }

    /// The relations of `m` mentions kept from their `scores` [pairs,
    /// labels]: every pair whose best label is not `none`, in order of
    /// head mention then tail.
    pub fn decode(&self, scores: &[f32], m: usize, out: &mut Vec<Relation>) -> Result<()> {
        let labels = self.spec.labels.len();
        if m > u32::MAX as usize || Some(scores.len()) != Self::pair_count(m).and_then(|p| p.checked_mul(labels)) {
            return Err(Error::new(INVALID_SHAPE, format!("{} relation scores for {m} mentions", scores.len())));
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a relation score is not finite"));
        }
        out.clear();
        let mut rows = scores.chunks_exact(labels);
        for a in 0..m {
            for b in (0..m).filter(|&b| b != a) {
                let Some(row) = rows.next() else { break };
                let best = (0..labels).fold(0, |c, l| if row[l] > row[c] { l } else { c });
                if best != self.none {
                    let margin = row[best] - row[self.none];
                    if !margin.is_finite() {
                        return Err(Error::new(INVALID_ARGUMENT, "a relation margin is past f32"));
                    }
                    out.push(Relation { head: a as u32, tail: b as u32, label: best as u32, margin });
                }
            }
        }
        Ok(())
    }
}
