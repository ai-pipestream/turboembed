//! A biaffine dependency parser head (`heads[]` of kind
//! HEAD_BIAFFINE_PARSER), after Dozat and Manning (2017) as SuPar
//! (MIT, commit bebdd350) lays out its weights: read on the host from
//! the encoder's final-layer subword states, whichever device ran it.
//!
//! The steps, for one sentence of n words:
//! 1. the word vectors, as the token tagger reads them (words.rs), with
//!    the learned `root` vector before them as node 0;
//! 2. four one-layer MLPs with LeakyReLU(0.1): the dependent and head
//!    views of each node for arcs (`biaffine.arc` wide) and for
//!    relations (`biaffine.rel` wide);
//! 3. arc scores s(d, h) = [dep_d; 1]^T U head_h, U [arc + 1, arc], for
//!    every dependent d in 1..=n and head h in 0..=n;
//! 4. decode to a tree (tree.rs): DECODE_MST or DECODE_EISNER;
//! 5. relation scores r_l(d) = [rdep_d; 1]^T W_l [rhead_h; 1] for each
//!    word's chosen head h, W [labels, rel + 1, rel + 1], and the best
//!    relation of each (a tie to the lower index).

use crate::bundle::Bundle;
use crate::manifest::{Decode, HeadKind, HeadSpec};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::tagger::load_tensors;
use super::words::{self, Take, WordReader};
use super::{Linear, crf, tree};

/// LeakyReLU's slope below zero, as SuPar's MLP has it.
pub const LEAKY_SLOPE: f32 = 0.1;

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let (mut out, w) = words::tensors(spec, hidden);
    let (arc, rel) = spec.biaffine.map_or((0, 0), |b| (b.arc as u64, b.rel as u64));
    let labels = spec.labels.len() as u64;
    out.push(("root".into(), vec![w]));
    for (name, width) in [("arc_dep", arc), ("arc_head", arc), ("rel_dep", rel), ("rel_head", rel)] {
        out.push((format!("{name}.weight"), vec![width, w]));
        out.push((format!("{name}.bias"), vec![width]));
    }
    out.push(("arc.weight".into(), vec![arc + 1, arc]));
    out.push(("rel.weight".into(), vec![labels, rel + 1, rel + 1]));
    out
}

/// A loaded biaffine parser. Read-only after loading, so one parser
/// serves many threads, each with its own [`Work`].
#[derive(Debug)]
pub struct BiaffineParser {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    root: Vec<f32>,
    arc_dep: Linear,
    arc_head: Linear,
    rel_dep: Linear,
    rel_head: Linear,
    /// U [arc + 1, arc] as a layer: a head view to its projections.
    arc: Linear,
    /// W [labels, rel + 1, rel + 1] as a layer of labels x (rel + 1)
    /// outputs: a head's [rhead; 1] to W_l [rhead; 1] for every label.
    rel: Linear,
}

/// What a parser computes in, kept across calls. After [`BiaffineParser::arcs`]
/// it holds the sentence's relation views for [`BiaffineParser::relations`].
#[derive(Default)]
pub struct Work {
    words: words::Work,
    nodes: Vec<f32>,
    dep: Vec<f32>,
    head: Vec<f32>,
    rel_dep: Vec<f32>,
    rel_head: Vec<f32>,
    projected: Vec<f32>,
    n: usize,
}

fn leaky(v: &mut [f32]) {
    v.iter_mut().for_each(|x| {
        if *x < 0.0 {
            *x *= LEAKY_SLOPE;
        }
    });
}

impl BiaffineParser {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<BiaffineParser> {
        let l = load_tensors(bundle, name, HeadKind::BiaffineParser, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A parser for `spec` over an encoder `hidden` wide, from `get`,
    /// which gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::BiaffineParser {
            return Err(invalid(format!("head {:?} is not a biaffine parser", spec.name)));
        }
        spec.validate("head")?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let w = reader.width();
        let b = spec.biaffine.expect("validated");
        let mlp = |name: &str, width: u32| {
            Linear::new(w, width as usize, t.take(&format!("{name}.weight"))?, t.take(&format!("{name}.bias"))?)
        };
        let (root, arc, rel) = (t.take("root")?, t.take("arc.weight")?, t.take("rel.weight")?);
        if !root.iter().chain(&arc).chain(&rel).all(|v| v.is_finite()) {
            return Err(invalid(format!("head {:?}: a root, arc or rel weight is not finite", spec.name)));
        }
        let (a, k, labels) = (b.arc as usize, b.rel as usize + 1, spec.labels.len());
        let parser = BiaffineParser {
            root,
            arc_dep: mlp("arc_dep", b.arc)?,
            arc_head: mlp("arc_head", b.arc)?,
            rel_dep: mlp("rel_dep", b.rel)?,
            rel_head: mlp("rel_head", b.rel)?,
            arc: Linear::new(a, a + 1, arc, vec![0.0; a + 1])?,
            rel: Linear::new(k, labels * k, rel, vec![0.0; labels * k])?,
            spec: spec.clone(),
            hidden,
            reader,
        };
        Ok(parser)
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    /// The relations, by label index.
    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    /// The encoder width the parser reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The arc scores [n, n + 1] of one sentence into `out` (row d - 1:
    /// dependent d's score for each head 0..=n, 0 the root), from its
    /// subword states and word map as [`super::TokenTagger::scores`]
    /// takes them. Returns n. Keeps the relation views in `w`.
    pub fn arcs(&self, states: &[f32], words: &[u32], out: &mut Vec<f32>, w: &mut Work) -> Result<usize> {
        out.clear();
        let n = self.reader.read(states, words, None, &mut w.words)?;
        w.n = n;
        if n == 0 {
            return Ok(0);
        }
        let width = self.reader.width();
        w.nodes.clear();
        w.nodes.extend_from_slice(&self.root);
        w.nodes.extend_from_slice(&w.words.pooled[..n * width]);
        let rows = n + 1;
        for (lin, dst) in [
            (&self.arc_dep, &mut w.dep),
            (&self.arc_head, &mut w.head),
            (&self.rel_dep, &mut w.rel_dep),
            (&self.rel_head, &mut w.rel_head),
        ] {
            dst.clear();
            dst.resize(rows * lin.out_dim, 0.0);
            lin.apply_rows(&w.nodes, rows, dst);
            leaky(dst);
        }
        // U head_h for each head, [rows, arc + 1], in one packed product;
        // then each dependent's [dep_d; 1] against every head's, a second
        // product with the heads' projections as its weights.
        let a = self.arc_dep.out_dim;
        w.projected.clear();
        w.projected.resize(rows * (a + 1), 0.0);
        self.arc.apply_rows(&w.head, rows, &mut w.projected);
        let (weights, bias): (Vec<f32>, Vec<f32>) = (
            w.projected.chunks_exact(a + 1).flat_map(|p| p[..a].iter().copied()).collect(),
            w.projected.chunks_exact(a + 1).map(|p| p[a]).collect(),
        );
        let heads = Linear::new(a, rows, weights, bias)?;
        out.resize(n * rows, 0.0);
        heads.apply_rows(&w.dep[a..rows * a], n, out);
        Ok(n)
    }

    /// Each word's head (0 the root) from its arc scores, by the head's
    /// decode.
    pub fn decode(&self, arcs: &[f32], n: usize, heads: &mut Vec<u32>) -> Result<()> {
        match self.spec.decode {
            Decode::Eisner => tree::eisner(arcs, n, heads),
            _ => tree::mst(arcs, n, heads),
        }
    }

    /// The relation scores [n, labels] of the sentence the last
    /// [`BiaffineParser::arcs`] read, each word under its head in `heads`.
    pub fn relations(&self, heads: &[u32], out: &mut Vec<f32>, w: &Work) -> Result<()> {
        let n = w.n;
        if heads.len() != n {
            return Err(Error::new(INVALID_SHAPE, format!("{} heads for {n} words", heads.len())));
        }
        if heads.iter().enumerate().any(|(i, &h)| h as usize > n || h as usize == i + 1) {
            return Err(Error::new(INVALID_ARGUMENT, "a head is out of range or the word itself"));
        }
        let r = self.rel_dep.out_dim;
        let labels = self.spec.labels.len();
        let k = r + 1;
        out.clear();
        out.resize(n * labels, 0.0);
        // Every word's head [rhead; 1] through all the labels' W_l in one
        // packed product, then each label's score against [rdep; 1].
        let mut ones = Vec::with_capacity(n * k);
        for &h in heads {
            ones.extend_from_slice(&w.rel_head[h as usize * r..][..r]);
            ones.push(1.0);
        }
        let mut projected = vec![0.0; n * labels * k];
        self.rel.apply_rows(&ones, n, &mut projected);
        for d in 1..=n {
            let x = &w.rel_dep[d * r..][..r];
            let z = &projected[(d - 1) * labels * k..][..labels * k];
            for (l, zl) in z.chunks_exact(k).enumerate() {
                let s: f64 = x.iter().zip(zl).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() + zl[r] as f64;
                out[(d - 1) * labels + l] = s as f32;
            }
        }
        Ok(())
    }

    /// The best relation of each word from its relation scores.
    pub fn relation_labels(&self, scores: &[f32], n: usize, out: &mut Vec<u32>) -> Result<()> {
        crf::argmax(scores, n, self.spec.labels.len(), out)
    }
}
