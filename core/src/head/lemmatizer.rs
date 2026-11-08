//! An edit-tree lemmatizer head (`heads[]` of kind
//! HEAD_EDIT_TREE_LEMMATIZER): each word classified into one of the
//! head's edit trees (edit_tree.rs), its lemma the tree applied to it.
//!
//! The steps, for one sequence of words:
//! 1. the word vectors, as every word head reads them (words.rs);
//! 2. with a `pos_input`, each word's tag (decoded by the named tagger
//!    head) looked up in the head's tag embedding and joined after its
//!    vector;
//! 3. one linear layer scores every tree;
//! 4. decode: a word in `seen_lookup` (unambiguous in training at least
//!    `min_freq` times, or unambiguous for its tag when `pos_input` is
//!    present) takes its recorded lemma; otherwise the best-scoring tree
//!    that fits the word (a tie to the lower index), or the word itself
//!    when none fits.
//!
//! The labels are the trees, so the tree table ships in the manifest
//! with the head and is checked when the bundle is opened.

use std::collections::HashMap;

use crate::bundle::Bundle;
use crate::manifest::{HeadKind, HeadSpec};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::Linear;
use super::edit_tree::EditTree;
use super::tagger::load_tensors;
pub use super::words::Work;
use super::words::{self, Take, WordReader};

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let labels = spec.labels.len() as u64;
    let (mut out, mut width) = words::tensors(spec, hidden);
    if let Some(p) = &spec.pos_input {
        out.push(("pos.embedding".into(), vec![p.labels as u64, p.dim as u64]));
        width += p.dim as u64;
    }
    out.push(("out.weight".into(), vec![labels, width]));
    out.push(("out.bias".into(), vec![labels]));
    words::push_mix(spec, &mut out);
    out
}

/// A loaded lemmatizer. Read-only after loading, so one serves many
/// threads, each with its own [`Work`].
#[derive(Debug)]
pub struct EditTreeLemmatizer {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    /// [tags, dim] and dim, with a `pos_input`.
    pos: Option<(Vec<f32>, usize)>,
    out: Linear,
    trees: Vec<EditTree>,
    seen_forms: HashMap<String, String>,
    seen_by_tag: Vec<HashMap<String, String>>,
}

impl EditTreeLemmatizer {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<EditTreeLemmatizer> {
        let l = load_tensors(bundle, name, HeadKind::EditTreeLemmatizer, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A lemmatizer for `spec` over an encoder `hidden` wide, from `get`,
    /// which gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::EditTreeLemmatizer {
            return Err(invalid(format!("head {:?} is not an edit-tree lemmatizer", spec.name)));
        }
        spec.validate("head")?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let pos = match &spec.pos_input {
            Some(p) => {
                let table = t.take("pos.embedding")?;
                if table.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(format!(
                        "head {:?}: pos.embedding holds a value that is not finite",
                        spec.name
                    )));
                }
                Some((table, p.dim as usize))
            }
            None => None,
        };
        let width = reader.width() + pos.as_ref().map_or(0, |(_, d)| *d);
        let out = Linear::new(width, spec.labels.len(), t.take("out.weight")?, t.take("out.bias")?)?;
        let trees = spec.labels.iter().map(|l| EditTree::parse(l)).collect::<Result<_>>()?;
        let (mut seen_forms, mut seen_by_tag) = (HashMap::new(), Vec::new());
        if let Some(s) = &spec.seen_lookup {
            for e in &s.entries {
                match e.tag {
                    None => {
                        seen_forms.insert(e.form.clone(), e.lemma.clone());
                    }
                    Some(t) => {
                        let idx = t as usize;
                        if seen_by_tag.len() <= idx {
                            seen_by_tag.resize_with(idx + 1, HashMap::new);
                        }
                        seen_by_tag[idx].insert(e.form.clone(), e.lemma.clone());
                    }
                }
            }
        }
        Ok(EditTreeLemmatizer { spec: spec.clone(), hidden, reader, pos, out, trees, seen_forms, seen_by_tag })
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    /// The trees, in label order.
    pub fn trees(&self) -> &[EditTree] {
        &self.trees
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The number of encoder layers each word vector mixes.
    pub fn taps(&self) -> usize {
        self.reader.taps()
    }

    /// The tree scores [words, trees] of one sequence into `out`, from its
    /// subword states and word map as `TokenTagger::scores` takes them
    /// and, with a `pos_input`, each word's tag index (`tags`, one a
    /// word; without one, None). Returns the number of words.
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        tags: Option<&[u32]>,
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<usize> {
        self.scores_language(states, words, tags, None, out, w)
    }

    /// As [`scores`](Self::scores), selecting a language embedding by name.
    pub fn scores_language(
        &self,
        states: &[f32],
        words: &[u32],
        tags: Option<&[u32]>,
        language: Option<&str>,
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<usize> {
        out.clear();
        let n = self.reader.read_language(states, words, None, language, w)?;
        let width = self.reader.width();
        let joined: std::borrow::Cow<'_, [f32]> = match (&self.pos, tags) {
            (None, None) => (&w.pooled[..n * width]).into(),
            (Some((table, dim)), Some(tags)) => {
                if tags.len() != n {
                    return Err(Error::new(INVALID_SHAPE, format!("{} tags for {n} words", tags.len())));
                }
                let rows = table.len() / dim;
                let mut x = Vec::with_capacity(n * (width + dim));
                for (i, &t) in tags.iter().enumerate() {
                    if t as usize >= rows {
                        return Err(Error::new(INVALID_ARGUMENT, format!("tag {t} past the {rows} tags")));
                    }
                    x.extend_from_slice(&w.pooled[i * width..][..width]);
                    x.extend_from_slice(&table[t as usize * dim..][..*dim]);
                }
                x.into()
            }
            _ => {
                return Err(invalid(format!(
                    "head {:?}: tags are given exactly when the head has a pos_input",
                    self.spec.name
                )));
            }
        };
        out.resize(n * self.out.out_dim, 0.0);
        if n > 0 {
            self.out.apply_rows(&joined, n, out);
        }
        Ok(n)
    }

    /// Each word's tree: the best-scoring one that fits `forms[i]` (a tie
    /// to the lower index), or None when none fits.
    pub fn decode(&self, scores: &[f32], forms: &[&str], out: &mut Vec<Option<u32>>) -> Result<()> {
        let k = self.trees.len();
        if Some(scores.len()) != forms.len().checked_mul(k) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("{} scores for {} words of {k} trees", scores.len(), forms.len()),
            ));
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a tree score is not finite"));
        }
        out.clear();
        let mut order: Vec<u32> = Vec::with_capacity(k);
        for (row, form) in scores.chunks_exact(k).zip(forms) {
            // Most words take their best tree: try it before sorting.
            let best = (0..k).fold(0, |b, j| if row[j] > row[b] { j } else { b });
            if self.trees[best].apply(form).is_some() {
                out.push(Some(best as u32));
                continue;
            }
            order.clear();
            order.extend(0..k as u32);
            // Stable, so equal scores keep the lower index first; equal is
            // numeric (the scores are finite), so -0 and 0 tie.
            order.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap_or(std::cmp::Ordering::Equal));
            out.push(order.iter().copied().find(|&t| self.trees[t as usize].apply(form).is_some()));
        }
        Ok(())
    }

    /// Each word's lemma from its decoded tree, the word itself for None,
    /// after `spec.seen_lookup` (unambiguous forms seen in training).
    pub fn lemmas(&self, trees: &[Option<u32>], forms: &[&str]) -> Result<Vec<String>> {
        self.lemmas_tagged(trees, forms, None)
    }

    /// [`lemmas`](Self::lemmas), reading each word's tag index `tags`
    /// when the head has a `pos_input` so tag-specific `seen_lookup`
    /// entries apply.
    pub fn lemmas_tagged(&self, trees: &[Option<u32>], forms: &[&str], tags: Option<&[u32]>) -> Result<Vec<String>> {
        if trees.len() != forms.len() {
            return Err(Error::new(INVALID_SHAPE, format!("{} trees for {} words", trees.len(), forms.len())));
        }
        if let Some(tags) = tags {
            let Some((table, dim)) = &self.pos else {
                return Err(invalid(format!(
                    "head {:?}: tags are given exactly when the head has a pos_input",
                    self.spec.name
                )));
            };
            if tags.len() != forms.len() {
                return Err(Error::new(INVALID_SHAPE, format!("{} tags for {} words", tags.len(), forms.len())));
            }
            let rows = table.len() / dim;
            if let Some(&t) = tags.iter().find(|&&t| t as usize >= rows) {
                return Err(Error::new(INVALID_ARGUMENT, format!("tag {t} past the {rows} tags")));
            }
        }
        trees
            .iter()
            .zip(forms)
            .enumerate()
            .map(|(i, (t, f))| {
                let by_tree = match t {
                    None => (*f).to_owned(),
                    Some(t) => self
                        .trees
                        .get(*t as usize)
                        .and_then(|tree| tree.apply(f))
                        .ok_or_else(|| Error::new(INVALID_ARGUMENT, format!("tree {t} does not fit {f:?}")))?,
                };
                if let Some(tags) = tags
                    && let Some(lemma) = self.seen_by_tag.get(tags[i] as usize).and_then(|m| m.get(*f))
                {
                    return Ok(lemma.clone());
                }
                if let Some(lemma) = self.seen_forms.get(*f) {
                    return Ok(lemma.clone());
                }
                Ok(by_tree)
            })
            .collect()
    }
}
