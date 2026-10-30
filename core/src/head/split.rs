//! A token split head (`heads[]` of kind HEAD_TOKEN_SPLIT): how each
//! token splits into the syntactic words a UD treebank gives a multi-word
//! token (Spanish `del` as `de` and `el`, Italian `della` as `di` and
//! `la`), so heads trained on UD words read UD words.
//!
//! Each label is `_`, which keeps the token one word, or a split: two or
//! more edit trees (edit_tree.rs) separated by single spaces, the k-th
//! tree rewriting the whole token into its k-th word. `del` splits by
//! `(m 0 1 () (s 'l' ''))` (keep `de`) and `(m 1 0 (s 'd' '') ())` (keep
//! `el`). A tree keeps no stem, so one label serves every token whose
//! words rewrite from it the same way, as a lemmatizer's tree serves
//! every word inflected the same way. The tensors are a token tagger's
//! (tagger.rs); the decode takes the best label that fits.

use crate::bundle::Bundle;
use crate::manifest::{HeadKind, HeadSpec, KEEP_TOKEN};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::edit_tree::EditTree;
use super::tagger::{self, TokenTagger, load_tensors};
use super::words::Work;

/// The most words one split gives.
pub const MAX_SPLIT_WORDS: usize = 8;

/// The trees of a split label, refused unless it is two to
/// [`MAX_SPLIT_WORDS`] canonical tree labels separated by single spaces.
pub fn parse(label: &str) -> Result<Vec<EditTree>> {
    let bad = |what: &str| invalid(format!("split {label:?}: {what}"));
    let bytes = label.as_bytes();
    let mut trees = Vec::new();
    let (mut start, mut depth, mut quoted, mut escaped) = (0, 0usize, false, false);
    for (i, &b) in bytes.iter().enumerate() {
        if quoted {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'\'') => quoted = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'\'' => quoted = true,
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1).ok_or_else(|| bad("a ')' closes nothing"))?;
                if depth == 0 {
                    trees.push(EditTree::parse(&label[start..=i])?);
                    match bytes.get(i + 1) {
                        None => start = i + 1,
                        Some(b' ') if bytes.get(i + 2) == Some(&b'(') => start = i + 2,
                        Some(_) => return Err(bad("trees are separated by one space")),
                    }
                }
            }
            // The separator after a tree, already checked.
            b' ' if depth == 0 && start == i + 1 => {}
            _ if depth == 0 => return Err(bad("text outside a tree")),
            _ => {}
        }
    }
    if depth != 0 || quoted || start != bytes.len() {
        return Err(bad("an unclosed tree"));
    }
    if trees.len() < 2 || trees.len() > MAX_SPLIT_WORDS {
        return Err(bad(&format!("{} trees, 2 to {MAX_SPLIT_WORDS}", trees.len())));
    }
    Ok(trees)
}

/// The words a split gives `form`: each tree applied to it, or None when
/// a tree does not fit or gives an empty word.
fn apply(trees: &[EditTree], form: &str) -> Option<Vec<String>> {
    trees.iter().map(|t| t.apply(form).filter(|w| !w.is_empty())).collect()
}

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape:
/// a token tagger's.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    tagger::tensors(spec, hidden)
}

/// A loaded token split head. Read-only after loading, so one head serves
/// many threads, each with its own [`Work`].
#[derive(Debug)]
pub struct TokenSplitter {
    scorer: TokenTagger,
    /// Each label's trees; None for `_`.
    splits: Vec<Option<Vec<EditTree>>>,
}

impl TokenSplitter {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<TokenSplitter> {
        let l = load_tensors(bundle, name, HeadKind::TokenSplit, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::TokenSplit {
            return Err(invalid(format!("head {:?} is not a token split head", spec.name)));
        }
        let scorer = TokenTagger::scoring_words(spec, hidden, get)?;
        let splits = spec
            .labels
            .iter()
            .map(|l| if l == KEEP_TOKEN { Ok(None) } else { parse(l).map(Some) })
            .collect::<Result<_>>()?;
        Ok(TokenSplitter { scorer, splits })
    }

    pub fn spec(&self) -> &HeadSpec {
        self.scorer.spec()
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.scorer.hidden()
    }

    /// Whether the head reads each word's text (a `hash_embed`).
    pub fn reads_forms(&self) -> bool {
        self.scorer.reads_forms()
    }

    /// The label scores [tokens, labels] of one sequence, as
    /// [`TokenTagger::scores_forms`] gives a tagger's. Returns the number
    /// of tokens.
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        out: &mut Vec<f32>,
        w: &mut Work,
    ) -> Result<usize> {
        self.scorer.scores_forms(states, words, forms, out, w)
    }

    /// Each token's label: the best-scoring one that fits `forms[i]` (a
    /// tie to the lower index). `_` always fits.
    pub fn decode(&self, scores: &[f32], forms: &[&str], out: &mut Vec<u32>) -> Result<()> {
        let k = self.splits.len();
        if Some(scores.len()) != forms.len().checked_mul(k) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("{} scores for {} tokens of {k} labels", scores.len(), forms.len()),
            ));
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a split score is not finite"));
        }
        out.clear();
        let mut order: Vec<usize> = Vec::with_capacity(k);
        for (row, form) in scores.chunks_exact(k).zip(forms) {
            order.clear();
            order.extend(0..k);
            // Stable, so equal scores keep the lower index first.
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            let fits = |&j: &usize| self.splits[j].as_ref().is_none_or(|t| apply(t, form).is_some());
            let best = order.iter().copied().find(fits).ok_or_else(|| invalid("a split head with no `_` label"))?;
            out.push(best as u32);
        }
        Ok(())
    }

    /// Each token's words from its decoded label: the token itself for
    /// `_`, else its split's words.
    pub fn words(&self, labels: &[u32], forms: &[&str]) -> Result<Vec<Vec<String>>> {
        if labels.len() != forms.len() {
            return Err(Error::new(INVALID_SHAPE, format!("{} labels for {} tokens", labels.len(), forms.len())));
        }
        labels
            .iter()
            .zip(forms)
            .map(|(&l, &f)| match self.splits.get(l as usize) {
                Some(None) => Ok(vec![f.to_owned()]),
                Some(Some(t)) => apply(t, f).ok_or_else(|| {
                    Error::new(
                        INVALID_ARGUMENT,
                        format!("split {:?} does not fit {f:?}", self.spec().labels[l as usize]),
                    )
                }),
                None => Err(Error::new(INVALID_ARGUMENT, format!("label {l} of {}", self.splits.len()))),
            })
            .collect()
    }
}
