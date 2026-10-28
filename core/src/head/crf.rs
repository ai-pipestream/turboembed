//! Label decoding for the token tagger (tagger.rs): the transitions a
//! scheme allows, first-order CRF Viterbi under them, argmax, and the
//! word spans a BIO or BILOU label sequence marks.
//!
//! Every decode is deterministic: sums are taken in f64 in a fixed order,
//! and a tie goes to the lowest label index.

use crate::manifest::{Part, Scheme};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

/// Which label sequences a scheme allows, as three masks over label
/// indices: may start a sequence, may follow another, may end one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allowed {
    labels: usize,
    start: Vec<bool>,
    /// [from * labels + to].
    next: Vec<bool>,
    end: Vec<bool>,
}

/// Where a label stands in a span: its part and the index of its type in
/// `Allowed::types`, so comparisons need no strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Outside,
    Begin(u32),
    Inside(u32),
    Last(u32),
    Unit(u32),
}

/// The span types of `labels` under `scheme`, each once, in the order the
/// labels first name them, and each label's role. A label the scheme has
/// no reading for is refused.
fn roles(scheme: Scheme, labels: &[String]) -> Result<(Vec<String>, Vec<Role>)> {
    let mut types: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(labels.len());
    for l in labels {
        let part = scheme.part(l).ok_or_else(|| invalid(format!("label {l:?} is not a label of {scheme:?}")))?;
        let mut kind = |t: &str| -> u32 {
            match types.iter().position(|k| k == t) {
                Some(i) => i as u32,
                None => {
                    types.push(t.to_owned());
                    (types.len() - 1) as u32
                }
            }
        };
        out.push(match part {
            Part::Outside => Role::Outside,
            Part::Begin(t) => Role::Begin(kind(t)),
            Part::Inside(t) => Role::Inside(kind(t)),
            Part::Last(t) => Role::Last(kind(t)),
            Part::Unit(t) => Role::Unit(kind(t)),
        });
    }
    Ok((types, out))
}

/// The span types and allowed transitions of `labels` under `scheme`.
pub fn allowed(scheme: Scheme, labels: &[String]) -> Result<(Vec<String>, Allowed)> {
    let (types, roles) = roles(scheme, labels)?;
    let n = labels.len();
    let open = |r: Role| match r {
        Role::Begin(t) | Role::Inside(t) => Some(t),
        _ => None,
    };
    let mut a = Allowed { labels: n, start: vec![true; n], next: vec![true; n * n], end: vec![true; n] };
    for (to, &r) in roles.iter().enumerate() {
        // What may come after a label: under BIO an Inside only continues
        // its own type; under BILOU an open span must go on (Inside) or
        // close (Last) in its own type, and nothing else may.
        let continues = |t: u32| matches!(r, Role::Inside(k) | Role::Last(k) if k == t);
        a.start[to] = match scheme {
            Scheme::Tags => true,
            Scheme::Bio => !matches!(r, Role::Inside(_)),
            Scheme::Bilou => !matches!(r, Role::Inside(_) | Role::Last(_)),
        };
        a.end[to] = scheme != Scheme::Bilou || open(r).is_none();
        for (from, &p) in roles.iter().enumerate() {
            a.next[from * n + to] = match (scheme, open(p)) {
                (Scheme::Tags, _) => true,
                (Scheme::Bio, Some(t)) => !matches!(r, Role::Inside(k) if k != t),
                (Scheme::Bio, None) => !matches!(r, Role::Inside(_)),
                (Scheme::Bilou, Some(t)) => continues(t),
                (Scheme::Bilou, None) => !matches!(r, Role::Inside(_) | Role::Last(_)),
            };
        }
    }
    Ok((types, a))
}

impl Allowed {
    pub fn may_start(&self, label: usize) -> bool {
        self.start[label]
    }

    pub fn may_follow(&self, from: usize, to: usize) -> bool {
        self.next[from * self.labels + to]
    }

    pub fn may_end(&self, label: usize) -> bool {
        self.end[label]
    }
}

/// A first-order CRF's scores: `transitions` [from, to], `start` and
/// `end` [labels], added to the emission scores of each word.
#[derive(Debug, Clone, PartialEq)]
pub struct Crf {
    pub labels: usize,
    pub transitions: Vec<f32>,
    pub start: Vec<f32>,
    pub end: Vec<f32>,
}

impl Crf {
    pub fn new(labels: usize, transitions: Vec<f32>, start: Vec<f32>, end: Vec<f32>) -> Result<Crf> {
        if labels == 0 || transitions.len() != labels * labels || start.len() != labels || end.len() != labels {
            return Err(Error::new(INVALID_SHAPE, format!("crf: shapes do not match {labels} labels")));
        }
        if !transitions.iter().chain(&start).chain(&end).all(|v| v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "crf: a weight is not finite"));
        }
        Ok(Crf { labels, transitions, start, end })
    }
}

fn check(scores: &[f32], words: usize, labels: usize) -> Result<()> {
    if Some(scores.len()) != words.checked_mul(labels) {
        return Err(Error::new(INVALID_SHAPE, format!("{} scores for {words} words of {labels} labels", scores.len())));
    }
    if !scores.iter().all(|v| v.is_finite()) {
        return Err(Error::new(INVALID_ARGUMENT, "a score is not finite"));
    }
    Ok(())
}

/// The best label of each word on its own; a tie goes to the lower index.
pub fn argmax(scores: &[f32], words: usize, labels: usize, out: &mut Vec<u32>) -> Result<()> {
    check(scores, words, labels)?;
    out.clear();
    for row in scores.chunks_exact(labels) {
        let mut best = 0;
        for (j, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = j;
            }
        }
        out.push(best as u32);
    }
    Ok(())
}

/// The label sequence with the highest CRF score among those `allowed`
/// admits: start[y0] + sum emit[i][yi] + sum trans[y(i-1)][yi] + end[y(n-1)].
/// A tie goes to the lower label index, at each step and at the end.
/// INVALID_ARGUMENT if the scheme admits no sequence of this length.
pub fn viterbi(crf: &Crf, allowed: &Allowed, scores: &[f32], words: usize, out: &mut Vec<u32>) -> Result<()> {
    let n = crf.labels;
    check(scores, words, n)?;
    out.clear();
    if words == 0 {
        return Ok(());
    }
    let mut score: Vec<f64> = (0..n)
        .map(|j| if allowed.may_start(j) { crf.start[j] as f64 + scores[j] as f64 } else { f64::NEG_INFINITY })
        .collect();
    let mut next = vec![f64::NEG_INFINITY; n];
    let mut back = vec![0u32; words * n];
    for i in 1..words {
        let emit = &scores[i * n..][..n];
        for to in 0..n {
            let mut best = f64::NEG_INFINITY;
            let mut from_best = 0u32;
            for (from, &prev) in score.iter().enumerate() {
                if prev == f64::NEG_INFINITY || !allowed.may_follow(from, to) {
                    continue;
                }
                let s = prev + crf.transitions[from * n + to] as f64;
                if s > best {
                    best = s;
                    from_best = from as u32;
                }
            }
            next[to] = if best == f64::NEG_INFINITY { best } else { best + emit[to] as f64 };
            back[i * n + to] = from_best;
        }
        std::mem::swap(&mut score, &mut next);
    }
    let mut last = None;
    let mut best = f64::NEG_INFINITY;
    for (j, &total) in score.iter().enumerate() {
        if total == f64::NEG_INFINITY || !allowed.may_end(j) {
            continue;
        }
        let s = total + crf.end[j] as f64;
        if s > best {
            best = s;
            last = Some(j);
        }
    }
    let Some(mut j) = last else {
        return Err(Error::new(INVALID_ARGUMENT, format!("the scheme admits no label sequence of {words} words")));
    };
    out.resize(words, 0);
    for i in (0..words).rev() {
        out[i] = j as u32;
        j = back[i * n + j] as usize;
    }
    Ok(())
}

/// A run of words a BIO or BILOU sequence marks: words `start..end`, of
/// type `kind` (an index into the tagger's types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordSpan {
    pub start: u32,
    pub end: u32,
    pub kind: u32,
}

/// The spans `tags` (label indices) mark under `scheme`. An ill-formed
/// sequence, which argmax can produce, is read by one rule: an Inside or
/// Last that does not continue an open span of its type starts a new one;
/// Begin and Unit close what is open; Outside closes; a span still open at
/// the end closes there. Under SCHEME_TAGS there are no spans.
pub fn spans(scheme: Scheme, labels: &[String], tags: &[u32]) -> Result<Vec<WordSpan>> {
    let (_, roles) = roles(scheme, labels)?;
    let mut out = Vec::new();
    if scheme == Scheme::Tags {
        return Ok(out);
    }
    let mut open: Option<(u32, u32)> = None;
    for (i, &t) in tags.iter().enumerate() {
        let i = i as u32;
        let r =
            *roles.get(t as usize).ok_or_else(|| invalid(format!("tag {t} is past the {} labels", labels.len())))?;
        let mut close = |open: &mut Option<(u32, u32)>, end: u32| {
            if let Some((start, kind)) = open.take() {
                out.push(WordSpan { start, end, kind });
            }
        };
        match r {
            Role::Outside => close(&mut open, i),
            Role::Begin(k) => {
                close(&mut open, i);
                open = Some((i, k));
            }
            Role::Inside(k) => {
                if open.is_none_or(|(_, o)| o != k) {
                    close(&mut open, i);
                    open = Some((i, k));
                }
            }
            Role::Last(k) => {
                if open.is_none_or(|(_, o)| o != k) {
                    close(&mut open, i);
                    open = Some((i, k));
                }
                close(&mut open, i + 1);
            }
            Role::Unit(k) => {
                close(&mut open, i);
                open = Some((i, k));
                close(&mut open, i + 1);
            }
        }
    }
    if let Some((start, kind)) = open {
        out.push(WordSpan { start, end: tags.len() as u32, kind });
    }
    Ok(out)
}
