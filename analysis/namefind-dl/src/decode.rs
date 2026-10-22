//! `NameFinderDL`'s decoding, on the model's per-piece label scores:
//!
//! - the text is cut into chunks of whitespace-separated words, at most
//!   `split` words each, consecutive chunks sharing `overlap` words;
//! - each piece's label is the first highest score that is not NaN, and its
//!   probability that label's softmax probability (with OpenNLP's handling
//!   of infinities);
//! - a `B-<type>` piece starts an entity that runs over the `I-<type>`
//!   pieces after it, with the lowest probability among them;
//! - candidates from every chunk are resolved longest first, then most
//!   probable, and a candidate overlapping one already kept is dropped.
//!
//! Where `NameFinderDL` rebuilds an entity's text from its pieces and
//! searches the chunk's text for it, this takes the pieces' own offsets,
//! so an entity is exactly the bytes its pieces cover.

/// One label of the model's label map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Label {
    /// `O` or any label that is not `B-` or `I-` with a type.
    Outside,
    /// `B-<type>`, the type's index in the finder's types.
    Begin(usize),
    /// `I-<type>`.
    Inside(usize),
}

/// The labels of a model's label map, in id order, and the entity types
/// they name, in order of first appearance.
pub fn labels(names: &[String]) -> (Vec<Label>, Vec<String>) {
    let mut types: Vec<String> = Vec::new();
    let mut ty = |t: &str| match types.iter().position(|x| x == t) {
        Some(i) => i,
        None => {
            types.push(t.to_owned());
            types.len() - 1
        }
    };
    let labels = names
        .iter()
        .map(|n| match (n.strip_prefix("B-"), n.strip_prefix("I-")) {
            (Some(t), _) if !t.is_empty() => Label::Begin(ty(t)),
            (_, Some(t)) if !t.is_empty() => Label::Inside(ty(t)),
            _ => Label::Outside,
        })
        .collect();
    (labels, types)
}

/// One entity candidate: its bytes in the text, its type and probability.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Candidate {
    pub start: usize,
    pub end: usize,
    pub kind: usize,
    pub probability: f64,
}

/// `AbstractDL.chunkRanges`: overlapping ranges over `count` words.
pub fn chunk_ranges(count: usize, split: usize, overlap: usize) -> Result<Vec<(usize, usize)>, String> {
    if split == 0 || overlap >= split {
        return Err(format!("a split of {split} words cannot advance with an overlap of {overlap}"));
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < count {
        let end = (start + split).min(count);
        out.push((start, end));
        start = if end == count { end } else { end - overlap };
    }
    Ok(out)
}

/// The byte ranges of the whitespace-separated words of `text`, Unicode's
/// `White_Space` as OpenNLP's `CharClass.whitespace` holds it.
pub fn words(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

/// `NameFinderDL`'s `maxIndex`: the first highest score that is not NaN.
pub fn max_index(scores: &[f32]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_nan() && best.is_none_or(|b| s > scores[b]) {
            best = Some(i);
        }
    }
    best
}

/// `NameFinderDL.labelProbability`: label `index`'s softmax probability,
/// NaN scores left out; with any positive infinity, the infinities share
/// it; with no finite or infinite maximum, every label is equally likely.
pub fn label_probability(scores: &[f32], index: usize) -> f64 {
    let infinite = scores.iter().filter(|&&s| s == f32::INFINITY).count();
    if infinite > 0 {
        return if scores[index] == f32::INFINITY { 1.0 / infinite as f64 } else { 0.0 };
    }
    let max = scores.iter().filter(|s| !s.is_nan()).fold(f64::from(f32::NEG_INFINITY), |m, &s| m.max(f64::from(s)));
    if max == f64::from(f32::NEG_INFINITY) {
        return 1.0 / scores.len() as f64;
    }
    let sum: f64 = scores.iter().filter(|s| !s.is_nan()).map(|&s| (f64::from(s) - max).exp()).sum();
    (f64::from(scores[index]) - max).exp() / sum
}

/// The candidates of one chunk: `pieces` are each piece's bytes in the text
/// (`None` for `[CLS]` and `[SEP]`), `scores` its label scores row by row.
///
/// # Errors
/// A row with no score that is not NaN, or a row count other than the
/// pieces'.
pub fn decode(pieces: &[Option<(usize, usize)>], scores: &[f32], labels: &[Label]) -> Result<Vec<Candidate>, String> {
    let n = labels.len();
    if n == 0 || scores.len() != pieces.len() * n {
        return Err(format!("{} scores for {} pieces of {n} labels", scores.len(), pieces.len()));
    }
    let predict = |x: usize| -> Result<(&Label, f64), String> {
        let row = &scores[x * n..(x + 1) * n];
        let i = max_index(row).ok_or_else(|| format!("piece {x} has no score that is a number"))?;
        Ok((&labels[i], label_probability(row, i)))
    };
    let mut out = Vec::new();
    let mut x = 0;
    while x < pieces.len() {
        let (label, p) = predict(x)?;
        let Label::Begin(kind) = *label else {
            x += 1;
            continue;
        };
        let (mut end, mut probability) = (x, p);
        while end + 1 < pieces.len() {
            let (next, q) = predict(end + 1)?;
            if *next != Label::Inside(kind) {
                break;
            }
            end += 1;
            probability = probability.min(q);
        }
        let covered: Vec<(usize, usize)> = pieces[x..=end].iter().flatten().copied().collect();
        if let (Some(first), Some(last)) = (covered.first(), covered.last()) {
            out.push(Candidate { start: first.0, end: last.1, kind, probability });
        }
        x = end + 1;
    }
    Ok(out)
}

/// `NameFinderDL.mergeOverlappingSpans`: longest first (in UTF-16 units of
/// `text`, as Java measures a span), then most probable, each kept unless
/// it overlaps one kept before it; the result in text order.
pub fn resolve(text: &str, mut candidates: Vec<Candidate>) -> Vec<Candidate> {
    let utf16 = |c: &Candidate| text[c.start..c.end].encode_utf16().count();
    candidates.sort_by(|a, b| utf16(b).cmp(&utf16(a)).then(b.probability.total_cmp(&a.probability)));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in candidates {
        if kept.iter().all(|k| !(k.start < c.end && c.start < k.end)) {
            kept.push(c);
        }
    }
    kept.sort_by_key(|c| c.start);
    kept
}

#[cfg(test)]
mod tests;
