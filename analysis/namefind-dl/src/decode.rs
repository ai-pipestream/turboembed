//! `NameFinderDL`'s decoding, on the model's per-piece label scores:
//!
//! - the text is cut into chunks of whitespace-separated words, at most
//!   `split` words each, consecutive chunks sharing `overlap` words;
//! - each piece's label is the first highest score that is not NaN, and its
//!   probability that label's softmax probability (with OpenNLP's handling
//!   of infinities);
//! - by default (`Decoding::Word`) each word takes its first piece's label,
//!   as the model was trained, and the units below are words; with
//!   `Decoding::Piece` every piece is a unit, as `NameFinderDL` decodes;
//! - a `B-<type>` unit starts an entity that runs over the `I-<type>`
//!   units after it, with the lowest probability among them;
//! - candidates from every chunk are resolved longest first, then most
//!   probable, and a candidate overlapping one already kept is dropped.
//!
//! Where `NameFinderDL` rebuilds an entity's text from its pieces and
//! searches the chunk's text for it, this takes the pieces' own offsets,
//! so an entity is exactly the bytes its pieces cover.

use std::collections::BTreeMap;

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

/// What a label is read for.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Decoding {
    /// Each word takes its first piece's label, as the model was trained:
    /// an entity is whole words, and a `B-` the model gives a piece inside
    /// a word does not start one there.
    #[default]
    Word,
    /// Every piece takes its own label, as OpenNLP's `NameFinderDL`
    /// decodes: for parity with it.
    Piece,
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

/// The rows a chunk of `count` pieces is read in, `width` pieces to a row:
/// one row when they fit; otherwise rows overlapping by half a row, each
/// piece's label taken from the row where it sits furthest from a cut
/// edge. Each row is `(start, end, own_start, own_end)`: the pieces it
/// reads and the pieces whose labels it gives, which run on from the
/// row before's.
pub fn row_ranges(count: usize, width: usize) -> Vec<(usize, usize, usize, usize)> {
    if count <= width || width == 0 {
        return vec![(0, count, 0, count)];
    }
    let margin = width / 4;
    let step = width - 2 * margin;
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + width).min(count);
        let own_start = if start == 0 { 0 } else { start + margin };
        let own_end = if end == count { count } else { end - margin };
        out.push((start, end, own_start, own_end));
        if end == count {
            return out;
        }
        start += step;
    }
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

/// The candidates of one chunk, every piece decoded: `pieces` are each
/// piece's bytes in the text (`None` for `[CLS]` and `[SEP]`), `scores` its
/// label scores row by row.
///
/// # Errors
/// A row with no score that is not NaN, or a row count other than the
/// pieces'.
pub fn decode(pieces: &[Option<(usize, usize)>], scores: &[f32], labels: &[Label]) -> Result<Vec<Candidate>, String> {
    let predict = predictor(pieces.len(), scores, labels)?;
    let units = (0..pieces.len()).map(|x| predict(x).map(|(l, p)| (pieces[x], l, p))).collect::<Result<Vec<_>, _>>()?;
    Ok(chain(&units))
}

/// The candidates of one chunk, each word labelled by its first piece:
/// `words` is each piece's word in the chunk (`None` for a special token,
/// which is no unit), a word's pieces consecutive. A word covers the bytes
/// of all its pieces.
///
/// # Errors
/// As `decode`, or a word count other than the pieces'.
pub fn decode_words(
    pieces: &[Option<(usize, usize)>],
    words: &[Option<usize>],
    scores: &[f32],
    labels: &[Label],
) -> Result<Vec<Candidate>, String> {
    if words.len() != pieces.len() {
        return Err(format!("{} words for {} pieces", words.len(), pieces.len()));
    }
    let predict = predictor(pieces.len(), scores, labels)?;
    let mut units: Vec<Unit<'_>> = Vec::new();
    let mut last = None;
    for (x, w) in words.iter().enumerate() {
        let Some(w) = *w else { continue };
        if last == Some(w) {
            let unit = units.last_mut().expect("a word before");
            unit.0 = match (unit.0, pieces[x]) {
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
                (a, b) => a.or(b),
            };
            continue;
        }
        let (label, p) = predict(x)?;
        units.push((pieces[x], label, p));
        last = Some(w);
    }
    Ok(chain(&units))
}

/// A piece or word to decode: its bytes, label and probability.
type Unit<'a> = (Option<(usize, usize)>, &'a Label, f64);

/// Each piece's label and probability, after checking the score count.
fn predictor<'a>(
    pieces: usize,
    scores: &'a [f32],
    labels: &'a [Label],
) -> Result<impl Fn(usize) -> Result<(&'a Label, f64), String> + 'a, String> {
    let n = labels.len();
    if n == 0 || scores.len() != pieces * n {
        return Err(format!("{} scores for {pieces} pieces of {n} labels", scores.len()));
    }
    Ok(move |x: usize| {
        let row = &scores[x * n..(x + 1) * n];
        let i = max_index(row).ok_or_else(|| format!("piece {x} has no score that is a number"))?;
        Ok((&labels[i], label_probability(row, i)))
    })
}

/// The entities of a run of units: a `B-<type>` unit and the `I-<type>`
/// units after it, the bytes the units cover and their lowest probability.
fn chain(units: &[Unit<'_>]) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut x = 0;
    while x < units.len() {
        let (_, label, p) = units[x];
        let Label::Begin(kind) = *label else {
            x += 1;
            continue;
        };
        let (mut end, mut probability) = (x, p);
        while end + 1 < units.len() && *units[end + 1].1 == Label::Inside(kind) {
            end += 1;
            probability = probability.min(units[end].2);
        }
        let covered: Vec<(usize, usize)> = units[x..=end].iter().filter_map(|u| u.0).collect();
        if let (Some(first), Some(last)) = (covered.first(), covered.last()) {
            out.push(Candidate { start: first.0, end: last.1, kind, probability });
        }
        x = end + 1;
    }
    out
}

/// `NameFinderDL.mergeOverlappingSpans`: longest first (in UTF-16 units of
/// `text`, as Java measures a span), then most probable, each kept unless
/// it overlaps one kept before it; the result in text order.
pub fn resolve(text: &str, mut candidates: Vec<Candidate>) -> Vec<Candidate> {
    let utf16 = |c: &Candidate| text[c.start..c.end].encode_utf16().count();
    candidates.sort_by(|a, b| utf16(b).cmp(&utf16(a)).then(b.probability.total_cmp(&a.probability)));
    // Kept spans do not overlap, so only the one starting at or before a
    // candidate and the one after it can overlap it.
    let mut kept: BTreeMap<usize, Candidate> = BTreeMap::new();
    for c in candidates {
        let before = kept.range(..=c.start).next_back().is_some_and(|(_, k)| c.start < k.end);
        let after = kept.range(c.start..).next().is_some_and(|(_, k)| k.start < c.end);
        // An empty candidate overlaps nothing, but would share a start.
        if !before && !after && !kept.contains_key(&c.start) {
            kept.insert(c.start, c);
        }
    }
    kept.into_values().collect()
}

#[cfg(test)]
mod tests;
