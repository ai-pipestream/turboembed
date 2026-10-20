//! YAKE's occurrence fold and term statistics (Campos et al. 2020,
//! section 3): one pass merges tokens against sentences and folds the
//! term table, one pass gathers the co-occurrence windows the paper
//! defines, then the five features combine into `TScore`.

use std::collections::HashMap;

use turbo_analysis::{Annotator, Error, Layer};
use turbo_analysis_layers::{Sentences, Terms, Tokens};

/// Token tags, computed from the surface form and the token's position
/// in its sentence. `D` numbers and `U` unknowns are excluded from
/// candidates; `A` acronyms, `N` mid-sentence capitalized and `P` plain
/// words form the candidate vocabulary.
pub(crate) const TAG_D: u8 = 0;
pub(crate) const TAG_U: u8 = 1;
pub(crate) const TAG_A: u8 = 2;
pub(crate) const TAG_N: u8 = 3;
pub(crate) const TAG_P: u8 = 4;

/// Marks accepted inside a word without making it unknown: apostrophes,
/// hyphens, periods and commas (so "1,000" parses as a number first and
/// "don't" stays a word; every other symbol tags the token `U`).
const INTERNAL_MARKS: [char; 5] = ['\'', '\u{2019}', '-', '.', ','];

/// Classifies one token by its surface text and its position in its
/// sentence: `d` when the whole token parses as a number, `u` when
/// letters mix with digits, no letter is present, or a code point is
/// neither letter nor digit nor an accepted internal mark, `a` when more
/// than one character long and every cased character is upper case, `n`
/// when more than one character long, exactly one character is upper
/// case and it is the first, and the in-sentence position is above zero,
/// `p` otherwise.
pub(crate) fn classify(surface: &str, pos_in_sentence: usize) -> u8 {
    let chars: Vec<char> = surface.chars().collect();
    if !chars.is_empty() && surface.replace(',', "").parse::<f64>().is_ok() {
        return TAG_D;
    }
    let letters = chars.iter().filter(|c| c.is_alphabetic()).count();
    let digits = chars.iter().filter(|c| c.is_numeric()).count();
    if (letters > 0 && digits > 0)
        || letters + digits == 0
        || chars.iter().any(|c| !c.is_alphabetic() && !c.is_numeric() && !INTERNAL_MARKS.contains(c))
    {
        return TAG_U;
    }
    let upper = chars.iter().filter(|c| c.is_uppercase()).count();
    if chars.len() > 1 && upper + chars.iter().filter(|c| c.is_lowercase()).count() == upper {
        return TAG_A;
    }
    if chars.len() > 1 && upper == 1 && chars[0].is_uppercase() && pos_in_sentence > 0 {
        return TAG_N;
    }
    TAG_P
}

/// The folded terms, one row per distinct folded form: its frequency,
/// the sentence index of every occurrence (for the median position), how
/// often it appeared as an acronym or a mid-sentence capitalized word,
/// and whether any occurrence was tagged a content tag (`p`, `a`, `n`).
#[derive(Default)]
pub(crate) struct TermTable {
    pub(crate) ids: HashMap<String, u32>,
    pub(crate) names: Vec<String>,
    pub(crate) tf: Vec<u64>,
    pub(crate) sentence_ixs: Vec<Vec<u32>>,
    pub(crate) acronym: Vec<u64>,
    pub(crate) proper: Vec<u64>,
    pub(crate) content: Vec<bool>,
}

impl TermTable {
    fn intern(&mut self, folded: &str) -> u32 {
        if let Some(&id) = self.ids.get(folded) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(folded.to_owned(), id);
        self.names.push(folded.to_owned());
        self.tf.push(0);
        self.sentence_ixs.push(Vec::new());
        self.acronym.push(0);
        self.proper.push(0);
        self.content.push(false);
        id
    }

    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }
}

/// What the fold produced: per-token sentence, term and tag, the term
/// table, the token span columns (borrowed from the tokens layer, for
/// occurrence spans), the left and right co-occurrence counts keyed by
/// (center, neighbor) term id, and the sentence count.
pub(crate) struct Fold<'t> {
    pub(crate) tok_start: &'t [u32],
    pub(crate) tok_end: &'t [u32],
    pub(crate) sentence_of: Vec<u32>,
    pub(crate) term_of: Vec<u32>,
    pub(crate) tag_of: Vec<u8>,
    pub(crate) table: TermTable,
    pub(crate) left: HashMap<(u32, u32), u64>,
    pub(crate) right: HashMap<(u32, u32), u64>,
    pub(crate) sentence_count: usize,
}

/// Folds the term table and co-occurrence windows of `window_size`
/// tokens to each side, over the merge-walk of [`crate::fold`]. A terms
/// layer not aligned with the tokens fails the document.
pub(crate) fn fold<'t>(
    sentences: &'t Sentences,
    tokens: &'t Tokens,
    terms: &Terms,
    window_size: usize,
) -> Result<Fold<'t>, Error> {
    if terms.len() != tokens.spans.len() {
        return Err(Error::Annotator {
            annotator: crate::YakeAnnotator::ID,
            detail: format!("{} terms and {} tokens", terms.len(), tokens.spans.len()),
        });
    }
    let (tok_start, tok_end) = tokens.spans.columns();
    let n = tok_start.len();
    let sentence_of = crate::fold::sentence_of(sentences, tokens, crate::YakeAnnotator::ID)?;
    // The in-sentence position: the offset from the sentence's first
    // token (sentence indices do not decrease along the tokens).
    let mut sentence_head = vec![0usize; sentences.spans.len() + 1];
    for (i, &s) in sentence_of.iter().enumerate() {
        sentence_head[s as usize + 1] = i + 1;
    }
    let mut term_of = vec![0u32; n];
    let mut tag_of = vec![0u8; n];
    let mut table = TermTable::default();

    for i in 0..n {
        let pos = i - sentence_head[sentence_of[i] as usize];
        let folded = terms.folded.get(i);
        let id = table.intern(folded) as usize;
        term_of[i] = id as u32;
        table.tf[id] += 1;
        table.sentence_ixs[id].push(sentence_of[i]);
        tag_of[i] = classify(terms.surface.get(i), pos);
        if tag_of[i] >= TAG_A {
            table.content[id] = true;
            if tag_of[i] == TAG_A {
                table.acronym[id] += 1;
            } else if tag_of[i] == TAG_N {
                table.proper[id] += 1;
            }
        }
    }

    // Co-occurrence, now that every token has its sentence: the terms at
    // [i - w, i) and (i, i + w] within the same sentence, stop words
    // included; they are context, only candidates get excluded.
    let (mut left, mut right) = (HashMap::new(), HashMap::new());
    for i in 0..n {
        for j in i.saturating_sub(window_size)..i {
            if sentence_of[j] == sentence_of[i] {
                *left.entry((term_of[i], term_of[j])).or_default() += 1;
            }
        }
        for j in i + 1..(i + 1 + window_size).min(n) {
            if sentence_of[j] != sentence_of[i] {
                break;
            }
            *right.entry((term_of[i], term_of[j])).or_default() += 1;
        }
    }
    Ok(Fold {
        tok_start,
        tok_end,
        sentence_of,
        term_of,
        tag_of,
        table,
        left,
        right,
        sentence_count: sentences.spans.len(),
    })
}

/// The median of `values` as an `f64`: the middle value, or the mean of
/// the two middle values.
pub(crate) fn median(values: &mut [u32]) -> f64 {
    values.sort_unstable();
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 { values[n / 2] as f64 } else { (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0 }
}

/// Per-term (distinct neighbors, total count) for one side of the
/// window, gathered in one pass over the count map; empty sides read
/// (0, 0) and turn into a dispersion of 0.
pub(crate) fn side_totals(counts: &HashMap<(u32, u32), u64>, terms: usize) -> Vec<(u64, u64)> {
    let mut agg = vec![(0u64, 0u64); terms];
    for (&(center, _), &n) in counts {
        let slot = &mut agg[center as usize];
        slot.0 += 1;
        slot.1 += n;
    }
    agg
}

/// `TScore` per term, over the candidate vocabulary only: terms with a
/// content-tagged occurrence that are not stop words. The frequency
/// moments are taken over that vocabulary, not over every token, where
/// stop word frequencies would flatten `TFNorm` for every content term
/// (a deliberate departure from the reference implementation, toward the
/// paper). Sentence indices are 0-based, the layer's own numbering.
/// Terms outside the vocabulary keep `f64::INFINITY` and are never read:
/// a surviving candidate window holds only scored terms and stop words.
pub(crate) fn score_terms(fold: &Fold<'_>, stopwords: &crate::Stopwords) -> Vec<f64> {
    let n = fold.table.len();
    let candidate: Vec<bool> =
        (0..n).map(|id| fold.table.content[id] && !stopwords.contains(&fold.table.names[id])).collect();
    let moments = {
        let cands: Vec<f64> = (0..n).filter(|&id| candidate[id]).map(|id| fold.table.tf[id] as f64).collect();
        let mean = cands.iter().sum::<f64>() / cands.len().max(1) as f64;
        let max = cands.iter().copied().fold(0.0, f64::max);
        let std = if !cands.is_empty() {
            (cands.iter().map(|tf| (tf - mean).powi(2)).sum::<f64>() / cands.len() as f64).sqrt()
        } else {
            0.0
        };
        (mean, std, max)
    };
    let (mean, std, max) = moments;
    let s = fold.sentence_count.max(1) as f64;
    let left = side_totals(&fold.left, n);
    let right = side_totals(&fold.right, n);

    let mut tscore = vec![f64::INFINITY; n];
    for id in 0..n {
        if !candidate[id] {
            continue;
        }
        let tf = fold.table.tf[id] as f64;
        let tcase = fold.table.acronym[id].max(fold.table.proper[id]) as f64 / (1.0 + tf.ln());
        let mut ixs = fold.table.sentence_ixs[id].clone();
        let tposition = (3.0 + median(&mut ixs)).ln().ln();
        let tfnorm = tf / if mean + std > 0.0 { mean + std } else { 1.0 };
        let mut distinct = fold.table.sentence_ixs[id].clone();
        distinct.sort_unstable();
        distinct.dedup();
        let tsentence = distinct.len() as f64 / s;
        let dl = if left[id].1 > 0 { left[id].0 as f64 / left[id].1 as f64 } else { 0.0 };
        let dr = if right[id].1 > 0 { right[id].0 as f64 / right[id].1 as f64 } else { 0.0 };
        let trel = 1.0 + (dl + dr) * tf / max;
        tscore[id] = tposition * trel / (tcase + tfnorm / trel + tsentence / trel);
    }
    tscore
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_follow_the_code_point_rules() {
        assert_eq!(classify("123", 0), TAG_D);
        assert_eq!(classify("1,000", 1), TAG_D);
        assert_eq!(classify("abc123", 0), TAG_U);
        assert_eq!(classify("!!!", 0), TAG_U);
        assert_eq!(classify("USA", 1), TAG_A);
        assert_eq!(classify("U", 1), TAG_P, "one character is not an acronym");
        assert_eq!(classify("Google", 3), TAG_N);
        assert_eq!(classify("Google", 0), TAG_P, "a sentence-initial capital is not");
        assert_eq!(classify("google", 0), TAG_P);
        assert_eq!(classify("don't", 2), TAG_P);
    }

    #[test]
    fn median_of_sentence_indices() {
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(median(&mut [0]), 0.0);
        assert_eq!(median(&mut [0, 4]), 2.0);
        assert_eq!(median(&mut [0, 1, 4]), 1.0);
    }
}

#[cfg(test)]
mod term_score_tests {
    //! The hand computation: three sentences, a two-token window, no
    //! stopwords, and every feature of every term checked against values
    //! worked out from the paper's formulas (an independent Python
    //! reference, itself checked by hand for one term). This is the test
    //! that pins the deliberate departures: the ln(ln(3 + median))
    //! position, DR from the right window's own dispersion, and the
    //! frequency moments over the candidate vocabulary.

    use super::*;
    use crate::Stopwords;
    use turbo_analysis::{Span, Spans, ViewId};
    use turbo_analysis_layers::Strs;

    /// "red Fish blue fish swim deep red swim Fish blue red": three
    /// sentences [0, 18), [19, 37), [38, 51), eleven tokens at the
    /// spaces, folded forms lowercased.
    fn fixture() -> (Sentences, Tokens, Terms) {
        let mut s = Spans::new(ViewId::SOURCE);
        for (a, b) in [(0u32, 18u32), (19, 37), (38, 51)] {
            s.push(Span::new(a, b));
        }
        let mut t = Spans::new(ViewId::SOURCE);
        for (a, b) in [
            (0u32, 3u32),
            (4, 8),
            (9, 13),
            (14, 18),
            (19, 23),
            (24, 28),
            (29, 32),
            (33, 37),
            (38, 42),
            (43, 47),
            (48, 51),
        ] {
            t.push(Span::new(a, b));
        }
        let mut surface = Strs::default();
        let mut folded = Strs::default();
        for token in ["red", "Fish", "blue", "fish", "swim", "deep", "red", "swim", "Fish", "blue", "red"] {
            surface.push(token);
            folded.push(&token.to_lowercase());
        }
        (
            Sentences { spans: s },
            Tokens { spans: t, sentence: None },
            Terms { surface, folded, stem: None, lemma: None },
        )
    }

    fn none() -> Stopwords {
        Stopwords::custom(["qwerty"]).unwrap()
    }

    #[test]
    fn folds_terms_tags_and_windows() {
        let (s, t, terms) = fixture();
        let f = fold(&s, &t, &terms, 2).unwrap();
        let id = |name: &str| f.table.ids[name];
        // Frequencies, casing counts, sentence indices.
        for (name, tf, proper) in [("red", 3, 0), ("fish", 3, 1), ("blue", 2, 0), ("swim", 2, 0), ("deep", 1, 0)] {
            let i = id(name) as usize;
            assert_eq!(f.table.tf[i], tf, "{name} tf");
            assert_eq!(f.table.proper[i], proper, "{name} proper");
            assert!(f.table.acronym[i] == 0, "{name} acronym");
            assert!(f.table.content[i], "{name} content");
        }
        assert_eq!(f.table.sentence_ixs[id("red") as usize], [0, 1, 2]);
        assert_eq!(f.table.sentence_ixs[id("fish") as usize], [0, 0, 2]);
        assert_eq!(f.table.sentence_ixs[id("swim") as usize], [1, 1]);
        // Tag rules: mid-sentence capitals are n, sentence-initial p.
        assert_eq!(f.tag_of[1], TAG_N, "Fish mid-sentence");
        assert_eq!(f.tag_of[8], TAG_P, "Fish at sentence start");
        assert_eq!(f.tag_of[0], TAG_P);
        // The paper's window around fish: left red, fish, blue (the
        // cross-sentence half of sentence 2 stays out), right blue x2,
        // fish, red.
        let fish = id("fish");
        for (neighbor, n) in [("red", 1), ("fish", 1), ("blue", 1)] {
            assert_eq!(f.left.get(&(fish, id(neighbor))), Some(&n), "left fish->{neighbor}");
        }
        for (neighbor, n) in [("blue", 2), ("fish", 1), ("red", 1)] {
            assert_eq!(f.right.get(&(fish, id(neighbor))), Some(&n), "right fish->{neighbor}");
        }
        assert_eq!(f.left.get(&(fish, id("swim"))), None, "no edge across sentences");
    }

    #[test]
    fn term_scores_match_the_hand_computation() {
        let (s, t, terms) = fixture();
        let f = fold(&s, &t, &terms, 2).unwrap();
        let scores = score_terms(&f, &none());
        let at = |name: &str| scores[f.table.ids[name] as usize];
        // Worked from the formulas: meanTF 2.2, stdTF 0.748331477355
        // (population), maxTF 3; fish alone carries a casing count (its
        // mid-sentence "Fish") and the lowest position (median sentence
        // 0, ln(ln 3)).
        for (name, want) in [
            ("blue", 1.082321595975),
            ("deep", 1.349154344013),
            ("fish", 0.237507911963),
            ("red", 1.457086691534),
            ("swim", 1.757805447230),
        ] {
            assert!((at(name) - want).abs() < 1e-9, "{name}: {} != {want}", at(name));
        }
    }

    #[test]
    fn the_window_size_is_the_papers_and_read() {
        // The reference implementation declares the window and never
        // reads it; here window 1, 2 and 3 give fish and red three
        // different dispersions, hence three different TRel values.
        let (s, t, terms) = fixture();
        let trel = |w: usize, name: &str| {
            let f = fold(&s, &t, &terms, w).unwrap();
            let id = f.table.ids[name] as usize;
            let tf = f.table.tf[id] as f64;
            let side = |counts: &HashMap<(u32, u32), u64>| {
                let (mut d, mut tot) = (0u64, 0u64);
                for (&(c, _), &n) in counts {
                    if c as usize == id {
                        d += 1;
                        tot += n;
                    }
                }
                if tot > 0 { d as f64 / tot as f64 } else { 0.0 }
            };
            1.0 + (side(&f.left) + side(&f.right)) * tf / 3.0
        };
        assert!((trel(1, "fish") - 2.5).abs() < 1e-12);
        assert!((trel(2, "fish") - 2.75).abs() < 1e-12);
        assert!((trel(3, "fish") - 2.5).abs() < 1e-12);
        assert!((trel(1, "red") - 3.0).abs() < 1e-12);
        assert!((trel(3, "red") - 2.75).abs() < 1e-12);
    }
}
