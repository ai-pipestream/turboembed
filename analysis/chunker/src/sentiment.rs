//! Sentiment, aspect, and incivility shift as a soft boundary feature on [`Boundaries`],
//! plus [`ChunkFacets`] exposing per-chunk polarity, aspects, and incivility labels.
//!
//! Generalizes the `ChunkEmbedProcessor` and `Centroids` sentence grouping logic from
//! `opennlp-sandbox` (`OPENNLP-1833-grpc-uber-server`): instead of forcing a hard chunk cut
//! whenever the sentence sentiment label changes, sentiment and aspect shifts write a
//! named [`Boundaries`] instance ([`SENTIMENT`]) that [`weigh_with_sentiment`](crate::weigh_with_sentiment)
//! and [`WeighAllBoundaries`](crate::WeighAllBoundaries) combine with structure and embedding drift.
//!
//! Because [`WeighAllBoundaries::default`](crate::WeighAllBoundaries) sets `sentiment: Weight(0)`,
//! this signal is off by default unless explicitly weighted in a pipeline specification.

mod facets;
#[cfg(test)]
mod tests;

pub use facets::{ChunkFacetBuilder, ChunkFacets, boundary_similarity, chunk_boundary_gaps};

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Bound, Error, Fnv1a, Ix, Layer, Refs, View};
use turbo_analysis_layers::{
    Aspects, Boundaries, Categories, Cut, Incivility, IncivilityLabel, IncivilityScores, IncivilityTarget, Polarity,
    Sentences, Tokens,
};

use crate::weigh::SENTIMENT;

pub(super) const MICRO: i64 = 1_000_000;
const MILLI: i64 = 1_000;

/// Parameters for [`sentiment_shift`] and [`SentimentBoundaries`].
///
/// All weights are in thousandths (`1000` = `1.0`) and thresholds are in millionths
/// (`1_000_000` = `1.0`), matching [`Drift`](crate::Drift) so boundary scoring is
/// platform-deterministic integer arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SentimentShift {
    /// Sentences on each side of a gap to pool when comparing sentiment, aspect, and
    /// incivility profiles across the gap. Default `1`.
    pub window: NonZeroU32,
    /// Weight in thousandths for sentence/aspect polarity reversal across a gap. Default `1000`.
    pub polarity_weight: u32,
    /// Weight in thousandths for aspect category/target shift or opposite aspect polarity
    /// on the same aspect category across a gap. Default `1000`.
    pub aspect_weight: u32,
    /// Weight in thousandths for incivility onset/offset or label change across a gap. Default `500`.
    pub incivility_weight: u32,
    /// Minimum combined shift in millionths before taking cost off a gap. Default `100_000` (`0.10`).
    pub floor: u32,
    /// Thousandths of a [`Boundaries::UNIT`] taken off a gap per unit shift:
    /// `cost = -(shift * scale / 1_000_000)`. Default `Boundaries::UNIT as u32` (`1000`).
    pub scale: u32,
    /// Minimum combined shift in millionths to label the gap [`Cut::TopicShift`].
    /// Default `500_000` (`0.50`).
    pub shift: u32,
}

impl Default for SentimentShift {
    fn default() -> Self {
        Self {
            window: NonZeroU32::new(1).expect("1 > 0"),
            polarity_weight: 1000,
            aspect_weight: 1000,
            incivility_weight: 500,
            floor: 100_000,
            scale: Boundaries::UNIT as u32,
            shift: 500_000,
        }
    }
}

/// Summary of sentiment, aspect, and incivility signals for a single sentence.
#[derive(Clone, Debug, Default)]
struct SentenceSignal {
    /// Signed polarity in `[-1_000_000, 1_000_000]` millionths (`None` if no polarity signal).
    polarity_micro: Option<i64>,
    /// Aspect `(normalized_key, Polarity)` pairs anchored in this sentence.
    aspects: Vec<(String, Polarity)>,
    /// Maximum incivility score in millionths `[0, 1_000_000]` and bitmask of flagged labels.
    incivility_micro: Option<(i64, u8)>,
}

pub(super) fn label_bit(label: IncivilityLabel) -> u8 {
    1u8 << (label as u8)
}

fn max_incivility_score(scores: IncivilityScores) -> f64 {
    scores.iter().into_iter().map(|(_, s)| if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 }).fold(0.0f64, f64::max)
}

pub(super) fn category_row_polarity_micro(cats: &Categories<Sentences>, i: usize) -> Option<i64> {
    let row = cats.get(i)?;
    let pos = row.score_of("positive").or_else(|| row.score_of("pos"));
    let neg = row.score_of("negative").or_else(|| row.score_of("neg"));
    if let (Some(p), Some(n)) = (pos, neg)
        && p.is_finite()
        && n.is_finite()
    {
        return Some(((p - n).clamp(-1.0, 1.0) * (MICRO as f64)).round() as i64);
    }
    let best = row.best;
    let sc = row.best_score();
    let mag = if sc.is_finite() { (sc.clamp(0.0, 1.0) * (MICRO as f64)).round() as i64 } else { MICRO };
    let mag = mag.clamp(1, MICRO);
    if best.eq_ignore_ascii_case("positive") || best.eq_ignore_ascii_case("pos") {
        Some(mag)
    } else if best.eq_ignore_ascii_case("negative") || best.eq_ignore_ascii_case("neg") {
        Some(-mag)
    } else if best.eq_ignore_ascii_case("neutral") || best.eq_ignore_ascii_case("neu") {
        Some(0)
    } else {
        None
    }
}

pub(super) fn resolve_aspect_sentence(
    a_ix: Ix<Aspects>,
    aspects: &Aspects,
    sents: &Sentences,
    toks: Option<&Tokens>,
) -> Option<usize> {
    let n = sents.spans.len();
    if n == 0 {
        return None;
    }
    let target = aspects.target(a_ix);
    if let Some(s_ix) = target.sentence() {
        return (s_ix.get() < n).then_some(s_ix.get());
    }
    let r = target.tokens()?;
    let toks = toks?;
    if r.start.get() >= toks.spans.len() {
        return None;
    }
    if let Some(sent_col) = &toks.sentence
        && let Some(&s_ix) = sent_col.get(r.start.get())
    {
        return (s_ix.get() < n).then_some(s_ix.get());
    }
    let tok_start = toks.spans.get(r.start).start;
    sents.spans.covering_at(tok_start).map(|i| i.get()).filter(|&i| i < n)
}

pub(super) fn resolve_aspect_label(
    a_ix: Ix<Aspects>,
    aspects: &Aspects,
    toks: Option<&Tokens>,
    view: Option<&View>,
) -> String {
    if let Some(cat) = aspects.category_label(a_ix)
        && !cat.is_empty()
    {
        return cat.to_owned();
    }
    if let (Some(r), Some(toks), Some(v)) = (aspects.target(a_ix).tokens(), toks, view)
        && r.start.get() < toks.spans.len()
        && r.end.get() <= toks.spans.len()
        && !r.is_empty()
    {
        let start = toks.spans.get(r.start).start as usize;
        let last = Ix::new(r.end.get() as u32 - 1);
        let end = toks.spans.get(last).end as usize;
        if let Some(slice) = v.text().get(start..end) {
            return slice.trim().to_owned();
        }
    }
    String::new()
}

/// Extracts per-sentence signals from optional `Categories<Sentences>`, `Aspects`, and `Incivility`.
fn extract_sentence_signals(
    sents: &Sentences,
    view: Option<&View>,
    toks: Option<&Tokens>,
    cats: Option<&Categories<Sentences>>,
    aspects: Option<&Aspects>,
    incivility: Option<&Incivility>,
) -> Vec<SentenceSignal> {
    let n = sents.spans.len();
    let mut out = vec![SentenceSignal::default(); n];
    if n == 0 {
        return out;
    }

    if let Some(cats) = cats {
        for (i, sig) in out.iter_mut().enumerate().take(cats.len().min(n)) {
            if let Some(p) = category_row_polarity_micro(cats, i) {
                sig.polarity_micro = Some(p);
            }
        }
    }

    if let Some(asp) = aspects {
        let mut compound_sum = vec![0i64; n];
        let mut compound_cnt = vec![0i64; n];
        for (a_ix, item) in asp.iter() {
            let Some(s_idx) = resolve_aspect_sentence(a_ix, asp, sents, toks) else {
                continue;
            };
            let key = resolve_aspect_label(a_ix, asp, toks, view).to_ascii_lowercase();
            let pol = item.polarity.label();
            out[s_idx].aspects.push((key, pol));
            let c_micro = (item.polarity.compound().clamp(-1.0, 1.0) * (MICRO as f64)).round() as i64;
            compound_sum[s_idx] += c_micro;
            compound_cnt[s_idx] += 1;
        }
        for i in 0..n {
            if out[i].polarity_micro.is_none() && compound_cnt[i] > 0 {
                out[i].polarity_micro = Some(compound_sum[i] / compound_cnt[i]);
            }
        }
    }

    if let Some(inc) = incivility {
        for s in &mut out {
            s.incivility_micro = Some((0, 0));
        }
        for i in 0..inc.len() {
            let ix = Ix::<Incivility>::new(i as u32);
            if let IncivilityTarget::Sentence(s_ix) = inc.target[i]
                && s_ix.get() < n
            {
                let sc = inc.scores(ix);
                let max_s = max_incivility_score(sc);
                let max_micro = (max_s * (MICRO as f64)).round() as i64;
                let mut mask = 0u8;
                for (lbl, _) in inc.evidence_of(ix) {
                    mask |= label_bit(lbl);
                }
                for (lbl, val) in sc.iter() {
                    if val >= 0.5 {
                        mask |= label_bit(lbl);
                    }
                }
                let prev = out[s_ix.get()].incivility_micro.unwrap_or((0, 0));
                out[s_ix.get()].incivility_micro = Some((prev.0.max(max_micro), prev.1 | mask));
            }
        }
    }

    out
}

/// Computes the shift in millionths `[0, 1_000_000]` across gap `g` (between `0..=g` and `g+1..`).
fn gap_shift_micro(signals: &[SentenceSignal], g: usize, cfg: SentimentShift) -> i64 {
    let w = cfg.window.get() as usize;
    let l_start = (g + 1).saturating_sub(w);
    let left = &signals[l_start..=g];
    let r_end = (g + 1 + w).min(signals.len());
    let right = &signals[g + 1..r_end];

    let mut weighted_sum: i128 = 0;
    let mut weight_total: i128 = 0;

    // 1. Polarity shift across the gap in [0, MICRO].
    if cfg.polarity_weight > 0 {
        let l_pols: Vec<i64> = left.iter().filter_map(|s| s.polarity_micro).collect();
        let r_pols: Vec<i64> = right.iter().filter_map(|s| s.polarity_micro).collect();
        if !l_pols.is_empty() && !r_pols.is_empty() {
            let l_mean = l_pols.iter().copied().sum::<i64>() / (l_pols.len() as i64);
            let r_mean = r_pols.iter().copied().sum::<i64>() / (r_pols.len() as i64);
            let raw_diff = (l_mean - r_mean).abs();
            let pol_shift = if (l_mean > 100_000 && r_mean < -100_000) || (l_mean < -100_000 && r_mean > 100_000) {
                raw_diff.min(MICRO)
            } else {
                (raw_diff / 2).min(MICRO)
            };
            weighted_sum += (pol_shift as i128) * (cfg.polarity_weight as i128);
            weight_total += cfg.polarity_weight as i128;
        }
    }

    // 2. Aspect category/target shift or aspect polarity flip on the same aspect category.
    if cfg.aspect_weight > 0 {
        let l_asp: Vec<&(String, Polarity)> = left.iter().flat_map(|s| s.aspects.iter()).collect();
        let r_asp: Vec<&(String, Polarity)> = right.iter().flat_map(|s| s.aspects.iter()).collect();
        if !l_asp.is_empty() && !r_asp.is_empty() {
            let mut shared_same_pol = 0usize;
            let mut shared_diff_pol = 0usize;
            for &(lt, lp) in &l_asp {
                for &(rt, rp) in &r_asp {
                    if !lt.is_empty() && lt == rt {
                        if lp == rp {
                            shared_same_pol += 1;
                        } else {
                            shared_diff_pol += 1;
                        }
                    }
                }
            }
            let asp_shift = if shared_same_pol == 0 {
                MICRO
            } else {
                let total_pairs = shared_same_pol + shared_diff_pol;
                ((shared_diff_pol as i64) * MICRO) / (total_pairs as i64)
            };
            weighted_sum += (asp_shift as i128) * (cfg.aspect_weight as i128);
            weight_total += cfg.aspect_weight as i128;
        }
    }

    // 3. Incivility shift (onset/offset of flagged incivility or large score jump).
    if cfg.incivility_weight > 0 {
        let l_inc: Vec<(i64, u8)> = left.iter().filter_map(|s| s.incivility_micro).collect();
        let r_inc: Vec<(i64, u8)> = right.iter().filter_map(|s| s.incivility_micro).collect();
        if !l_inc.is_empty() && !r_inc.is_empty() {
            let l_max = l_inc.iter().map(|&(s, _)| s).max().unwrap_or(0);
            let r_max = r_inc.iter().map(|&(s, _)| s).max().unwrap_or(0);
            let l_mask = l_inc.iter().fold(0u8, |acc, &(_, m)| acc | m);
            let r_mask = r_inc.iter().fold(0u8, |acc, &(_, m)| acc | m);
            if l_max > 0 || r_max > 0 || l_mask != 0 || r_mask != 0 || weight_total == 0 {
                let score_diff = (l_max - r_max).abs().min(MICRO);
                let inc_shift = if (l_mask == 0) != (r_mask == 0) {
                    score_diff.max(MICRO)
                } else if l_mask != r_mask {
                    score_diff.max(MICRO / 2)
                } else {
                    score_diff
                };
                weighted_sum += (inc_shift as i128) * (cfg.incivility_weight as i128);
                weight_total += cfg.incivility_weight as i128;
            }
        }
    }

    if weight_total == 0 {
        return 0;
    }

    let denom = weight_total.max(MILLI as i128);
    ((weighted_sum / denom) as i64).clamp(0, MICRO)
}

/// Scores every sentence boundary in `sents` for sentiment, aspect, and incivility shift.
///
/// Like [`drift`](crate::drift), returns a [`Boundaries`] layer aligned to `sents`
/// (`cost.len() == sents.spans.len()`), where entry `g` (`0 <= g < n - 1`) is the cost of
/// cutting after sentence `g`, and the final entry after the last sentence is `(0, Cut::Sentence)`.
pub fn sentiment_shift(
    sents: &Sentences,
    view: Option<&View>,
    toks: Option<&Tokens>,
    cats: Option<&Categories<Sentences>>,
    aspects: Option<&Aspects>,
    incivility: Option<&Incivility>,
    cfg: SentimentShift,
) -> Boundaries {
    let n = sents.spans.len();
    if n == 0 {
        return Boundaries { cost: Vec::new(), reason: Vec::new() };
    }

    let signals = extract_sentence_signals(sents, view, toks, cats, aspects, incivility);
    let mut cost = vec![0i32; n];
    let mut reason = vec![Cut::Sentence; n];

    let floor = i64::from(cfg.floor);
    let scale = i64::from(cfg.scale);
    let shift = i64::from(cfg.shift);

    for g in 0..n.saturating_sub(1) {
        let s = gap_shift_micro(&signals, g, cfg);
        if s >= floor {
            let c = -((s * scale) / MICRO);
            cost[g] = c.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        }
        if s >= shift {
            reason[g] = Cut::TopicShift;
        }
    }

    Boundaries { cost, reason }
}

/// Annotator that writes [`Boundaries`] under [`SENTIMENT`] from sentence sentiment,
/// aspect, and incivility layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct SentimentBoundaries {
    pub shift: SentimentShift,
}

impl SentimentBoundaries {
    pub const fn new(shift: SentimentShift) -> Self {
        Self { shift }
    }

    /// Binds the output [`Boundaries`] to [`SENTIMENT`].
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, SENTIMENT)
    }
}

impl Annotator for SentimentBoundaries {
    const ID: &'static str = "turbo:sentiment-boundaries";
    type Reads =
        (Sentences, Option<View>, Option<Tokens>, Option<Categories<Sentences>>, Option<Aspects>, Option<Incivility>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sents, view, toks, cats, aspects, incivility): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let view_id = sents.spans.view();
        if view.is_some_and(|v| v.id() != view_id) || toks.is_some_and(|t| t.spans.view() != view_id) {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        if let Some(c) = cats
            && c.len() != sents.spans.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} sentence categories for {} sentences", c.len(), sents.spans.len()),
            });
        }
        Ok((sentiment_shift(sents, view, toks, cats, aspects, incivility, self.shift),))
    }

    fn config_hash(&self) -> u64 {
        let s = self.shift;
        [
            u64::from(s.window.get()),
            u64::from(s.polarity_weight),
            u64::from(s.aspect_weight),
            u64::from(s.incivility_weight),
            u64::from(s.floor),
            u64::from(s.scale),
            u64::from(s.shift),
        ]
        .iter()
        .fold(Fnv1a::new(), |mut h, &x| {
            h.u64(x);
            h
        })
        .finish()
    }
}
