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

use std::num::NonZeroU32;
use std::ops::Range;

use turbo_analysis::{Anchor, Annotator, Bound, Error, Ix, Key, KeyId, Layer, LayerName, Refs, View};
use turbo_analysis_layers::{
    Aspects, Boundaries, Categories, Chunks, Cut, Incivility, IncivilityLabel, IncivilityScores, IncivilityTarget,
    Polarity, Sentences, Strs, Tokens,
};

use crate::weigh::SENTIMENT;

const MICRO: i64 = 1_000_000;
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

fn label_bit(label: IncivilityLabel) -> u8 {
    1u8 << (label as u8)
}

fn max_incivility_score(scores: IncivilityScores) -> f64 {
    scores.iter().into_iter().map(|(_, s)| if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 }).fold(0.0f64, f64::max)
}

fn category_row_polarity_micro(cats: &Categories<Sentences>, i: usize) -> Option<i64> {
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

fn resolve_aspect_sentence(
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
    let (_, ends) = sents.spans.columns();
    let idx = ends.partition_point(|&e| e <= tok_start);
    (idx < n).then_some(idx)
}

fn resolve_aspect_label(a_ix: Ix<Aspects>, aspects: &Aspects, toks: Option<&Tokens>, view: Option<&View>) -> String {
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
        let text = v.text();
        if start <= end && end <= text.len() {
            return text[start..end].trim().to_owned();
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
        .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}

/// Per-chunk sentiment, aspect, and incivility facets aligned 1-to-1 with [`Chunks`].
///
/// Follows the sparse-row layout of [`TermStats`](crate::terms::TermStats): each chunk `c`
/// owns `aspect_row[c]..aspect_row[c + 1]` in `aspect`, `aspect_label`, and `aspect_polarity`,
/// and `incivility_row[c]..incivility_row[c + 1]` in `incivility_label`.
#[derive(Clone, Debug)]
pub struct ChunkFacets {
    pub polarity: Vec<Polarity>,
    pub polarity_score: Vec<f64>,
    pub aspect_row: Vec<u32>,
    pub aspect: Vec<Ix<Aspects>>,
    pub aspect_label: Strs,
    pub aspect_polarity: Vec<Polarity>,
    pub incivility_row: Vec<u32>,
    pub incivility_label: Vec<IncivilityLabel>,
    pub incivility_max: Vec<IncivilityScores>,
}

impl ChunkFacets {
    fn aspect_range(&self, c: Ix<Chunks>) -> Range<usize> {
        let i = c.get();
        self.aspect_row[i] as usize..self.aspect_row[i + 1] as usize
    }

    /// Aggregated sentiment polarity for chunk `c`.
    #[inline]
    pub fn chunk_polarity(&self, c: Ix<Chunks>) -> Polarity {
        self.polarity[c.get()]
    }

    /// Signed polarity score in `[-1.0, 1.0]` for chunk `c`.
    #[inline]
    pub fn chunk_polarity_score(&self, c: Ix<Chunks>) -> f64 {
        self.polarity_score[c.get()]
    }

    /// Aspect indices from [`Aspects`] falling within chunk `c`.
    #[inline]
    pub fn aspect_ixs(&self, c: Ix<Chunks>) -> &[Ix<Aspects>] {
        &self.aspect[self.aspect_range(c)]
    }

    /// Aspect polarities parallel to [`Self::aspect_ixs`] for chunk `c`.
    #[inline]
    pub fn aspect_polarities(&self, c: Ix<Chunks>) -> &[Polarity] {
        &self.aspect_polarity[self.aspect_range(c)]
    }

    /// Iterates `(Ix<Aspects>, label, Polarity)` for all aspects falling within chunk `c`.
    pub fn aspects_of(&self, c: Ix<Chunks>) -> impl ExactSizeIterator<Item = (Ix<Aspects>, &str, Polarity)> + '_ {
        let r = self.aspect_range(c);
        r.map(move |k| (self.aspect[k], self.aspect_label.get(k), self.aspect_polarity[k]))
    }

    /// Flagged incivility labels present in chunk `c`, in canonical [`IncivilityLabel`] order.
    #[inline]
    pub fn incivility_labels(&self, c: Ix<Chunks>) -> &[IncivilityLabel] {
        let i = c.get();
        let r = self.incivility_row[i] as usize..self.incivility_row[i + 1] as usize;
        &self.incivility_label[r]
    }

    /// Element-wise maximum incivility dimension scores across sentences in chunk `c`.
    #[inline]
    pub fn incivility_scores(&self, c: Ix<Chunks>) -> IncivilityScores {
        self.incivility_max[c.get()]
    }
}

impl Layer for ChunkFacets {
    const ID: &'static str = "turbo:chunk-facets";
    const OF: Option<&'static LayerName> = Some(Chunks::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }

    fn len(&self) -> usize {
        self.polarity.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_asp = self.aspect.iter().map(|ix| ix.get() + 1).max().unwrap_or(0);
        if max_asp > 0 { vec![(Key::<Aspects>::DEFAULT.id(), max_asp)] } else { Vec::new() }
    }

    fn check(&self) -> Result<(), String> {
        let n = self.polarity.len();
        if self.polarity_score.len() != n || self.incivility_max.len() != n {
            return Err("a polarity score and incivility scores for every chunk".into());
        }
        if self.aspect_row.len() != n + 1
            || self.aspect_row.first().copied() != Some(0)
            || self.aspect_row.last().copied().unwrap_or(0) as usize != self.aspect.len()
            || !self.aspect_row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("aspect_row must start at 0, be non-decreasing, and end at aspect.len()".into());
        }
        if self.aspect_label.len() != self.aspect.len() || self.aspect_polarity.len() != self.aspect.len() {
            return Err("an aspect label and polarity for every aspect entry".into());
        }
        if self.incivility_row.len() != n + 1
            || self.incivility_row.first().copied() != Some(0)
            || self.incivility_row.last().copied().unwrap_or(0) as usize != self.incivility_label.len()
            || !self.incivility_row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("incivility_row must start at 0, be non-decreasing, and end at incivility_label.len()".into());
        }
        for i in 0..n {
            let r = self.incivility_row[i] as usize..self.incivility_row[i + 1] as usize;
            if !self.incivility_label[r].windows(2).all(|w| w[0] < w[1]) {
                return Err(format!("chunk {i}: incivility labels must be in strictly ascending order"));
            }
        }
        if self.polarity_score.iter().any(|x| !x.is_finite() || !(-1.0..=1.0).contains(x)) {
            return Err("polarity_score must be finite and in -1.0..=1.0".into());
        }
        if self.incivility_max.iter().any(|s| !s.is_valid()) {
            return Err("incivility_max scores must be finite and in 0.0..=1.0".into());
        }
        Ok(())
    }
}

/// Annotator that projects sentence-level sentiment categories, [`Aspects`], and
/// [`Incivility`] onto [`Chunks`] as [`ChunkFacets`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkFacetBuilder;

impl Annotator for ChunkFacetBuilder {
    const ID: &'static str = "turbo:chunk-facet-builder";
    type Reads = (
        Chunks,
        Sentences,
        Option<View>,
        Option<Tokens>,
        Option<Categories<Sentences>>,
        Option<Aspects>,
        Option<Incivility>,
    );
    type Writes = (ChunkFacets,);

    fn annotate(
        &self,
        (chunks, sents, view, toks, cats, aspects, incivility): Refs<'_, Self>,
    ) -> Result<(ChunkFacets,), Error> {
        let n_chunks = chunks.len();
        let n_sents = sents.spans.len();
        let mut polarity = Vec::with_capacity(n_chunks);
        let mut polarity_score = Vec::with_capacity(n_chunks);
        let mut aspect_row = Vec::with_capacity(n_chunks + 1);
        let mut aspect_ixs = Vec::new();
        let mut aspect_label = Strs::default();
        let mut aspect_polarity = Vec::new();
        let mut incivility_row = Vec::with_capacity(n_chunks + 1);
        let mut incivility_label = Vec::new();
        let mut incivility_max = Vec::with_capacity(n_chunks);

        aspect_row.push(0);
        incivility_row.push(0);

        for c_idx in 0..n_chunks {
            let c_ix = Ix::<Chunks>::new(c_idx as u32);
            let sr = chunks.sentences[c_idx];
            let s_start = sr.start.get();
            let s_end = sr.end.get().min(n_sents);

            // 1. Collect aspects whose sentence falls in [s_start, s_end).
            let chunk_aspect_start = aspect_ixs.len();
            let mut aspect_compound_sum = 0.0f64;
            let mut aspect_compound_n = 0usize;
            if let Some(asp) = aspects {
                for (a_ix, item) in asp.iter() {
                    if let Some(s_idx) = resolve_aspect_sentence(a_ix, asp, sents, toks)
                        && (s_start..s_end).contains(&s_idx)
                    {
                        let lbl = resolve_aspect_label(a_ix, asp, toks, view);
                        aspect_ixs.push(a_ix);
                        aspect_label.push(&lbl);
                        aspect_polarity.push(item.polarity.label());
                        aspect_compound_sum += item.polarity.compound().clamp(-1.0, 1.0);
                        aspect_compound_n += 1;
                    }
                }
            }
            aspect_row.push(aspect_ixs.len() as u32);

            // 2. Compute chunk polarity and signed score.
            let mut signed_sum = 0.0f64;
            let mut signed_n = 0usize;

            if let Some(cats) = cats {
                for s_idx in s_start..s_end.min(cats.len()) {
                    if let Some(p_micro) = category_row_polarity_micro(cats, s_idx) {
                        signed_sum += (p_micro as f64) / (MICRO as f64);
                        signed_n += 1;
                    }
                }
            }

            if signed_n == 0 && aspect_compound_n > 0 {
                signed_sum = aspect_compound_sum;
                signed_n = aspect_compound_n;
            } else if signed_n == 0 && aspect_ixs.len() > chunk_aspect_start {
                for &pol in &aspect_polarity[chunk_aspect_start..] {
                    match pol {
                        Polarity::Positive => signed_sum += 1.0,
                        Polarity::Negative => signed_sum -= 1.0,
                        Polarity::Neutral => {}
                    }
                    signed_n += 1;
                }
            }

            let mean_score = if signed_n > 0 { (signed_sum / (signed_n as f64)).clamp(-1.0, 1.0) } else { 0.0 };
            let chunk_pol = if mean_score > 0.05 {
                Polarity::Positive
            } else if mean_score < -0.05 {
                Polarity::Negative
            } else {
                Polarity::Neutral
            };
            polarity.push(chunk_pol);
            polarity_score.push(mean_score);

            // 3. Aggregate incivility scores and distinct flagged labels over sentences in chunk `c`
            //    or direct `IncivilityTarget::Chunk(c_ix)` entries.
            let mut max_scores = IncivilityScores::default();
            let mut mask = 0u8;
            if let Some(inc) = incivility {
                for i in 0..inc.len() {
                    let item_ix = Ix::<Incivility>::new(i as u32);
                    let in_chunk = match inc.target[i] {
                        IncivilityTarget::Sentence(s_ix) => (s_start..s_end).contains(&s_ix.get()),
                        IncivilityTarget::Chunk(ch_ix) => ch_ix == c_ix,
                        IncivilityTarget::Block(_) => false,
                    };
                    if in_chunk {
                        let sc = inc.scores(item_ix);
                        max_scores.insult = max_scores.insult.max(sc.insult);
                        max_scores.profanity = max_scores.profanity.max(sc.profanity);
                        max_scores.threat = max_scores.threat.max(sc.threat);
                        max_scores.identity_attack = max_scores.identity_attack.max(sc.identity_attack);
                        max_scores.intolerance = max_scores.intolerance.max(sc.intolerance);
                        max_scores.severe_toxicity = max_scores.severe_toxicity.max(sc.severe_toxicity);
                        for (lbl, _) in inc.evidence_of(item_ix) {
                            mask |= label_bit(lbl);
                        }
                        for (lbl, val) in sc.iter() {
                            if val >= 0.5 {
                                mask |= label_bit(lbl);
                            }
                        }
                    }
                }
            }
            for lbl in IncivilityLabel::ALL {
                if (mask & label_bit(lbl)) != 0 {
                    incivility_label.push(lbl);
                }
            }
            incivility_row.push(incivility_label.len() as u32);
            incivility_max.push(max_scores);
        }

        let facets = ChunkFacets {
            polarity,
            polarity_score,
            aspect_row,
            aspect: aspect_ixs,
            aspect_label,
            aspect_polarity,
            incivility_row,
            incivility_label,
            incivility_max,
        };
        Ok((facets,))
    }
}

/// Fournier (2013) Boundary Similarity ($B$) over internal sentence-gap boundary positions.
///
/// Given reference boundary gaps `gold_gaps` and predicted boundary gaps `pred_gaps`
/// (each a gap index `g` after sentence `g`, as returned by [`chunk_boundary_gaps`]),
/// pairs exact boundary matches with weight `1.0` and near-miss transpositions within
/// `near_distance` gaps (`>= 1`) with weight `0.5`, divided by the total number of matched
/// and unmatched boundary edits:
///
/// $$B = \frac{|M_{\text{exact}}| + 0.5 \cdot |M_{\text{near}}|}{|M_{\text{exact}}| + |M_{\text{near}}| + |U_{\text{gold}}| + |U_{\text{pred}}|}$$
///
/// Returns `1.0` when both `gold_gaps` and `pred_gaps` are empty.
pub fn boundary_similarity(gold_gaps: &[u32], pred_gaps: &[u32], near_distance: u32) -> f64 {
    let mut g: Vec<u32> = gold_gaps.to_vec();
    let mut p: Vec<u32> = pred_gaps.to_vec();
    g.sort_unstable();
    g.dedup();
    p.sort_unstable();
    p.dedup();

    if g.is_empty() && p.is_empty() {
        return 1.0;
    }

    let mut g_used = vec![false; g.len()];
    let mut p_used = vec![false; p.len()];
    let mut exact = 0usize;

    for (gi, &gv) in g.iter().enumerate() {
        if let Ok(pi) = p.binary_search(&gv) {
            g_used[gi] = true;
            p_used[pi] = true;
            exact += 1;
        }
    }

    let mut near = 0usize;
    if near_distance > 0 {
        for d in 1..=near_distance {
            for (gi, &gv) in g.iter().enumerate() {
                if g_used[gi] {
                    continue;
                }
                for (pi, &pv) in p.iter().enumerate() {
                    if p_used[pi] {
                        continue;
                    }
                    if gv.abs_diff(pv) == d {
                        g_used[gi] = true;
                        p_used[pi] = true;
                        near += 1;
                        break;
                    }
                }
            }
        }
    }

    let u_gold = g_used.iter().filter(|&&u| !u).count();
    let u_pred = p_used.iter().filter(|&&u| !u).count();
    let denom = exact + near + u_gold + u_pred;
    if denom == 0 { 1.0 } else { ((exact as f64) + 0.5 * (near as f64)) / (denom as f64) }
}

/// Extracts internal sentence-gap indices (`0..sents - 1`) where level-0 `chunks` place a boundary.
pub fn chunk_boundary_gaps(chunks: &Chunks) -> Vec<u32> {
    let level0: Vec<usize> = (0..chunks.len()).filter(|&i| chunks.level[i] == 0).collect();
    if level0.len() <= 1 {
        return Vec::new();
    }
    let mut gaps = Vec::with_capacity(level0.len() - 1);
    for &i in &level0[..level0.len() - 1] {
        let end = chunks.sentences[i].end.get() as u32;
        if end > 0 {
            gaps.push(end - 1);
        }
    }
    gaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Budget, Chunker, Costs, STRUCTURE, StructureBoundaries, WeighAllBoundaries, Weight, weigh_with_sentiment,
    };
    use turbo_analysis::{Analyzer, Document, IxRange, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
    use turbo_analysis_layers::{Aspect, AspectTarget, IncivilityItem, PolarityScores, Subwords};

    fn build_doc_with_sentences_and_subwords(sents_text: &[&str], subwords_per_sent: u32) -> Document {
        let full = sents_text.join(" ");
        let mut doc = Document::new(full);
        let mut s_spans = Spans::new(ViewId::SOURCE);
        let mut sw_spans = Spans::new(ViewId::SOURCE);
        let mut sw_ids = Vec::new();
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        let mut tok_sent = Vec::new();

        let mut cursor = 0u32;
        for (s_idx, s_str) in sents_text.iter().enumerate() {
            let len = s_str.len() as u32;
            s_spans.push(Span::new(cursor, cursor + len));
            tok_spans.push(Span::new(cursor, cursor + len));
            tok_sent.push(Ix::new(s_idx as u32));
            for k in 0..subwords_per_sent {
                let a = cursor + (k * len) / subwords_per_sent;
                let b = cursor + ((k + 1) * len) / subwords_per_sent;
                sw_spans.push(Span::new(a, b.max(a + 1).min(cursor + len)));
                sw_ids.push(1);
            }
            cursor += len + 1;
        }

        doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("t")).unwrap();
        doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sent) }, Provenance::by("t")).unwrap();
        doc.add(
            &Key::DEFAULT,
            Subwords {
                spans: sw_spans,
                ids: sw_ids,
                model: ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
                token: None,
            },
            Provenance::by("t"),
        )
        .unwrap();
        doc
    }

    #[test]
    fn default_weigh_all_boundaries_keeps_sentiment_off() {
        let w = WeighAllBoundaries::default();
        assert_eq!(w.structure, Weight::ONE);
        assert_eq!(w.drift, Weight::ONE);
        assert_eq!(w.sentiment, Weight(0));

        let s = Boundaries { cost: vec![1000, 1000, 0], reason: vec![Cut::Sentence; 3] };
        let d = Boundaries { cost: vec![0, 0, 0], reason: vec![Cut::Sentence; 3] };
        let sent =
            Boundaries { cost: vec![-1000, -1000, 0], reason: vec![Cut::TopicShift, Cut::TopicShift, Cut::Sentence] };

        let off = weigh_with_sentiment(&s, &d, &sent, (w.structure, w.drift, w.sentiment)).unwrap();
        assert_eq!(off.cost, [1000, 1000, 0]);
        assert_eq!(off.reason, [Cut::Sentence, Cut::Sentence, Cut::Sentence]);

        let on = weigh_with_sentiment(&s, &d, &sent, (Weight::ONE, Weight::ONE, Weight::ONE)).unwrap();
        assert_eq!(on.cost, [0, 0, 0]);
        assert_eq!(on.reason, [Cut::TopicShift, Cut::TopicShift, Cut::Sentence]);
    }

    #[test]
    fn sentiment_and_aspect_shift_places_boundary_and_populates_chunk_facets() {
        let sents_text = [
            "The handmade pasta was fresh and delicious.",
            "The risotto had wonderful texture and flavor.",
            "Our dessert was warm and plated with care.",
            "Then the waiter was rude and ignored our table.",
            "The manager yelled insults when we asked for water.",
            "Billing service was a complete disaster.",
        ];
        let mut doc = build_doc_with_sentences_and_subwords(&sents_text, 100);

        let cat_set = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
        let mut cats = Categories::<Sentences>::with_capacity(cat_set, 6);
        for &(best, scores) in &[
            (Tag(0), [0.92, 0.04, 0.04]),
            (Tag(0), [0.90, 0.05, 0.05]),
            (Tag(0), [0.94, 0.03, 0.03]),
            (Tag(1), [0.05, 0.90, 0.05]),
            (Tag(1), [0.03, 0.94, 0.03]),
            (Tag(1), [0.04, 0.92, 0.04]),
        ] {
            cats.push(best, &scores);
        }

        let asp_cats = TagSet::new("aspect", &["food", "service"]);
        let mut aspects = Aspects::new(asp_cats);
        for (s_idx, cat_tag, pol) in [
            (0u32, Tag(0), PolarityScores::new(0.90, 0.05, 0.05)),
            (1u32, Tag(0), PolarityScores::new(0.88, 0.06, 0.06)),
            (2u32, Tag(0), PolarityScores::new(0.92, 0.04, 0.04)),
            (3u32, Tag(1), PolarityScores::new(0.05, 0.90, 0.05)),
            (4u32, Tag(1), PolarityScores::new(0.04, 0.92, 0.04)),
            (5u32, Tag(1), PolarityScores::new(0.03, 0.94, 0.03)),
        ] {
            aspects.push(Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(s_idx, s_idx + 1), sentence: Some(Ix::new(s_idx)) },
                category: Some(cat_tag),
                polarity: pol,
                opinions: &[],
            });
        }

        let mut incivility = Incivility::new();
        incivility.push(IncivilityItem {
            target: IncivilityTarget::Sentence(Ix::new(4)),
            scores: IncivilityScores { insult: 0.85, profanity: 0.10, ..IncivilityScores::default() },
            evidence: &[(IncivilityLabel::Insult, IxRange::new(4, 5))],
        });

        doc.add(&Key::DEFAULT, cats, Provenance::by("t")).unwrap();
        doc.add(&Key::DEFAULT, aspects, Provenance::by("t")).unwrap();
        doc.add(&Key::DEFAULT, incivility, Provenance::by("t")).unwrap();

        let budget = Budget { max: 400, target: 300, size_weight: 0, overhead: 0 };

        // 1. Default pipeline (sentiment Weight(0), off by default):
        let mut doc_off = doc.clone();
        let analyzer_off = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<Categories<Sentences>>::DEFAULT)
            .given(Key::<Aspects>::DEFAULT)
            .given(Key::<Incivility>::DEFAULT)
            .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
            .step(SentimentBoundaries::default().bound())
            .step(WeighAllBoundaries::default().bound())
            .add(Chunker { budget })
            .add(ChunkFacetBuilder)
            .build()
            .unwrap();
        analyzer_off.run(std::slice::from_mut(&mut doc_off))[0].as_ref().unwrap();

        let chunks_off = doc_off.get::<Chunks>().unwrap();
        assert_eq!(chunk_boundary_gaps(chunks_off), vec![1]);

        // 2. Pipeline with sentiment boundary signal enabled (`sentiment: Weight::ONE`):
        let mut doc_on = doc;
        let analyzer_on = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<Categories<Sentences>>::DEFAULT)
            .given(Key::<Aspects>::DEFAULT)
            .given(Key::<Incivility>::DEFAULT)
            .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
            .step(SentimentBoundaries::default().bound())
            .step(WeighAllBoundaries { structure: Weight::ONE, drift: Weight::ONE, sentiment: Weight::ONE }.bound())
            .add(Chunker { budget })
            .add(ChunkFacetBuilder)
            .build()
            .unwrap();
        analyzer_on.run(std::slice::from_mut(&mut doc_on))[0].as_ref().unwrap();

        let sent_bounds = doc_on.get_key(&Key::<Boundaries>::named(SENTIMENT)).unwrap();
        assert_eq!(sent_bounds.cost[0], 0);
        assert_eq!(sent_bounds.cost[1], 0);
        assert!(sent_bounds.cost[2] < -800, "expected strong negative cost at gap 2, got {}", sent_bounds.cost[2]);
        assert_eq!(sent_bounds.reason[2], Cut::TopicShift);

        let chunks_on = doc_on.get::<Chunks>().unwrap();
        assert_eq!(chunks_on.len(), 2);
        assert_eq!(chunks_on.sentences[0], IxRange::new(0, 3));
        assert_eq!(chunks_on.sentences[1], IxRange::new(3, 6));
        assert_eq!(chunks_on.cut[1], Cut::TopicShift);

        let gold_gaps = [2u32];
        let b_sim_off = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_off), 1);
        let b_sim_on = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_on), 1);
        assert!(
            (b_sim_off - 0.5).abs() < 1e-12,
            "without signal, cut at 1 vs gold 2 is a near miss (0.5): {b_sim_off}"
        );
        assert!((b_sim_on - 1.0).abs() < 1e-12, "with signal, cut at 2 vs gold 2 is exact (1.0): {b_sim_on}");

        let facets = doc_on.get::<ChunkFacets>().unwrap();
        assert_eq!(facets.len(), 2);
        assert_eq!(facets.chunk_polarity(Ix::new(0)), Polarity::Positive);
        assert!(facets.chunk_polarity_score(Ix::new(0)) > 0.80);
        let c0_aspects: Vec<_> = facets.aspects_of(Ix::new(0)).map(|(_, l, p)| (l, p)).collect();
        assert_eq!(
            c0_aspects,
            vec![("food", Polarity::Positive), ("food", Polarity::Positive), ("food", Polarity::Positive),]
        );
        assert!(facets.incivility_labels(Ix::new(0)).is_empty());

        assert_eq!(facets.chunk_polarity(Ix::new(1)), Polarity::Negative);
        assert!(facets.chunk_polarity_score(Ix::new(1)) < -0.80);
        let c1_aspects: Vec<_> = facets.aspects_of(Ix::new(1)).map(|(_, l, p)| (l, p)).collect();
        assert_eq!(
            c1_aspects,
            vec![("service", Polarity::Negative), ("service", Polarity::Negative), ("service", Polarity::Negative),]
        );
        assert_eq!(facets.incivility_labels(Ix::new(1)), &[IncivilityLabel::Insult]);
        assert!((facets.incivility_scores(Ix::new(1)).insult - 0.85).abs() < 1e-9);
    }

    #[test]
    fn boundary_similarity_scores_exact_near_and_missed_cuts() {
        assert!((boundary_similarity(&[], &[], 1) - 1.0).abs() < 1e-12);
        assert!((boundary_similarity(&[2, 5], &[2, 5], 1) - 1.0).abs() < 1e-12);
        assert!((boundary_similarity(&[2, 5], &[2, 6], 1) - 0.75).abs() < 1e-12);
        assert!((boundary_similarity(&[2, 5], &[2, 8], 1) - (1.0 / 3.0)).abs() < 1e-12);
    }
}
