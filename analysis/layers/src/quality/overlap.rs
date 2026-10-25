use turbo_analysis::{Overlap, Span, Spans};

use super::{QualityFlags, SpanQuality};
use turbo_analysis::Layer;

impl SpanQuality {
    /// Rejects an item iff one of its actual nonempty fragments overlaps a
    /// region below the floor or carrying a forbidden flag. Gaps and uncovered
    /// text do not imply poor quality. Empty items are retained.
    ///
    /// O(F + S + C) work and memory for F query fragments, S quality rows and
    /// C query items. The shared interval join returns ranges; a prefix sum
    /// tests whole ranges in O(1), without expanding all overlapping pairs.
    pub fn rejected<L>(&self, spans: &Spans<L>, min_score: f32, forbidden: QualityFlags) -> Result<Vec<bool>, String> {
        self.check()?;
        spans.check(Overlap::Any)?;
        if self.spans.view() != spans.view() {
            return Err("quality and candidate spans must use the same view".into());
        }
        if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) || forbidden.0 & !QualityFlags::ALL.0 != 0 {
            return Err("invalid quality floor or forbidden flags".into());
        }
        let mut bad_prefix = Vec::with_capacity(self.metrics.len() + 1);
        bad_prefix.push(0usize);
        for m in &self.metrics {
            let bad = m.composite_score < min_score || m.flags.intersects(forbidden);
            bad_prefix.push(bad_prefix.last().unwrap() + usize::from(bad));
        }
        let mut fragments = Vec::new();
        for (i, _) in spans.iter() {
            for fragment in spans.fragments(i).filter(|s| !s.is_empty()) {
                fragments.push((fragment, i.get()));
            }
        }
        order(&mut fragments);
        let mut probes: Spans<()> = Spans::with_capacity(spans.view(), fragments.len());
        for &(fragment, _) in &fragments {
            probes.push(fragment);
        }
        let mut rejected = vec![false; spans.len()];
        for (probe, _, range) in probes.join_overlapping(&self.spans) {
            if bad_prefix[range.end.get()] != bad_prefix[range.start.get()] {
                rejected[fragments[probe.get()].1] = true;
            }
        }
        Ok(rejected)
    }
}

fn order(items: &mut Vec<(Span, usize)>) {
    if items.windows(2).all(|w| w[0].0.start <= w[1].0.start) {
        return;
    }
    let mut buffer = items.clone();
    for shift in [0, 8, 16, 24] {
        let mut next = [0usize; 256];
        for (span, _) in items.iter() {
            next[((span.start >> shift) & 255) as usize] += 1;
        }
        let mut sum = 0;
        for n in &mut next {
            let count = *n;
            *n = sum;
            sum += count;
        }
        for &item in items.iter() {
            let bucket = ((item.0.start >> shift) & 255) as usize;
            buffer[next[bucket]] = item;
            next[bucket] += 1;
        }
        std::mem::swap(items, &mut buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QualityMetrics;
    use turbo_analysis::{Ix, ViewId};

    #[test]
    fn shared_join_mask_matches_brute_force_for_unordered_nested_discontinuous_and_empty_items() {
        let mut q = SpanQuality::new(ViewId::SOURCE);
        for (a, b, score, flags) in [
            (2, 5, 0.8, QualityFlags::CLEAN),
            (5, 10, 0.2, QualityFlags::CLEAN),
            (12, 16, 0.9, QualityFlags::MOJIBAKE_SUSPECT),
        ] {
            q.push(Span::new(a, b), QualityMetrics { composite_score: score, flags, ..Default::default() });
        }
        let mut probes: Spans<()> = Spans::new(ViewId::SOURCE);
        for a in (0..=18).rev() {
            for b in a..=18 {
                probes.push(Span::new(a, b));
            }
        }
        probes.push_fragments(&[Span::new(0, 2), Span::new(10, 12), Span::new(16, 18)]);
        let got = q.rejected(&probes, 0.5, QualityFlags::MOJIBAKE_SUSPECT).unwrap();
        for (i, _) in probes.iter() {
            let expected = probes.fragments(i).filter(|s| !s.is_empty()).any(|f| {
                q.spans.iter().any(|(j, s)| {
                    f.start < s.end
                        && s.start < f.end
                        && (q.metrics[j.get()].composite_score < 0.5
                            || q.metrics[j.get()].flags.intersects(QualityFlags::MOJIBAKE_SUSPECT))
                })
            });
            assert_eq!(got[i.get()], expected, "item {}", i.get());
        }
        assert!(!*got.last().unwrap());
        assert!(q.rejected(&Spans::<()>::new(ViewId(999)), 0.5, QualityFlags::CLEAN).is_err());
        assert!(q.rejected(&probes, f32::NAN, QualityFlags::CLEAN).is_err());
        assert_eq!(probes.get(Ix::new(0)), Span::new(18, 18));
    }
}
