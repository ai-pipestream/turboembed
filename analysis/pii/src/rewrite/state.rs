//! The per-document numbering state and per-cluster writer every
//! [`RewriteStrategy`] shares ([`RewriteState`]).

use std::collections::HashMap;

use turbo_analysis_layers::{CustomKinds, PiiKind, PiiMention};

use super::mask::CharRetention;
use super::replace::MentionCluster;
use super::{Abbreviator, RewriteError, RewriteStrategy, absolute, kind_name};

#[derive(Copy, Clone)]
pub(super) enum MemberInitial {
    Abbreviate,
    Retention(CharRetention),
}

impl MemberInitial {
    #[inline]
    fn keeps(self, piece: &str, rel_at: usize, c: char) -> bool {
        match self {
            MemberInitial::Abbreviate => Abbreviator::is_initial_at(piece, rel_at),
            MemberInitial::Retention(ret) => ret.keeps(rel_at, c),
        }
    }
}

pub(crate) struct RewriteState<'a> {
    kinds: &'a CustomKinds,
    assigned: HashMap<(PiiKind, &'a str), u32>,
    counters: HashMap<PiiKind, u32>,
}

impl<'a> RewriteState<'a> {
    /// Numbering state for mentions of a layer whose custom kinds index
    /// `kinds`.
    pub(crate) fn new(kinds: &'a CustomKinds) -> RewriteState<'a> {
        RewriteState { kinds, assigned: HashMap::new(), counters: HashMap::new() }
    }

    pub(crate) fn number(&mut self, kind: PiiKind, normalized: &'a str) -> u32 {
        *self.assigned.entry((kind, normalized)).or_insert_with(|| {
            let n = self.counters.entry(kind).or_insert(0);
            *n += 1;
            *n
        })
    }

    pub(crate) fn write_strategy(
        &mut self,
        strategy: &RewriteStrategy,
        text: &str,
        m: &PiiMention<'a>,
        out: &mut String,
    ) -> Result<(), RewriteError> {
        match strategy {
            RewriteStrategy::Mask(policy) => {
                out.push_str(&policy.masked(&text[m.span.range()]));
                Ok(())
            }
            RewriteStrategy::Pseudonym(pseudo) => {
                let n = self.number(m.kind, m.normalized);
                pseudo.write_label(kind_name(self.kinds, m.kind), n, out);
                Ok(())
            }
            RewriteStrategy::Hmac(hmac) => hmac
                .write_token(kind_name(self.kinds, m.kind), m.normalized, out)
                .map_err(|_| RewriteError::EmptyValue(m.span)),
            RewriteStrategy::Format(fmt) => {
                let n = if fmt.uses_number() { self.number(m.kind, m.normalized) } else { 1 };
                fmt.write_label(kind_name(self.kinds, m.kind), n, out);
                Ok(())
            }
            RewriteStrategy::Static(stat) => {
                out.push_str(stat.replacement_in(self.kinds, m.kind));
                Ok(())
            }
            RewriteStrategy::Sha256(sha) => sha
                .write_digest(kind_name(self.kinds, m.kind), m.normalized, out)
                .map_err(|_| RewriteError::EmptyValue(m.span)),
            RewriteStrategy::Map(map) => map.write_mention(self, text, m, out),
            RewriteStrategy::Truncate(trunc) => {
                trunc.write_truncated(&text[m.span.range()], out);
                Ok(())
            }
            RewriteStrategy::Abbreviate(abbr) => {
                abbr.write_abbreviated(&text[m.span.range()], out);
                Ok(())
            }
            RewriteStrategy::ZipSafeHarbor(zip) => {
                zip.write_mention(m.kind, &text[m.span.range()], out);
                Ok(())
            }
            RewriteStrategy::AgeSafeHarbor(age) => {
                age.write_mention(m.kind, &text[m.span.range()], out);
                Ok(())
            }
            RewriteStrategy::Date(date) => {
                date.write_mention(m.kind, &text[m.span.range()], out);
                Ok(())
            }
            // Both encrypt the surface, so decryption gives back the text.
            #[cfg(feature = "reversible")]
            RewriteStrategy::AesGcm(aes) => aes
                .write_token(kind_name(self.kinds, m.kind), &text[m.span.range()], out)
                .map_err(|e| RewriteError::Encrypt(m.span, e.to_string())),
            #[cfg(feature = "reversible")]
            RewriteStrategy::Ff1(ff1) => ff1
                .write_encrypted(&text[m.span.range()], out)
                .map_err(|e| RewriteError::Encrypt(m.span, e.to_string())),
        }
    }

    pub(crate) fn write_cluster<'b>(
        &mut self,
        strategy_for: impl Fn(&PiiMention<'a>) -> &'b RewriteStrategy,
        text: &str,
        cluster: &MentionCluster<'a>,
        out: &mut String,
    ) -> Result<(), RewriteError> {
        for mem in &cluster.members {
            if mem.normalized.is_empty() && strategy_for(mem).requires_normalized(text, mem) {
                return Err(RewriteError::EmptyValue(mem.span));
            }
        }
        let winner = &cluster.merged;
        if cluster.members.len() == 1 {
            return self.write_strategy(strategy_for(winner), text, winner, out);
        }
        let mut strategy = strategy_for(winner);
        while let RewriteStrategy::Map(map) = strategy {
            if let Some(rep) = map.lookup_mention(text, winner) {
                out.push_str(rep);
                return Ok(());
            }
            strategy = map.fallback();
        }
        let piece = &text[winner.span.range()];
        match strategy {
            RewriteStrategy::Truncate(trunc) => {
                let winner_ret = trunc.retention(piece);
                let member_rets: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, strategy_for(mem).retention(text, mem))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = absolute(winner.span, rel_at);
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    out.push(if keep { c } else { trunc.mask() });
                }
                Ok(())
            }
            RewriteStrategy::Mask(policy) => {
                let winner_ret = policy.retention(piece);
                let member_rets: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, strategy_for(mem).retention(text, mem))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = absolute(winner.span, rel_at);
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    if keep {
                        out.push(c);
                    } else {
                        for _ in 0..c.len_utf8() {
                            out.push(policy.mask());
                        }
                    }
                }
                Ok(())
            }
            RewriteStrategy::Abbreviate(abbr) => {
                let member_inits: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, strategy_for(mem).member_initial(text, mem))).collect();
                abbr.write_abbreviated_filtered(
                    piece,
                    |rel_at, c| {
                        let abs_at = absolute(winner.span, rel_at);
                        member_inits
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, init)| init.keeps(&text[span.range()], (abs_at - span.start) as usize, c))
                    },
                    out,
                );
                Ok(())
            }
            // Overlapping mentions are not one ZIP code, and another
            // member's strategy may keep none of its characters.
            RewriteStrategy::ZipSafeHarbor(zip) => {
                zip.write_masked(piece, out);
                Ok(())
            }
            // Overlapping mentions are not one age either.
            RewriteStrategy::AgeSafeHarbor(age) => {
                age.write_masked(piece, out);
                Ok(())
            }
            // Nor are they one date.
            RewriteStrategy::Date(date) => {
                date.write_masked(piece, out);
                Ok(())
            }
            other => self.write_strategy(other, text, winner, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use turbo_analysis::{Document, Span, View, ViewId};
    use turbo_analysis_layers::{CustomKindId, CustomKinds, Pii, PiiEvidence, PiiKind, PiiMention};

    use crate::rewrite::mask::RETENTION_WORK;
    use crate::rewrite::{MaskPolicy, RewritePolicy, Truncate};

    /// Precomputing `CharRetention` once per cluster member keeps overlapping
    /// cluster rewrites linear in cluster byte length. Counts retention work,
    /// never wall-clock time (#1936).
    #[test]
    fn overlapping_cluster_rewrites_in_linear_time() {
        const PERSON: PiiKind = PiiKind::Custom(CustomKindId::new(0));
        let kinds = CustomKinds::from_names(["person"]).unwrap();
        let policy = RewritePolicy::new(MaskPolicy::new('*').unwrap())
            .with_custom_kinds(kinds.clone())
            .with(PiiKind::Email, Truncate::keep_end(4));
        let work_for = |n: usize| {
            let doc = Document::new("a".repeat(n));
            let view = doc.get::<View>().unwrap();
            let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, kinds.clone());
            let det = pii.detector("test", "1").unwrap();
            for (end, kind, norm) in [(n as u32, PiiKind::Email, "e"), ((n / 2) as u32, PERSON, "p")] {
                pii.push(PiiMention {
                    span: Span::new(0, end),
                    kind,
                    detector: det,
                    evidence: &[PiiEvidence::Shape],
                    score: None,
                    normalized: norm,
                });
            }
            RETENTION_WORK.with(|c| c.set(0));
            let out = policy.rewrite(view, &pii).unwrap();
            assert!(out.text().ends_with("aaaa") && out.text()[..n - 4].chars().all(|c| c == '*'));
            RETENTION_WORK.with(Cell::get)
        };
        let (w1, w4) = (work_for(2_000), work_for(8_000));
        assert!(w1 > 0 && w4 <= w1 * 4, "cluster retention work must grow linearly: N={w1}, 4N={w4}");
    }
}
