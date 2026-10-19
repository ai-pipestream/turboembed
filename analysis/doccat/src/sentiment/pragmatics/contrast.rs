use super::*;

impl PragmaticsAnnotator {
    pub(super) fn detect_riloff_incongruity(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Option<(f64, Incongruity)> {
        let positives = self.collect_positive_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let negatives = self.collect_negative_candidates(tok_range, lower_tokens, opinions, emoji_spans);

        if positives.is_empty() || negatives.is_empty() {
            return None;
        }

        let mut best: Option<(f64, Incongruity)> = None;
        for pos in &positives {
            for neg in &negatives {
                if ranges_overlap(pos.tokens, neg.tokens) {
                    continue;
                }
                let (first, second) = if pos.tokens.start.get() < neg.tokens.start.get() {
                    (pos.tokens, neg.tokens)
                } else {
                    (neg.tokens, pos.tokens)
                };
                let between = &lower_tokens[first.end.get()..second.start.get()];

                // Balanced clauses separated by an adversative conjunction ("but",
                // "however", "although", ...) are literal contrast unless the
                // positive span carries an explicit sarcastic framing cue.
                let has_adversative = between.iter().any(|w| ADVERSATIVE_CONJUNCTIONS.contains(&w.as_str()));
                if has_adversative && !pos.cued {
                    continue;
                }

                // When the negative span comes before the positive span and the
                // positive span is not a sarcastic cue or a clause-final ironic tag,
                // avoid firing on plain narrative sequence ("had a headache, and
                // now feel good").
                let pos_after_neg = pos.tokens.start.get() >= neg.tokens.end.get();
                if pos_after_neg && !pos.cued && !is_clause_final_tag(pos.tokens, tok_range, lower_tokens) {
                    continue;
                }

                // Require at least one Riloff situation or sarcastic framing cue, or
                // a strong positive sentiment expression (>= 1.8) paired with a
                // strong negative opinion (>= 1.5), so incidental mild lexicon words
                // ("original ... blame") do not false-trigger without the head.
                if !pos.cued && !neg.cued && (pos.strength < 1.8 || neg.strength < 1.5) {
                    continue;
                }

                let cue_bonus = match (pos.cued, neg.cued) {
                    (true, true) => 0.25,
                    (true, false) | (false, true) => 0.15,
                    (false, false) => 0.0,
                };
                let raw = (pos.strength + neg.strength) * 0.18 + cue_bonus;
                let score = (0.45 + 0.50 * (1.0 - (-raw).exp())).clamp(0.52, 0.98);

                let inc = Incongruity { positive: pos.tokens, negative: neg.tokens };
                if best.is_none_or(|(b_score, _)| score > b_score) {
                    best = Some((score, inc));
                }
            }
        }

        best
    }

    pub(super) fn detect_emoji_mismatch(
        &self,
        sentence_index: usize,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        categories: Option<&Categories<Sentences>>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Option<(f64, Incongruity)> {
        if emoji_spans.is_empty() {
            return None;
        }

        let positives = self.collect_positive_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let negatives = self.collect_negative_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let non_emoji_range = first_non_emoji_token_range(tok_range, emoji_spans);

        let cat_net_polarity = categories.map(|cats| {
            let pos = cats.score_of(sentence_index, "positive").unwrap_or(0.0);
            let neg = cats.score_of(sentence_index, "negative").unwrap_or(0.0);
            pos - neg
        });

        let mut best: Option<(f64, Incongruity)> = None;

        for &(em_range, em_sent) in emoji_spans {
            if em_sent < 0 {
                // Negative emoji paired with positive text opinion.
                let pos_fallback = match (cat_net_polarity, non_emoji_range) {
                    (Some(net), Some(r)) if net >= 0.25 => {
                        Some(SpanCandidate { tokens: r, strength: net * 2.5, cued: false })
                    }
                    _ => None,
                };
                let pos_text = positives
                    .iter()
                    .filter(|p| !ranges_overlap(p.tokens, em_range))
                    .max_by(|a, b| a.strength.total_cmp(&b.strength))
                    .copied()
                    .or(pos_fallback);

                if let Some(pos) = pos_text {
                    let em_mag = (em_sent.unsigned_abs() as f64).max(1.0);
                    let score = (0.58 + 0.12 * em_mag + 0.06 * pos.strength.min(3.0)).clamp(0.60, 0.96);
                    let inc = Incongruity { positive: pos.tokens, negative: em_range };
                    if best.is_none_or(|(b, _)| score > b) {
                        best = Some((score, inc));
                    }
                }
            } else if em_sent > 0 {
                // Positive emoji paired with negative text opinion or negative situation.
                let neg_fallback = match (cat_net_polarity, non_emoji_range) {
                    (Some(net), Some(r)) if net <= -0.25 => {
                        Some(SpanCandidate { tokens: r, strength: (-net) * 2.5, cued: false })
                    }
                    _ => None,
                };
                let neg_text = negatives
                    .iter()
                    .filter(|n| !ranges_overlap(n.tokens, em_range))
                    .max_by(|a, b| a.strength.total_cmp(&b.strength))
                    .copied()
                    .or(neg_fallback);

                if let Some(neg) = neg_text {
                    let em_mag = (em_sent as f64).max(1.0);
                    let score = (0.58 + 0.12 * em_mag + 0.06 * neg.strength.min(3.0)).clamp(0.60, 0.96);
                    let inc = Incongruity { positive: em_range, negative: neg.tokens };
                    if best.is_none_or(|(b, _)| score > b) {
                        best = Some((score, inc));
                    }
                }
            }
        }

        best
    }
}
