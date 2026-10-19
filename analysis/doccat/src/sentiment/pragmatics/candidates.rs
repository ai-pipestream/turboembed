use super::*;

impl PragmaticsAnnotator {
    pub(super) fn collect_positive_candidates(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Vec<SpanCandidate> {
        let mut out = Vec::new();
        let start = tok_range.start.get();
        let end = tok_range.end.get();

        // 1. Multi-token sarcastic positive phrases in this sentence.
        for &(phrase, strength) in SARCASTIC_POSITIVE_PHRASES {
            let plen = phrase.len();
            if plen == 0 || end - start < plen {
                continue;
            }
            for i in start..=end - plen {
                if (0..plen).all(|k| lower_tokens[i + k] == phrase[k]) {
                    let r = IxRange::new(i as u32, (i + plen) as u32);
                    if !overlaps_any_emoji(r, emoji_spans) {
                        out.push(SpanCandidate { tokens: r, strength, cued: true });
                    }
                }
            }
        }

        // 2. Positive opinions from `turbo:opinions`.
        if let Some(ops) = opinions {
            for (_, op) in ops.iter() {
                if op.tokens.start.get() < start || op.tokens.end.get() > end {
                    continue;
                }
                if overlaps_any_emoji(op.tokens, emoji_spans) {
                    continue;
                }
                let eff = op.effective_polarity();
                if eff >= self.min_positive_polarity {
                    let cued = has_preceding_sarcastic_cue(op.tokens, tok_range, lower_tokens);
                    out.push(SpanCandidate { tokens: op.tokens, strength: eff, cued });
                }
            }
        } else {
            // Fallback when `turbo:opinions` is not on the document: scan tokens
            // with the bundled VADER lexicon.
            for (i, w) in lower_tokens.iter().enumerate().take(end).skip(start) {
                if w.is_empty() {
                    continue;
                }
                let r = IxRange::new(i as u32, (i + 1) as u32);
                if overlaps_any_emoji(r, emoji_spans) {
                    continue;
                }
                if let Some(valence) = self.lexicon.get(w)
                    && valence >= self.min_positive_polarity
                    && !has_preceding_negation(i, start, lower_tokens)
                {
                    let cued = has_preceding_sarcastic_cue(r, tok_range, lower_tokens);
                    out.push(SpanCandidate { tokens: r, strength: valence, cued });
                }
            }
        }

        out
    }

    pub(super) fn collect_negative_candidates(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Vec<SpanCandidate> {
        let mut out = Vec::new();
        let start = tok_range.start.get();
        let end = tok_range.end.get();

        // 1. Riloff negative situation phrases.
        for &(phrase, strength) in NEGATIVE_SITUATIONS {
            let plen = phrase.len();
            if plen == 0 || end - start < plen {
                continue;
            }
            for i in start..=end - plen {
                if (0..plen).all(|k| lower_tokens[i + k] == phrase[k]) {
                    let r = IxRange::new(i as u32, (i + plen) as u32);
                    if !overlaps_any_emoji(r, emoji_spans) {
                        out.push(SpanCandidate { tokens: r, strength, cued: true });
                    }
                }
            }
        }

        // 2. Negative opinions from `turbo:opinions`.
        if let Some(ops) = opinions {
            for (_, op) in ops.iter() {
                if op.tokens.start.get() < start || op.tokens.end.get() > end {
                    continue;
                }
                if overlaps_any_emoji(op.tokens, emoji_spans) {
                    continue;
                }
                let eff = op.effective_polarity();
                if eff <= -self.min_negative_polarity {
                    // Exclude "yeah right", which VADER scores negatively but
                    // functions as the positive ironic cue in a pair.
                    let len = op.tokens.end.get() - op.tokens.start.get();
                    if len == 2
                        && lower_tokens[op.tokens.start.get()] == "yeah"
                        && lower_tokens[op.tokens.start.get() + 1] == "right"
                    {
                        continue;
                    }
                    out.push(SpanCandidate { tokens: op.tokens, strength: -eff, cued: false });
                }
            }
        } else {
            for (i, w) in lower_tokens.iter().enumerate().take(end).skip(start) {
                if w.is_empty() {
                    continue;
                }
                let r = IxRange::new(i as u32, (i + 1) as u32);
                if overlaps_any_emoji(r, emoji_spans) {
                    continue;
                }
                if let Some(valence) = self.lexicon.get(w) {
                    let negated = has_preceding_negation(i, start, lower_tokens);
                    let eff = if negated { -0.74 * valence } else { valence };
                    if eff <= -self.min_negative_polarity {
                        out.push(SpanCandidate { tokens: r, strength: -eff, cued: false });
                    }
                }
            }
        }

        out
    }
}
