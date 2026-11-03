use super::*;

impl IncivilityAnnotator {
    /// Scores a single unit spanning `tok_start..tok_end` in `tokens`,
    /// combining lexicon matches, optional reply-context conflict against
    /// `parent_tok_range`, and optional embedding vector `unit_vec`.
    pub fn score_unit(
        &self,
        token_lower: &[String],
        tok_start: usize,
        tok_end: usize,
        parent_tok_range: Option<(usize, usize)>,
        unit_vec: Option<&[f32]>,
    ) -> Result<(IncivilityScores, Vec<IncivilityEvidenceSpan>), String> {
        self.score_unit_in_context(token_lower, tok_start, tok_end, parent_tok_range, unit_vec, None)
    }

    /// [`IncivilityAnnotator::score_unit`] with `context_vec`, the vector the
    /// head reads beside the unit's own ([`IncivilityVectorContext`]).
    pub fn score_unit_in_context(
        &self,
        token_lower: &[String],
        tok_start: usize,
        tok_end: usize,
        parent_tok_range: Option<(usize, usize)>,
        unit_vec: Option<&[f32]>,
        context_vec: Option<&[f32]>,
    ) -> Result<(IncivilityScores, Vec<IncivilityEvidenceSpan>), String> {
        let valid_threshold = |t: f64| t.is_finite() && (0.05..=0.99).contains(&t);
        if !valid_threshold(self.threshold)
            || self.thresholds.as_ref().is_some_and(|ts| ts.iter().any(|&t| !valid_threshold(t)))
        {
            return Err("incivility threshold must be finite and in 0.05..=0.99".into());
        }
        if !self.fusion_gate.is_finite() || !(0.0..=1.0).contains(&self.fusion_gate) {
            return Err("incivility fusion gate must be finite and in 0.0..=1.0".into());
        }
        let valid_range =
            |start: usize, end: usize| start <= end && end <= token_lower.len() && end < u32::MAX as usize;
        if !valid_range(tok_start, tok_end) {
            return Err("unit token range is reversed or outside the token slice".into());
        }
        if let Some((start, end)) = parent_tok_range
            && !valid_range(start, end)
        {
            return Err("parent token range is reversed or outside the token slice".into());
        }
        let mut scores = IncivilityScores::default();
        let mut evidence: Vec<IncivilityEvidenceSpan> = Vec::new();

        let lexicon = self.stage != IncivilityStage::Learned;
        if lexicon && tok_start < tok_end {
            // 1. Scan lexicon phrases inside `tok_start..tok_end`.
            let (lex_scores, lex_ev) = scan_lexicon(&self.lexicon, token_lower, tok_start, tok_end);
            for (lbl, val) in lex_scores.iter() {
                if val > scores.get(lbl) {
                    scores.set(lbl, val);
                }
            }
            evidence.extend(lex_ev);

            // 2. Reply-context conflict scoring (Opinion Conflicts, Zhang et al. 2018)
            // when enabled and a parent/quote token range is present.
            if self.use_reply_context
                && let Some((p_start, p_end)) = parent_tok_range
                && p_start < p_end
            {
                let (ctx_scores, ctx_ev) = score_reply_conflict(token_lower, p_start, p_end, tok_start, tok_end);
                for (lbl, val) in ctx_scores.iter() {
                    if val > 0.0 {
                        let combined = 1.0 - (1.0 - scores.get(lbl)) * (1.0 - val);
                        scores.set(lbl, combined.clamp(0.0, 1.0));
                    }
                }
                evidence.extend(ctx_ev);
            }
        }

        // 3. Score embedding vector when present and `self.vector_scorer` is configured.
        let scorer = self.vector_scorer.as_ref().filter(|_| self.stage != IncivilityStage::Lexicon);
        if let (Some(scorer), Some(vec)) = (scorer, unit_vec) {
            if vec.iter().chain(context_vec.into_iter().flatten()).any(|v| !v.is_finite()) {
                return Err("incivility vector must be finite".into());
            }
            let head_scores = scorer.score_incivility_in_context(vec, context_vec)?;
            if head_scores.iter().into_iter().any(|(_, v)| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
                return Err("incivility head scores must be finite and in 0..=1".into());
            }
            for (lbl, h_val) in head_scores.iter() {
                let l_val = scores.get(lbl);
                // If both lexicon and head fire at or above `self.fusion_gate`, combine probabilistically;
                // otherwise take the stronger signal (`#963`).
                let merged = if l_val >= self.fusion_gate && h_val >= self.fusion_gate {
                    (1.0 - (1.0 - l_val) * (1.0 - h_val)).clamp(0.0, 1.0)
                } else {
                    l_val.max(h_val).clamp(0.0, 1.0)
                };
                scores.set(lbl, merged);
            }
        }

        // 4. Co-occurrence rule for SevereToxicity: when Threat co-occurs with
        // Insult, Profanity, IdentityAttack, or Intolerance at >= threshold,
        // elevate SevereToxicity and merge their evidence spans.
        // The learned stage reports the head's scores as they are.
        let other_labels = [
            IncivilityLabel::Insult,
            IncivilityLabel::Profanity,
            IncivilityLabel::IdentityAttack,
            IncivilityLabel::Intolerance,
        ];
        let max_positive_other = other_labels
            .into_iter()
            .filter(|&l| scores.get(l) >= self.threshold_for(l))
            .map(|l| scores.get(l))
            .fold(0.0, f64::max);
        if lexicon
            && self.use_severe_toxicity_rule
            && scores.threat >= self.threshold_for(IncivilityLabel::Threat)
            && max_positive_other > 0.0
        {
            let sev_thresh = self.threshold_for(IncivilityLabel::SevereToxicity);
            let sev = (0.55 * scores.threat + 0.45 * max_positive_other).clamp(sev_thresh, 0.99);
            if sev > scores.severe_toxicity {
                scores.severe_toxicity = sev;
            }
            let supporting: Vec<IxRange<Tokens>> = evidence
                .iter()
                .filter(|(l, _)| {
                    matches!(
                        l,
                        IncivilityLabel::Threat
                            | IncivilityLabel::Insult
                            | IncivilityLabel::Profanity
                            | IncivilityLabel::IdentityAttack
                            | IncivilityLabel::Intolerance
                    )
                })
                .map(|(_, r)| *r)
                .collect();
            for r in supporting {
                evidence.push((IncivilityLabel::SevereToxicity, r));
            }
        }

        // 5. Guarantee that every positive label (`score >= self.threshold_for(lbl)`) has at least
        // one non-empty `IxRange<Tokens>` evidence span inside `tok_start..tok_end`.
        if tok_start < tok_end {
            let fallback_span = select_salient_token_span(token_lower, tok_start, tok_end);
            for (lbl, val) in scores.iter() {
                if val >= self.threshold_for(lbl) && !evidence.iter().any(|(l, _)| *l == lbl) {
                    evidence.push((lbl, fallback_span));
                }
            }
        }

        evidence.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        evidence.dedup();
        Ok((scores, evidence))
    }
}
