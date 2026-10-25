use super::*;

impl LexiconSentiment {
    /// Scores raw text in VADER-compatible mode, matching Python
    /// `vaderSentiment`'s `SentimentIntensityAnalyzer.polarity_scores(text)`.
    pub fn score_text(&self, text: &str) -> VaderScore {
        let raw_tokens: Vec<&str> =
            text.split(turbo_unicode::whitespace::is_python_space).filter(|t| !t.is_empty()).collect();
        let words_and_emoticons: Vec<&str> = raw_tokens.iter().map(|&t| strip_punc_if_word(t)).collect();
        let lower_words: Vec<String> = words_and_emoticons.iter().map(|w| w.to_lowercase()).collect();
        let is_cap_diff = allcap_differential(&words_and_emoticons);

        let n = words_and_emoticons.len();
        let mut sentiments = Vec::with_capacity(n);

        for i in 0..n {
            let item = words_and_emoticons[i];
            let item_lower = &lower_words[i];
            if booster_value(item_lower).is_some() {
                sentiments.push(0.0);
                continue;
            }
            if i + 1 < n && item_lower == "kind" && lower_words[i + 1] == "of" {
                sentiments.push(0.0);
                continue;
            }
            let (valence, _, _) =
                self.window_token_valence(item, item_lower, i, &words_and_emoticons, &lower_words, is_cap_diff, None);
            sentiments.push(valence);
        }

        apply_but_check_vader(&lower_words, &mut sentiments);
        compute_vader_score(&sentiments, text, self.alpha)
    }

    /// Scores a single sentence over `tokens` in `tok_range` (`start..end`),
    /// producing both the [`VaderScore`] and the exact [`Opinion`] evidence
    /// spans (`IxRange<Tokens>`).
    ///
    /// When `deps` (`opennlp:dependencies`) is supplied and `self.negation_mode`
    /// is [`NegationMode::Auto`] or [`NegationMode::Dependency`], negation scope
    /// is read from dependency relations within the sentence rather than the
    /// fixed 3-token window.
    ///
    /// # Panics
    /// The caller must supply valid UTF-8 spans, an in-bounds token range,
    /// and aligned dependency/emoji layers. Use the annotator entry point
    /// for checked document inputs.
    pub fn score_sentence(
        &self,
        view_text: &str,
        sentence_span: Span,
        tokens: &Tokens,
        tok_range: IxRange<Tokens>,
        deps: Option<&Dependencies>,
        emoji: Option<&Emoji>,
    ) -> SentenceSentiment {
        let start = tok_range.start.get();
        let end = tok_range.end.get();
        let use_deps = match self.negation_mode {
            NegationMode::Auto | NegationMode::Dependency => deps.is_some(),
            NegationMode::Window => false,
        };
        if start >= end {
            return SentenceSentiment {
                score: VaderScore::default(),
                opinions: Vec::new(),
                used_dependencies: use_deps,
            };
        }

        let n = end - start;
        let mut raw_surfaces = Vec::with_capacity(n);
        let mut words = Vec::with_capacity(n);
        let mut lower_words = Vec::with_capacity(n);
        for tok_idx in start..end {
            let sp = tokens.spans.get(Ix::new(tok_idx as u32));
            let surf = &view_text[sp.range()];
            let stripped = strip_punc_if_word(surf);
            raw_surfaces.push(surf);
            words.push(stripped);
            lower_words.push(stripped.to_lowercase());
        }

        let is_cap_diff = allcap_differential(&words);
        let mut sentiments = Vec::with_capacity(n);
        let mut priors = Vec::with_capacity(n);

        for i in 0..n {
            let tok_global = start + i;
            let item = words[i];
            let item_lower = &lower_words[i];
            if booster_value(item_lower).is_some() || (i + 1 < n && item_lower == "kind" && lower_words[i + 1] == "of")
            {
                sentiments.push(0.0);
                priors.push(0.0);
                continue;
            }

            let emoji_prior = if self.use_opennlp_emoji {
                lookup_emoji_prior(raw_surfaces[i], tokens.spans.get(Ix::new(tok_global as u32)), view_text, emoji)
            } else {
                None
            };

            let (valence, prior, _negated) = match deps {
                Some(deps) if use_deps => self.dependency_token_valence(
                    item,
                    item_lower,
                    i,
                    start,
                    end,
                    &words,
                    &lower_words,
                    is_cap_diff,
                    emoji_prior,
                    deps,
                ),
                _ => self.window_token_valence(item, item_lower, i, &words, &lower_words, is_cap_diff, emoji_prior),
            };
            sentiments.push(valence);
            priors.push(prior);
        }

        // Contrastive "but": by token position, not by vaderSentiment's value
        // lookup (`sentiments.index(sentiment)`), which rescales the first token
        // carrying a repeated value instead of the token itself. `score_text`
        // keeps that quirk for parity; evidence spans must point at the token
        // that was actually rescaled.
        if let Some(but_idx) = lower_words.iter().position(|w| w == "but") {
            for (sidx, s) in sentiments.iter_mut().enumerate() {
                if sidx < but_idx {
                    *s *= 0.5;
                } else if sidx > but_idx {
                    *s *= 1.5;
                }
            }
        }

        let mut opinions = Vec::new();
        for i in 0..n {
            let final_val = sentiments[i];
            let prior = priors[i];
            if final_val == 0.0 {
                continue;
            }
            // `negated` must agree with the sign of the adjusted valence so that
            // `Opinion::effective_polarity` reproduces it: a doubly negated token
            // ("no", then "not") ends up positive and is reported as not negated.
            // A token whose prior was zeroed ("NO" before a lexicon word) but
            // that the ALLCAPS or booster rules still scored keeps its evidence
            // row with the adjusted valence as its polarity.
            let (polarity, intensity, negated) = if prior == 0.0 {
                (final_val, 1.0, false)
            } else {
                (prior, (final_val / prior).abs(), final_val.signum() != prior.signum())
            };
            let tok_ix = (start + i) as u32;
            opinions.push(Opinion { tokens: IxRange::new(tok_ix, tok_ix + 1), polarity, intensity, negated });
        }

        let sent_text = &view_text[sentence_span.range()];
        let score = compute_vader_score(&sentiments, sent_text, self.alpha);
        SentenceSentiment { score, opinions, used_dependencies: use_deps }
    }
}
