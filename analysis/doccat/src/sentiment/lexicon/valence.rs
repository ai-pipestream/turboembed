use super::*;

impl LexiconSentiment {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn window_token_valence(
        &self,
        item: &str,
        item_lower: &str,
        i: usize,
        words: &[&str],
        lower_words: &[String],
        is_cap_diff: bool,
        emoji_prior: Option<f64>,
    ) -> (f64, f64, bool) {
        let base_opt = self.lexicon.get(item_lower).or(emoji_prior);
        let Some(base_valence) = base_opt else {
            return (0.0, 0.0, false);
        };
        let n = words.len();
        let mut prior = base_valence;
        let mut valence = base_valence;
        let mut negated = false;

        // "no" immediately before a lexicon word acts as a negator rather than its
        // own lexicon item. Its valence becomes 0.0 but, as in vaderSentiment, the
        // ALLCAPS and booster rules below still apply to it ("NO good" scores
        // "NO" at -0.733).
        if item_lower == "no" && i + 1 < n && self.lexicon.contains(&lower_words[i + 1]) {
            valence = 0.0;
            prior = 0.0;
        }
        if (i > 0 && lower_words[i - 1] == "no")
            || (i > 1 && lower_words[i - 2] == "no")
            || (i > 2 && lower_words[i - 3] == "no" && matches!(lower_words[i - 1].as_str(), "or" | "nor"))
        {
            valence = base_valence * N_SCALAR;
            negated = !negated;
        }

        if is_all_caps(item) && is_cap_diff {
            if valence > 0.0 {
                valence += C_INCR;
            } else {
                valence -= C_INCR;
            }
        }

        for start_i in 0..3 {
            if i > start_i {
                let prev_idx = i - (start_i + 1);
                let prev_lower = &lower_words[prev_idx];
                if !self.lexicon.contains(prev_lower) {
                    let mut s = scalar_inc_dec(words[prev_idx], valence, is_cap_diff);
                    if start_i == 1 && s != 0.0 {
                        s *= 0.95;
                    }
                    if start_i == 2 && s != 0.0 {
                        s *= 0.9;
                    }
                    valence += s;
                    let (v_after_neg, neg_flipped) = negation_check(valence, lower_words, start_i, i);
                    valence = v_after_neg;
                    if neg_flipped {
                        negated = !negated;
                    }
                    if start_i == 2 {
                        let (v_after_idiom, idiom_prior) = special_idioms_check(valence, lower_words, i);
                        valence = v_after_idiom;
                        if let Some(ip) = idiom_prior {
                            prior = ip;
                            negated = false;
                        }
                    }
                }
            }
        }

        let (v_after_least, least_flipped) = least_check(&self.lexicon, valence, lower_words, i);
        valence = v_after_least;
        if least_flipped {
            negated = !negated;
        }

        (valence, prior, negated)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dependency_token_valence(
        &self,
        item: &str,
        item_lower: &str,
        i: usize,
        sent_start: usize,
        sent_end: usize,
        words: &[&str],
        lower_words: &[String],
        is_cap_diff: bool,
        emoji_prior: Option<f64>,
        deps: &Dependencies,
    ) -> (f64, f64, bool) {
        let base_opt = self.lexicon.get(item_lower).or(emoji_prior);
        let Some(base_valence) = base_opt else {
            return (0.0, 0.0, false);
        };
        let n = words.len();
        let mut prior = base_valence;
        let mut valence = base_valence;

        // "no" acts as a negator rather than its own lexicon item when it precedes a
        // lexicon word or is a dependent of a lexicon head in this sentence. Its
        // valence becomes 0.0 but the ALLCAPS and booster rules below still apply,
        // as in the window path.
        let tok_global = sent_start + i;
        if item_lower == "no" {
            let precedes_lexicon_word = i + 1 < n && self.lexicon.contains(&lower_words[i + 1]);
            let modifies_lexicon_head = match deps.head.get(tok_global) {
                Some(Some(head_ix)) => {
                    let h = head_ix.get();
                    h >= sent_start
                        && h < sent_end
                        && h != tok_global
                        && self.lexicon.contains(&lower_words[h - sent_start])
                }
                _ => false,
            };
            if precedes_lexicon_word || modifies_lexicon_head {
                valence = 0.0;
                prior = 0.0;
            }
        }

        if is_all_caps(item) && is_cap_diff {
            if valence > 0.0 {
                valence += C_INCR;
            } else {
                valence -= C_INCR;
            }
        }

        // Apply booster/dampener scalars and special-case idioms (plus "never so/this"
        // and "without doubt" amplifier patterns), while leaving negation scope to the
        // dependency tree.
        for start_i in 0..3 {
            if i > start_i {
                let prev_idx = i - (start_i + 1);
                let prev_lower = &lower_words[prev_idx];
                if !self.lexicon.contains(prev_lower) {
                    let mut s = scalar_inc_dec(words[prev_idx], valence, is_cap_diff);
                    if start_i == 1 && s != 0.0 {
                        s *= 0.95;
                    }
                    if start_i == 2 && s != 0.0 {
                        s *= 0.9;
                    }
                    valence += s;
                    if (start_i == 1
                        && lower_words[i - 2] == "never"
                        && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this"))
                        || (start_i == 2
                            && ((lower_words[i - 3] == "never"
                                && (lower_words[i - 2] == "so" || lower_words[i - 2] == "this"))
                                || (lower_words[i - 1] == "so" || lower_words[i - 1] == "this")))
                    {
                        valence *= 1.25;
                    }
                    if start_i == 2 {
                        let (v_after_idiom, idiom_prior) = special_idioms_check(valence, lower_words, i);
                        valence = v_after_idiom;
                        if let Some(ip) = idiom_prior {
                            prior = ip;
                        }
                    }
                }
            }
        }

        // Check whether `never so/this` or `without doubt` suppressed negation locally.
        let never_so_this =
            (i >= 2 && lower_words[i - 2] == "never" && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this"))
                || (i >= 3
                    && lower_words[i - 3] == "never"
                    && (lower_words[i - 2] == "so"
                        || lower_words[i - 2] == "this"
                        || lower_words[i - 1] == "so"
                        || lower_words[i - 1] == "this"));
        let without_doubt = (i >= 2 && lower_words[i - 2] == "without" && lower_words[i - 1] == "doubt")
            || (i >= 3
                && lower_words[i - 3] == "without"
                && (lower_words[i - 2] == "doubt" || lower_words[i - 1] == "doubt"));

        let mut negated = false;
        if !never_so_this
            && !without_doubt
            && is_negated_in_dependencies(i, sent_start, sent_end, lower_words, &self.lexicon, deps)
        {
            valence *= N_SCALAR;
            negated = true;
        }

        (valence, prior, negated)
    }
}
