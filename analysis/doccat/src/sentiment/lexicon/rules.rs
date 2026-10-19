use super::*;

pub(super) fn negation_check(valence: f64, lower_words: &[String], start_i: usize, i: usize) -> (f64, bool) {
    if start_i == 0 {
        if is_negation_word(&lower_words[i - 1]) {
            return (valence * N_SCALAR, true);
        }
    } else if start_i == 1 {
        if lower_words[i - 2] == "never" && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this") {
            return (valence * 1.25, false);
        } else if lower_words[i - 2] == "without" && lower_words[i - 1] == "doubt" {
            return (valence, false);
        } else if is_negation_word(&lower_words[i - 2]) {
            return (valence * N_SCALAR, true);
        }
    } else if start_i == 2 {
        if (lower_words[i - 3] == "never" && (lower_words[i - 2] == "so" || lower_words[i - 2] == "this"))
            || (lower_words[i - 1] == "so" || lower_words[i - 1] == "this")
        {
            return (valence * 1.25, false);
        } else if lower_words[i - 3] == "without" && (lower_words[i - 2] == "doubt" || lower_words[i - 1] == "doubt") {
            return (valence, false);
        } else if is_negation_word(&lower_words[i - 3]) {
            return (valence * N_SCALAR, true);
        }
    }
    (valence, false)
}

pub(super) fn special_idioms_check(valence: f64, lower_words: &[String], i: usize) -> (f64, Option<f64>) {
    let mut out = valence;
    let mut matched_idiom = None;

    let onezero = format!("{} {}", lower_words[i - 1], lower_words[i]);
    let twoonezero = format!("{} {} {}", lower_words[i - 2], lower_words[i - 1], lower_words[i]);
    let twoone = format!("{} {}", lower_words[i - 2], lower_words[i - 1]);
    let threetwoone = format!("{} {} {}", lower_words[i - 3], lower_words[i - 2], lower_words[i - 1]);
    let threetwo = format!("{} {}", lower_words[i - 3], lower_words[i - 2]);

    for seq in [&onezero, &twoonezero, &twoone, &threetwoone, &threetwo] {
        if let Some(v) = special_case_value(seq) {
            out = v;
            matched_idiom = Some(v);
            break;
        }
    }

    if lower_words.len() - 1 > i {
        let zeroone = format!("{} {}", lower_words[i], lower_words[i + 1]);
        if let Some(v) = special_case_value(&zeroone) {
            out = v;
            matched_idiom = Some(v);
        }
    }
    if lower_words.len() - 1 > i + 1 {
        let zeroonetwo = format!("{} {} {}", lower_words[i], lower_words[i + 1], lower_words[i + 2]);
        if let Some(v) = special_case_value(&zeroonetwo) {
            out = v;
            matched_idiom = Some(v);
        }
    }

    for ngram in [&threetwoone, &threetwo, &twoone] {
        if let Some(b) = booster_value(ngram) {
            out += b;
        }
    }

    (out, matched_idiom)
}

pub(super) fn least_check(lexicon: &VaderLexicon, valence: f64, lower_words: &[String], i: usize) -> (f64, bool) {
    if i > 1 && !lexicon.contains(&lower_words[i - 1]) && lower_words[i - 1] == "least" {
        if lower_words[i - 2] != "at" && lower_words[i - 2] != "very" {
            return (valence * N_SCALAR, true);
        }
    } else if i > 0 && !lexicon.contains(&lower_words[i - 1]) && lower_words[i - 1] == "least" {
        return (valence * N_SCALAR, true);
    }
    (valence, false)
}

/// Applies `_but_check` exactly as `vaderSentiment.py` does.
pub(super) fn apply_but_check_vader(lower_words: &[String], sentiments: &mut [f64]) {
    let Some(bi) = lower_words.iter().position(|w| w == "but") else {
        return;
    };
    let snapshot = sentiments.to_vec();
    for sentiment in snapshot {
        if let Some(si) = sentiments.iter().position(|&s| s == sentiment) {
            if si < bi {
                sentiments[si] = sentiment * 0.5;
            } else if si > bi {
                sentiments[si] = sentiment * 1.5;
            }
        }
    }
}

pub(super) fn punctuation_emphasis(text: &str) -> f64 {
    let ep_count = text.bytes().filter(|&b| b == b'!').count().min(4);
    let ep_amp = (ep_count as f64) * 0.292;

    let qm_count = text.bytes().filter(|&b| b == b'?').count();
    let qm_amp = if qm_count > 1 { if qm_count <= 3 { (qm_count as f64) * 0.18 } else { 0.96 } } else { 0.0 };

    ep_amp + qm_amp
}

pub(super) fn sift_sentiment_scores(sentiments: &[f64]) -> (f64, f64, usize) {
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    let mut neu_count = 0usize;
    for &s in sentiments {
        if s > 0.0 {
            pos_sum += s + 1.0;
        }
        if s < 0.0 {
            neg_sum += s - 1.0;
        }
        if s == 0.0 {
            neu_count += 1;
        }
    }
    (pos_sum, neg_sum, neu_count)
}

pub(super) fn compute_vader_score(sentiments: &[f64], text: &str, alpha: f64) -> VaderScore {
    if sentiments.is_empty() {
        return VaderScore::default();
    }
    let mut sum_s: f64 = sentiments.iter().sum();
    let punct_amp = punctuation_emphasis(text);
    if sum_s > 0.0 {
        sum_s += punct_amp;
    } else if sum_s < 0.0 {
        sum_s -= punct_amp;
    }
    let compound = normalize_score(sum_s, alpha);

    let (mut pos_sum, mut neg_sum, neu_count) = sift_sentiment_scores(sentiments);
    if pos_sum > neg_sum.abs() {
        pos_sum += punct_amp;
    } else if pos_sum < neg_sum.abs() {
        neg_sum -= punct_amp;
    }

    let total = pos_sum + neg_sum.abs() + (neu_count as f64);
    if total == 0.0 {
        return VaderScore { neg: 0.0, neu: 0.0, pos: 0.0, compound };
    }
    let pos = (pos_sum / total).abs();
    let neg = (neg_sum / total).abs();
    let neu = ((neu_count as f64) / total).abs();
    VaderScore { neg, neu, pos, compound }
}
