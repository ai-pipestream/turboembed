//! The decoder's beam bound, counted rather than timed, and the decode
//! checked against every path where the beam prunes nothing.

use turbo_ngram::NGramLanguageModel;

use super::{Candidate, DecodeStats, stupid_backoff_prob, unigram_prob, viterbi_decode_counted};

fn candidate(word: &str, cost: f64) -> Candidate {
    Candidate { word: word.to_owned(), trellis_edit_cost: cost, report_distance: 0, is_original: cost == 0.0 }
}

#[test]
fn the_beam_holds_the_states_and_the_work_to_its_width() {
    let group: Vec<String> = (0..64).map(|c| format!("w{c:02}")).collect();
    let sets = || -> Vec<Vec<Candidate>> {
        (0..12).map(|_| group.iter().enumerate().map(|(i, w)| candidate(w, i as f64 * 0.01)).collect()).collect()
    };
    for order in [3usize, 5, 6] {
        let mut lm = NGramLanguageModel::new(order).unwrap();
        lm.add(&vec!["w01"; order]).unwrap();
        for width in [64usize, 16] {
            let mut stats = DecodeStats::default();
            let out = viterbi_decode_counted(&lm, 1.0, width, sets(), &mut stats);
            assert_eq!(out.len(), 12);
            // 64 first-layer candidates, then each of the `width` kept
            // states times 64 candidates over the 11 later tokens. States
            // that share a history merge, so a layer builds at most that.
            assert_eq!(stats.expansions, 64 + 11 * 64 * width, "order {order}, width {width}");
            assert_eq!(stats.widest_kept, width, "order {order}, width {width}");
            assert!(stats.widest_built <= 64 * width, "order {order}, width {width}: {stats:?}");
        }
    }
}

/// The cost of one path, scored n-gram by n-gram as the decoder scores it.
fn path_cost(lm: &NGramLanguageModel, words: &[&str], edits: &[f64], use_bos: bool, use_eos: bool) -> f64 {
    let total = (lm.number_of_grams() as f64).max(lm.size() as f64).max(1.0);
    let h = lm.order() - 1;
    let mut cost: f64 = edits.iter().sum();
    for i in 0..words.len() {
        let mut ngram: Vec<&str> = Vec::new();
        if use_bos && i < h {
            ngram.push("<s>");
        }
        ngram.extend_from_slice(&words[i.saturating_sub(h)..=i]);
        let p =
            if ngram.len() == 1 { unigram_prob(lm, ngram[0], total) } else { stupid_backoff_prob(lm, &ngram, total) };
        cost -= p.ln();
    }
    if use_eos {
        let mut ngram: Vec<&str> = words[words.len().saturating_sub(h)..].to_vec();
        ngram.push("</s>");
        cost -= stupid_backoff_prob(lm, &ngram, total).ln();
    }
    cost
}

#[test]
fn with_a_beam_that_prunes_nothing_the_decode_is_the_cheapest_of_every_path() {
    let words = ["aa", "bb", "cc"];
    let edits = [0.0, 0.7, 1.3];
    let corpus: [&[&str]; 5] = [
        &["aa", "bb", "cc", "aa"],
        &["cc", "cc", "bb", "aa", "bb"],
        &["bb", "aa", "aa", "cc"],
        &["aa", "cc", "bb", "bb", "cc", "aa"],
        &["cc", "aa", "bb"],
    ];
    let positions = 6;
    for order in [2usize, 3] {
        for (use_bos, use_eos) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut lm = NGramLanguageModel::new(order).unwrap();
            for (k, sentence) in corpus.iter().enumerate() {
                let mut tokens: Vec<&str> = Vec::new();
                if use_bos {
                    tokens.push("<s>");
                }
                tokens.extend_from_slice(sentence);
                if use_eos {
                    tokens.push("</s>");
                }
                for _ in 0..=k {
                    lm.add(&tokens).unwrap();
                }
            }
            // Each position's candidates in a different order and with
            // different edit costs, so no path wins by its indices.
            let sets: Vec<Vec<Candidate>> = (0..positions)
                .map(|p| (0..3).map(|c| (c + p) % 3).map(|c| candidate(words[c], edits[(c + 2 * p) % 3])).collect())
                .collect();
            let mut stats = DecodeStats::default();
            let out = viterbi_decode_counted(&lm, 1.0, 1 << 12, sets.clone(), &mut stats);
            assert!(stats.widest_kept == stats.widest_built, "the beam pruned");
            let chosen: Vec<&str> = out.iter().map(|c| c.word.as_str()).collect();
            let chosen_edits: Vec<f64> = out.iter().map(|c| c.trellis_edit_cost).collect();
            let got = path_cost(&lm, &chosen, &chosen_edits, use_bos, use_eos);

            let mut best = f64::INFINITY;
            let mut paths = 0;
            for code in 0..3usize.pow(positions as u32) {
                let pick: Vec<&Candidate> = (0..positions).map(|p| &sets[p][code / 3usize.pow(p as u32) % 3]).collect();
                let w: Vec<&str> = pick.iter().map(|c| c.word.as_str()).collect();
                let e: Vec<f64> = pick.iter().map(|c| c.trellis_edit_cost).collect();
                best = best.min(path_cost(&lm, &w, &e, use_bos, use_eos));
                paths += 1;
            }
            assert_eq!(paths, 729);
            assert!((got - best).abs() < 1e-9, "order {order}, bos {use_bos}, eos {use_eos}: {got} against {best}");
        }
    }
}
