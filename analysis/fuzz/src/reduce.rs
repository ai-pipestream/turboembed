//! Reducing a diverging text to a minimal one: delta debugging (ddmin,
//! Zeller and Hildebrandt, 2002) over the text's lines, then over its
//! characters. The oracle says, for a batch of candidate texts, which of
//! them still diverge; batching matters because the oracle here is a
//! container run of the JVM generator, whose start up costs seconds, so
//! every round of the search asks about all its candidates at once.

/// Answers, for each candidate text, whether it still shows the failure.
pub trait Oracle {
    fn fails(&mut self, texts: &[String]) -> Result<Vec<bool>, String>;
}

impl<F: FnMut(&[String]) -> Result<Vec<bool>, String>> Oracle for F {
    fn fails(&mut self, texts: &[String]) -> Result<Vec<bool>, String> {
        self(texts)
    }
}

/// How a reduction went.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduced {
    /// The smallest text found that still fails.
    pub text: String,
    /// How many times the oracle was asked (container runs).
    pub rounds: usize,
    /// How many candidate texts it was asked about in all.
    pub candidates: usize,
}

/// Reduces `text`, which fails, to a text that still fails and from which
/// no one line, then no one character, can be removed. `max_rounds` bounds
/// the oracle calls; when it is reached the best text so far is returned.
pub fn reduce(text: &str, oracle: &mut dyn Oracle, max_rounds: usize) -> Result<Reduced, String> {
    let mut state = Reduced { text: text.to_string(), rounds: 0, candidates: 0 };
    let lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
    let lines = ddmin(lines, oracle, &mut state, max_rounds)?;
    let joined: String = lines.concat();
    let chars: Vec<String> = joined.chars().map(String::from).collect();
    let chars = ddmin(chars, oracle, &mut state, max_rounds)?;
    state.text = chars.concat();
    if !state.text.is_empty() && state.rounds < max_rounds {
        let empty = [String::new()];
        let fails = oracle.fails(&empty)?;
        state.rounds += 1;
        state.candidates += 1;
        if fails.len() != 1 {
            return Err(format!("the oracle answered {} of 1 candidates", fails.len()));
        }
        if fails[0] {
            state.text.clear();
        }
    }
    Ok(state)
}

/// ddmin over `units`, whose concatenation fails. Each round tests every
/// subset and every complement of the current partition in one oracle
/// call; a failing subset narrows the search to it, a failing complement
/// drops the subset, and otherwise the partition gets finer.
fn ddmin(
    mut units: Vec<String>,
    oracle: &mut dyn Oracle,
    state: &mut Reduced,
    max_rounds: usize,
) -> Result<Vec<String>, String> {
    let mut n = 2;
    while units.len() >= 2 && state.rounds < max_rounds {
        let chunks = partition(&units, n);
        let mut candidates: Vec<Vec<String>> = Vec::with_capacity(2 * chunks.len());
        let mut texts: Vec<String> = Vec::with_capacity(2 * chunks.len());
        let mut is_subset: Vec<bool> = Vec::with_capacity(2 * chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let sub_text = chunk.concat();
            if !texts.contains(&sub_text) {
                texts.push(sub_text);
                candidates.push(chunk.clone());
                is_subset.push(true);
            }
            if n > 2 {
                let mut complement = Vec::new();
                for (j, other) in chunks.iter().enumerate() {
                    if j != i {
                        complement.extend(other.iter().cloned());
                    }
                }
                let comp_text = complement.concat();
                if !texts.contains(&comp_text) {
                    texts.push(comp_text);
                    candidates.push(complement);
                    is_subset.push(false);
                }
            }
        }
        let fails = oracle.fails(&texts)?;
        state.rounds += 1;
        state.candidates += texts.len();
        if fails.len() != texts.len() {
            return Err(format!("the oracle answered {} of {} candidates", fails.len(), texts.len()));
        }
        // A subset first (it is the smaller), then a complement.
        let subset = (0..candidates.len()).find(|&k| fails[k] && is_subset[k]);
        let complement = (0..candidates.len()).find(|&k| fails[k] && !is_subset[k]);
        if let Some(k) = subset {
            units = candidates[k].clone();
            n = 2;
        } else if let Some(k) = complement {
            units = candidates[k].clone();
            n = (n - 1).max(2);
        } else if n < units.len() {
            n = (2 * n).min(units.len());
        } else {
            break;
        }
    }
    Ok(units)
}

/// `units` cut into `n` parts of as equal a size as possible, in order.
fn partition(units: &[String], n: usize) -> Vec<Vec<String>> {
    let n = n.min(units.len()).max(1);
    let (base, extra) = (units.len() / n, units.len() % n);
    let mut out = Vec::with_capacity(n);
    let mut at = 0;
    for i in 0..n {
        let len = base + usize::from(i < extra);
        out.push(units[at..at + len].to_vec());
        at += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An oracle that fails on any text holding every one of `needles`,
    /// counting its calls.
    fn needles(needles: &'static [&'static str]) -> impl FnMut(&[String]) -> Result<Vec<bool>, String> {
        move |texts: &[String]| Ok(texts.iter().map(|t| needles.iter().all(|n| t.contains(n))).collect())
    }

    #[test]
    fn a_text_reduces_to_the_characters_the_failure_needs() {
        let text = "the quick brown fox\njumps over\nthe lazy dog\n";
        let mut oracle = needles(&["z"]);
        let r = reduce(text, &mut oracle, 100).unwrap();
        assert_eq!(r.text, "z");
        assert!(r.rounds > 1 && r.candidates >= r.rounds);
    }

    #[test]
    fn an_unconditional_failure_reduces_to_the_empty_string_and_deduplicates_batches() {
        let mut oracle = |texts: &[String]| {
            for (i, a) in texts.iter().enumerate() {
                assert!(!texts[..i].contains(a), "duplicate candidate {a:?} in batch {texts:?}");
            }
            Ok(vec![true; texts.len()])
        };
        let r = reduce("aaaa\naaaa\n", &mut oracle, 100).unwrap();
        assert_eq!(r.text, "");

        let mut keep_two_as = |texts: &[String]| {
            for (i, a) in texts.iter().enumerate() {
                assert!(!texts[..i].contains(a), "duplicate candidate {a:?} in batch {texts:?}");
            }
            Ok(texts.iter().map(|t| t == "aaaa").collect())
        };
        let r = reduce("aaaa", &mut keep_two_as, 100).unwrap();
        assert_eq!(r.text, "aaaa");
    }

    #[test]
    fn two_needles_far_apart_both_stay() {
        let text = "alpha beta\ngamma delta\nepsilon zeta\neta theta\n";
        let mut oracle = needles(&["b", "z"]);
        let r = reduce(text, &mut oracle, 100).unwrap();
        // Minimal: nothing can be removed, and both needles are there.
        assert!(r.text.contains('b') && r.text.contains('z'), "{:?}", r.text);
        assert_eq!(r.text.len(), 2, "{:?}", r.text);
    }

    #[test]
    fn a_failure_needing_everything_keeps_everything() {
        let text = "ab\ncd";
        let mut oracle = needles(&["a", "b", "c", "d"]);
        let r = reduce(text, &mut oracle, 100).unwrap();
        assert_eq!(r.text, "abcd");
    }

    #[test]
    fn the_round_limit_stops_the_search_with_a_failing_text() {
        let text: String = (0..200).map(|i| format!("line {i}\n")).collect();
        let mut oracle = needles(&["line 150"]);
        let r = reduce(&text, &mut oracle, 2).unwrap();
        assert_eq!(r.rounds, 2);
        assert!(r.text.contains("line 150"));
        assert!(r.text.len() < text.len());
    }

    #[test]
    fn an_oracle_error_comes_back() {
        let mut oracle = |_: &[String]| Err::<Vec<bool>, String>("the container did not start".into());
        assert_eq!(reduce("a\nb", &mut oracle, 5).unwrap_err(), "the container did not start");
    }

    #[test]
    fn partitions_are_contiguous_and_cover_the_units() {
        let units: Vec<String> = (0..7).map(|i| i.to_string()).collect();
        let parts = partition(&units, 3);
        assert_eq!(parts.iter().map(Vec::len).collect::<Vec<_>>(), [3, 2, 2]);
        assert_eq!(parts.concat(), units);
    }
}
