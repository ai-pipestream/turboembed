//! Generators that remember the outcomes of earlier sentences in a
//! document.

use std::collections::HashMap;

use super::FeatureGenerator;
use crate::features::Features;

/// The outcome each token had the last time it was seen, filled by
/// `update_adaptive_data`.
#[derive(Default)]
struct OutcomeMap {
    outcomes: HashMap<String, String>,
}

impl OutcomeMap {
    fn get(&self, token: &str) -> Option<&str> {
        self.outcomes.get(token).map(String::as_str)
    }

    fn update(&mut self, tokens: &[&str], outcomes: &[&str]) {
        for (token, outcome) in tokens.iter().zip(outcomes) {
            match self.outcomes.get_mut(*token) {
                Some(old) => {
                    old.clear();
                    old.push_str(outcome);
                }
                None => {
                    self.outcomes.insert((*token).to_owned(), (*outcome).to_owned());
                }
            }
        }
    }
}

/// `PreviousMapFeatureGenerator`: `pd=` and the outcome the token had when
/// it was last seen in an earlier sentence of the document, or `pd=null`.
#[derive(Default)]
pub struct PreviousMap {
    map: OutcomeMap,
}

impl PreviousMap {
    pub fn new() -> PreviousMap {
        PreviousMap::default()
    }
}

impl FeatureGenerator for PreviousMap {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        out.push_str("pd=").push_or_null(self.map.get(tokens[index])).finish();
    }

    fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        self.map.update(tokens, outcomes);
    }

    fn clear_adaptive_data(&mut self) {
        self.map.outcomes.clear();
    }
}

/// `PreviousTwoMapFeatureGenerator`: from the second token on, `ppd=` and
/// the remembered outcomes of the token and of the one before it, each
/// `null` when there is none.
#[derive(Default)]
pub struct PreviousTwoMap {
    map: OutcomeMap,
}

impl PreviousTwoMap {
    pub fn new() -> PreviousTwoMap {
        PreviousTwoMap::default()
    }
}

impl FeatureGenerator for PreviousTwoMap {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        if index > 0 {
            out.push_str("ppd=")
                .push_or_null(self.map.get(tokens[index]))
                .push_str(",")
                .push_or_null(self.map.get(tokens[index - 1]))
                .finish();
        }
    }

    fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        self.map.update(tokens, outcomes);
    }

    fn clear_adaptive_data(&mut self) {
        self.map.outcomes.clear();
    }
}
