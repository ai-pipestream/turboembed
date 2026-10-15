//! Spanish dates and times.

use std::sync::OnceLock;

use crate::engine::{Dim, Grammar, Lexicon, NumeralStyle, Output, Rule, WordForms};

use super::{DIMS as NUMBER_DIMS, NUMERAL, SPANISH};

const DIMS: u16 = NUMBER_DIMS;

/// The Spanish time grammar.
pub struct SpanishTimes {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance.
pub static SPANISH_TIMES: SpanishTimes = SpanishTimes { lexicon: OnceLock::new() };

impl WordForms for SpanishTimes {
    fn folds(&self) -> bool {
        true
    }

    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        SPANISH.split(word, cuts);
    }
}

impl Grammar for SpanishTimes {
    fn name(&self) -> &'static str {
        "es-times"
    }

    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| Lexicon::new(super::lexicon::ENTRIES))
    }

    fn rules(&self) -> &'static [Rule] {
        SPANISH.rules()
    }

    fn dims(&self) -> u16 {
        DIMS
    }

    fn numeral(&self) -> Dim {
        NUMERAL
    }

    fn style(&self) -> NumeralStyle {
        SPANISH.style()
    }

    fn words(&self) -> &dyn WordForms {
        self
    }

    fn output(&self, _dim: Dim) -> Option<Output> {
        None
    }
}
