//! Dutch dates and times.

use std::sync::OnceLock;

use crate::engine::{Dim, Grammar, Lexicon, NumeralStyle, Output, Rule, WordForms};

use super::{DIMS as NUMBER_DIMS, DUTCH, NUMERAL};

const DIMS: u16 = NUMBER_DIMS;

/// The Dutch time grammar.
pub struct DutchTimes {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance.
pub static DUTCH_TIMES: DutchTimes = DutchTimes { lexicon: OnceLock::new() };

impl Grammar for DutchTimes {
    fn name(&self) -> &'static str {
        "nl-times"
    }

    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| Lexicon::new(super::lexicon::ENTRIES))
    }

    fn rules(&self) -> &'static [Rule] {
        DUTCH.rules()
    }

    fn dims(&self) -> u16 {
        DIMS
    }

    fn numeral(&self) -> Dim {
        NUMERAL
    }

    fn style(&self) -> NumeralStyle {
        DUTCH.style()
    }

    fn words(&self) -> &dyn WordForms {
        &DUTCH
    }

    fn output(&self, _dim: Dim) -> Option<Output> {
        None
    }
}
