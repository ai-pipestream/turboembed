//! Language modules, each a [`crate::Grammar`].

pub mod cjk;
pub mod en;
pub mod es;
pub mod ja;
pub mod nl;
pub mod zh;

pub mod it;

pub mod pt;

mod unit_pack;

use std::sync::OnceLock;

use crate::engine::{Entry, Lexicon, Rule};

/// The lexicon kept in `cell`: made, the first time only, of a number
/// grammar's `base` entries and the entries `more` gives.
pub(crate) fn concat_lexicon<'a>(
    cell: &'a OnceLock<Lexicon>,
    base: &[Entry],
    more: impl FnOnce() -> Vec<Entry>,
) -> &'a Lexicon {
    cell.get_or_init(|| {
        let mut entries = base.to_vec();
        entries.extend(more());
        Lexicon::new(&entries)
    })
}

/// The rules kept in `cell`: made, the first time only, of `parts` in
/// order.
pub(crate) fn concat_rules(cell: &'static OnceLock<Vec<Rule>>, parts: &[&[Rule]]) -> &'static [Rule] {
    cell.get_or_init(|| parts.concat())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Dim;

    #[test]
    fn concat_lexicon_holds_both_parts_and_is_made_once() {
        let e = |word| Entry { word, dim: Dim(0), value: 0, flags: 0 };
        let cell = OnceLock::new();
        let lex = concat_lexicon(&cell, &[e("one")], || vec![e("km")]);
        assert_eq!(lex.lookup("one").len(), 1);
        assert_eq!(lex.lookup("km").len(), 1);
        let again = concat_lexicon(&cell, &[], || unreachable!("made once"));
        assert!(std::ptr::eq(lex, again));
    }

    #[test]
    fn concat_rules_keeps_the_parts_in_order_and_is_made_once() {
        static CELL: OnceLock<Vec<Rule>> = OnceLock::new();
        let (a, b) = (&crate::lang::cjk::RULES[..1], &crate::lang::cjk::RULES[1..2]);
        let all = concat_rules(&CELL, &[b, a]);
        assert_eq!(all.iter().map(|r| r.name).collect::<Vec<_>>(), [b[0].name, a[0].name]);
        assert!(std::ptr::eq(all, concat_rules(&CELL, &[])));
    }
}
