//! Quantities next to dollar-prefixed region symbols, in both notations.
//!
//! The texts are the `nightly-*` inputs in `analysis/goldens/data/inputs/quantity`. A
//! region symbol such as `A$`, `US$` or `HK$` makes the amount money, so
//! the trailing `2m` is not a quantity of metres. The controls (`$2m`,
//! `2m` alone, `xA$2m`, and `A$1,2m` read in US notation) pin what the
//! extractors find when no region symbol applies.

mod money_common;

use money_common::{FoundQuantity, cur, dec, extract_money, extract_quantity, same};
use turbo_analysis::Span;
use turbo_numeric::NumberNotation::{self, LatinEu, LatinUs};
use turbo_numeric::QuantityExtractor;
use turbo_numeric::money::MoneyExtractor;

/// What the default extractors find in one input.
enum Want {
    /// Money with this span, amount and currency, and no quantity.
    Money(u32, u32, &'static str, &'static str),
    /// One quantity with this span, value and unit, and no money.
    Quantity(u32, u32, &'static str, &'static str),
    /// Neither money nor a quantity.
    Nothing,
}

fn input(id: &str) -> String {
    let path = turbo_goldens::data_dir().join("inputs/quantity").join(format!("{id}.txt"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn cases(notation: NumberNotation) -> [(&'static str, &'static str, Want); 8] {
    let comma = match notation {
        // `1,2` is not a valid US grouping, so nothing is read.
        LatinUs => Want::Nothing,
        LatinEu => Want::Money(0, 6, "1200000", "AUD"),
    };
    [
        ("nightly-aud-prefix", "A$2m", Want::Money(0, 4, "2000000", "AUD")),
        ("nightly-usd-prefix", "US$2m", Want::Money(0, 5, "2000000", "USD")),
        ("nightly-hkd-prefix", "HK$2m", Want::Money(0, 5, "2000000", "HKD")),
        ("nightly-dollar-control", "$2m", Want::Money(0, 3, "2000000", "USD")),
        ("nightly-plain-control", "2m", Want::Quantity(0, 2, "2", "m")),
        ("nightly-decimal-comma", "A$1,2m", comma),
        // `é` is two bytes, so `A$` starts at byte 3.
        ("nightly-utf8-prefix", "\u{e9} A$2m", Want::Money(3, 7, "2000000", "AUD")),
        // `xA` is no region prefix and a letter before `$` rejects the
        // symbol, so `2m` after it is an ordinary quantity.
        ("nightly-inner-prefix", "xA$2m", Want::Quantity(3, 5, "2", "m")),
    ]
}

#[track_caller]
fn assert_one_quantity(found: &[FoundQuantity], span: Span, value: &str, unit: &str) {
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].span, span);
    assert!(same(&found[0].value, &dec(value)), "expected {value}, got {}", found[0].value);
    assert_eq!(found[0].unit, unit);
}

fn check(notation: NumberNotation) {
    let quantities = QuantityExtractor::new(notation);
    let money = MoneyExtractor::new(notation);
    assert!(!quantities.opennlp_parity() && !money.opennlp_parity());
    for (id, text, want) in cases(notation) {
        assert_eq!(input(id), text, "{id}: the input file changed");
        let q = extract_quantity(&quantities, text);
        let m = extract_money(&money, text);
        match want {
            Want::Money(start, end, amount, currency) => {
                assert!(q.is_empty(), "{notation:?} {id}: money must exclude the quantity: {q:?}");
                assert_eq!(m.len(), 1, "{notation:?} {id}: {m:?}");
                assert_eq!(m[0].span, Span::new(start, end), "{notation:?} {id}");
                assert!(same(&m[0].amount, &dec(amount)), "{notation:?} {id}: {}", m[0].amount);
                assert_eq!(m[0].currency, cur(currency), "{notation:?} {id}");
            }
            Want::Quantity(start, end, value, unit) => {
                assert!(m.is_empty(), "{notation:?} {id}: {m:?}");
                assert_one_quantity(&q, Span::new(start, end), value, unit);
            }
            Want::Nothing => {
                assert!(q.is_empty() && m.is_empty(), "{notation:?} {id}: {q:?} {m:?}");
            }
        }
    }
}

#[test]
fn latin_us_region_prefixes_are_money_not_quantities() {
    check(LatinUs);
}

#[test]
fn latin_eu_region_prefixes_are_money_not_quantities() {
    check(LatinEu);
}

/// The money exclusion is what removes the quantity: the same number after
/// a plain word and a space is a quantity.
#[test]
fn the_number_alone_is_a_quantity_in_both_notations() {
    for notation in [LatinUs, LatinEu] {
        let q = extract_quantity(&QuantityExtractor::new(notation), "A 2m");
        assert_one_quantity(&q, Span::new(2, 4), "2", "m");
    }
}
