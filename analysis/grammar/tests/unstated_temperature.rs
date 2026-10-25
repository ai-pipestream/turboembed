//! A temperature with no scale keeps its stated unit, as a custom unit
//! does, without inventing a conversion to Kelvin or a range failure.

use turbo_analysis::{Ix, Layer, ViewId};
use turbo_grammar::Dec;
use turbo_grammar::annotator::GrammarAnnotator;

#[test]
fn degrees_without_a_temperature_scale_keep_the_written_unit() {
    let q = GrammarAnnotator::english().find("It was five degrees outside.", ViewId::SOURCE).quantities;
    assert_eq!(q.check(), Ok(()));
    assert_eq!(q.len(), 1);
    assert_eq!(q.value(Ix::new(0)).to_plain_string(), "5");
    assert_eq!(q.unit(Ix::new(0)), "deg");
    assert_eq!(q.base_unit(Ix::new(0)), "deg");
    assert_eq!(q.normalized_dec(Ix::new(0)), Some(Dec::from_i64(5)));
}
