//! Checks the checked in decision point dumps at the header level: every
//! golden of an annotator with a model has a dump beside it, every dump a
//! golden, and each dump's header parses and names the input and the model
//! of its path.

use turbo_goldens_tool::{output, paths, recipe};

#[test]
fn every_checked_in_golden_has_a_dump_whose_header_parses() {
    let inputs: usize = recipe::INPUT_SETS
        .iter()
        .filter(|set| **set != recipe::NORMALIZER_INPUT_SET)
        .map(|set| std::fs::read_dir(paths::inputs_dir().join(set)).unwrap().count())
        .sum();
    let dumps = output::check_dumps(&paths::data_dir()).unwrap();
    assert_eq!(dumps, recipe::DUMPED_ANNOTATORS.len() * inputs);
}
