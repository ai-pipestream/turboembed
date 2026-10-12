//! Checks the checked in decision point dumps at the header level: every
//! golden of an annotator with a model has a dump beside it, unless the
//! annotator threw on its input (an error record), every dump a golden, and
//! each dump's header parses and names the input and the model of its path.

use turbo_goldens_tool::{output, paths, recipe};

#[test]
fn every_checked_in_golden_has_a_dump_whose_header_parses() {
    // The annotators with a model run on every input set but the
    // normalizers' own; a golden with an error record has no dump.
    let (mut goldens, mut threw) = (0, 0);
    for annotator in recipe::DUMPED_ANNOTATORS {
        let mut stack = vec![paths::data_dir().join(annotator)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|e| e == paths::GOLDEN_EXTENSION) {
                    goldens += 1;
                    threw += usize::from(output::golden_threw(&p).unwrap());
                }
            }
        }
    }
    let inputs: usize = recipe::INPUT_SETS
        .iter()
        .filter(|set| ![recipe::NORMALIZER_INPUT_SET, recipe::NUMBER_INPUT_SET, recipe::MONEY_INPUT_SET].contains(set))
        .map(|set| std::fs::read_dir(paths::inputs_dir().join(set)).unwrap().count())
        .sum();
    assert_eq!(goldens, recipe::DUMPED_ANNOTATORS.len() * inputs);
    assert_eq!(output::check_dumps(&paths::data_dir()).unwrap(), goldens - threw);
}
