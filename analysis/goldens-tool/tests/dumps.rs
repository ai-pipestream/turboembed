//! Checks the checked in decision point dumps at the header level: every
//! golden of a run that asks a model has a dump beside it, unless the
//! annotator threw on its input (an error record), every dump a golden, and
//! each dump's header parses and names the input and the model of its path.

use turbo_goldens_tool::{output, paths, recipe};

#[test]
fn every_checked_in_golden_of_a_model_has_a_dump_whose_header_parses() {
    // A golden with an error record has no dump, and neither has one of a
    // run without a model (the plain tokenizers and the newline detector).
    let (mut with_model, mut without_model, mut threw) = (0, 0, 0);
    for annotator in recipe::DUMPED_ANNOTATORS {
        let mut stack = vec![paths::data_dir().join(annotator)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|e| e == paths::GOLDEN_EXTENSION) {
                    if output::golden_names_a_model(&p).unwrap() {
                        with_model += 1;
                        threw += usize::from(output::golden_threw(&p).unwrap());
                    } else {
                        without_model += 1;
                    }
                }
            }
        }
    }
    assert!(with_model > 600, "{with_model} goldens of runs with a model");
    assert!(without_model > 300, "{without_model} goldens of runs without a model");
    assert_eq!(output::check_dumps(&paths::data_dir()).unwrap(), with_model - threw);
}
