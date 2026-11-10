//! The wire code of a failed step (#2125): an input over an annotator's
//! limit is a capacity refusal, not a library bug.

use turbo_analysis::Error;

use super::step_error_code;
use crate::api::{TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_PANIC};

#[test]
fn an_input_over_a_limit_is_capacity_and_other_failures_keep_their_codes() {
    let over = Error::over_limit("turbo-relation:zero-shot", "encoder max_sequence", 512, 528);
    assert_eq!(step_error_code(&over), TURBO_E_CAPACITY);
    let other = Error::Annotator { annotator: "x", detail: "the scorer failed: oom".to_owned() };
    assert_eq!(step_error_code(&other), TURBO_E_INTERNAL);
    let panic = Error::Annotator { annotator: "x", detail: "panic: index out of range".to_owned() };
    assert_eq!(step_error_code(&panic), TURBO_E_PANIC);
    // A detail that only looks like a limit, and could quote the document,
    // is not one.
    let quoted = Error::Annotator { annotator: "x", detail: "limit: Jane Doe".to_owned() };
    assert_eq!(step_error_code(&quoted), TURBO_E_INTERNAL);
}
