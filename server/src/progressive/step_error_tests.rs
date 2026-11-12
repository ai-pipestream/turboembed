//! The wire code of a failed step (#2125, #2134): an input over an
//! annotator's limit is a capacity refusal, not a library bug, and the
//! step failure, the step's layer error and an exported record of it all
//! say so. An [`Error::Annotator`] detail is never mistaken for one,
//! whatever its shape.

use turbo_analysis::Error;
use turbo_analysis::export::ErrorRecord;

use crate::api::{TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_PANIC};
use crate::proto::result::LayerErrorKind as K;
use crate::results::receipt::error::{kind_code, of_error, of_record};

fn record(e: &Error) -> ErrorRecord {
    match e {
        Error::Annotator { annotator, detail } => {
            ErrorRecord::Annotator { annotator: (*annotator).to_owned(), detail: detail.clone() }
        }
        Error::OverLimit { annotator, limit, max, got } => ErrorRecord::OverLimit {
            annotator: (*annotator).to_owned(),
            limit: (*limit).to_owned(),
            max: *max,
            got: *got,
        },
        _ => unreachable!(),
    }
}

#[test]
fn an_input_over_a_limit_is_capacity_and_other_failures_keep_their_codes() {
    let over = Error::over_limit("turbo-relation:zero-shot", "encoder max_sequence", 512, 528);
    let other = Error::Annotator { annotator: "x", detail: "the scorer failed: oom".to_owned() };
    let panic = Error::Annotator { annotator: "x", detail: "panic: index out of range".to_owned() };
    // An Error::Annotator detail is never treated as a limit, even when
    // it has the exact shape of a limit string.
    let quoted = Error::Annotator { annotator: "x", detail: "limit: patient jane doe 1, got 2".to_owned() };
    let shaped = Error::Annotator { annotator: "x", detail: "limit: encoder max_sequence 512, got 528".to_owned() };
    for (e, kind, code) in [
        (&over, K::Limit, TURBO_E_CAPACITY),
        (&other, K::Annotator, TURBO_E_INTERNAL),
        (&panic, K::Panic, TURBO_E_PANIC),
        (&quoted, K::Annotator, TURBO_E_INTERNAL),
        (&shaped, K::Annotator, TURBO_E_INTERNAL),
    ] {
        for got in [of_error(e), of_record(&record(e))] {
            assert_eq!((got.kind, got.turbo_code), (kind as i32, code), "{e}");
            assert_eq!(kind_code(kind), code);
        }
    }
    let unlisted =
        ErrorRecord::OverLimit { annotator: "custom".into(), limit: "patient jane doe".into(), max: 1, got: 2 };
    let got = of_record(&unlisted);
    assert_eq!((got.kind, got.turbo_code), (K::Annotator as i32, TURBO_E_INTERNAL));
}
