//! A real bundle, with its reference produced by the upstream pipeline:
//! set TURBO_TEST_BUNDLE to its directory and run with --ignored. Without
//! the variable the test fails rather than passing.

mod common;

use common::*;

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_loads_and_matches_its_reference_ids() {
    let dir = std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set");
    // turbo_tokenizer_create runs loader rules 1 to 5, which include
    // encoding every reference case and comparing the ids exactly.
    let tok = Tok::create(std::path::Path::new(&dir)).unwrap_or_else(|e| panic!("{e:?}"));
    let info = tok.info();
    assert!(info.vocab_size > 0 && info.max_seq > 0);
}
