//! Trained sarcasm and irony head on the MIT-licensed CSC (#925).
//!
//! Set `TURBO_CSC_CSV` to CSC `data_full.csv` and `TURBO_MINILM_BUNDLE` to
//! an all-MiniLM-L6-v2 bundle. `TURBO_PRAGMATICS_OUT` trains and writes a
//! head and configuration; `TURBO_PRAGMATICS_HEAD` reloads published assets.

mod pragmatics_csc_support;

#[test]
#[ignore = "external CSC and MiniLM receipt; run explicitly with release optimizations"]
fn trained_pragmatics_matches_the_recorded_metrics() {
    pragmatics_csc_support::run();
}
