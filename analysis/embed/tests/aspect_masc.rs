//! Trained aspect sentiment on MASC 3.0.0 MPQA annotations (#924).
//!
//! Run with `TURBO_MASC_ROOT` naming MASC's `data` directory and
//! `TURBO_MINILM_BUNDLE` naming an all-MiniLM-L6-v2 bundle. Set
//! `TURBO_ASPECT_OUT` to train and write the BIO and polarity heads, or
//! `TURBO_ASPECT_HEAD` to evaluate two previously published `THEAD001` files.
//! The ModernBERT receipt runs its encoder on the device `TURBO_TEST_DEVICE`
//! names, and on the CPU when it names none; a `TURBO_TEST_REQUIRE_<BACKEND>`
//! variable does not choose the device.

mod aspect_masc_support;

#[test]
#[ignore = "external MASC and MiniLM receipt; run explicitly with release optimizations"]
fn trained_aspect_sentiment_matches_the_recorded_metrics() {
    aspect_masc_support::run();
}

#[test]
#[ignore = "external MASC and ModernBERT receipt; run explicitly with release optimizations"]
fn trained_aspect_sentiment_modernbert_receipt() {
    aspect_masc_support::run_modernbert();
}
