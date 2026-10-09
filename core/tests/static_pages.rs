//! TURBO_CPU_STATIC_PAGES on the CPU backend this build links, over the
//! sealed bundle in testdata/tiny-static-bundle: a model whose sessions
//! read copies of its tables in huge pages gives the same bits as one that
//! reads the mapped file, at every precision, and a value it does not know
//! refuses the session. The variable is the process's, so this file holds
//! the one test that sets it.

mod common;

use common::*;
use turbo::*;

const TEXTS: [&str; 5] =
    ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a", "Café naïve RÉSUMÉ, 东京", ""];

/// Each precision's vectors, from a model loaded and first run with
/// TURBO_CPU_STATIC_PAGES as `pages` says.
fn vectors(pages: &str) -> Vec<Vec<Vec<f32>>> {
    // SAFETY: this binary's one test, so no other thread reads the
    // environment.
    unsafe { std::env::set_var("TURBO_CPU_STATIC_PAGES", pages) };
    let l = Loaded::load(&testdata().join("tiny-static-bundle")).unwrap_or_else(|e| panic!("{e:?}"));
    [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT]
        .iter()
        .map(|&p| {
            let s = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap_or_else(|e| panic!("{e:?}"));
            s.embed(&TEXTS, None).unwrap_or_else(|e| panic!("{e:?}"))
        })
        .collect()
}

#[test]
fn huge_pages_give_the_mapped_table_s_bits() {
    let mapped = vectors("mapped");
    assert_eq!(vectors("huge"), mapped);
    assert_eq!(vectors(""), mapped);

    unsafe { std::env::set_var("TURBO_CPU_STATIC_PAGES", "big") };
    let l = Loaded::load(&testdata().join("tiny-static-bundle")).unwrap_or_else(|e| panic!("{e:?}"));
    let Err(e) = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL))) else {
        panic!("TURBO_CPU_STATIC_PAGES=big made a session");
    };
    assert!(e.is(status::INVALID_ARGUMENT, "TURBO_CPU_STATIC_PAGES"), "{e:?}");
    unsafe { std::env::remove_var("TURBO_CPU_STATIC_PAGES") };
}
