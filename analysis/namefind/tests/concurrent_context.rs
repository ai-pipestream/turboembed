//! Rule finders share immutable dictionaries without document state leaking.
mod native_support;

use std::sync::Barrier;

#[test]
fn shared_rule_finder_is_deterministic_on_multiple_threads() {
    let finder = native_support::finder(2000, "person");
    let expected = native_support::entities(&finder);
    let barrier = Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (finder, expected, barrier) = (&finder, &expected, &barrier);
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..16 {
                    let actual = native_support::entities(finder);
                    assert_eq!(actual.spans.columns(), expected.spans.columns());
                    assert_eq!(actual.tokens, expected.tokens);
                    assert_eq!(actual.kind, expected.kind);
                    assert_eq!(actual.score, expected.score);
                    assert!(std::sync::Arc::ptr_eq(&actual.kinds, &expected.kinds));
                }
            });
        }
    });
}
