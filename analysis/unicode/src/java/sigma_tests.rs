use super::VISITED;

/// Characters the final-sigma machine looks at to lowercase `text`.
fn visited(text: &str, expected: &str) -> u64 {
    VISITED.with(|n| n.set(0));
    assert_eq!(crate::java::lowercase_cow(text).as_ref(), expected);
    VISITED.with(std::cell::Cell::get)
}

type Case = fn(usize) -> (String, String);

/// The four long-word shapes: contiguous sigmas, a cased mark between them,
/// an uncased mark and a MidWord joiner between them, and a long run of
/// Transparent format characters before a final sigma.
const CASES: [(&str, Case); 4] = [
    ("contiguous", |n| ("\u{3A3}".repeat(n), format!("{}\u{3C2}", "\u{3C3}".repeat(n - 1)))),
    ("mark-interleaved", |n| {
        (
            format!("\u{391}{}", "\u{345}\u{3A3}".repeat(n)),
            format!("\u{3B1}{}\u{345}\u{3C2}", "\u{345}\u{3C3}".repeat(n - 1)),
        )
    }),
    ("joiner-interleaved", |n| {
        (format!("{}\u{3A3}", "\u{3A3}\u{301}-".repeat(n - 1)), format!("{}\u{3C2}", "\u{3C3}\u{301}-".repeat(n - 1)))
    }),
    ("transparent-run", |n| {
        (format!("\u{391}-{}\u{3A3}", "\u{200B}".repeat(n)), format!("\u{3B1}-{}\u{3C2}", "\u{200B}".repeat(n)))
    }),
];

#[test]
fn long_sigma_words_are_looked_at_a_bounded_number_of_times_a_character() {
    for (what, case) in CASES {
        let [n1, n4] = [2_500usize, 10_000].map(|n| {
            let (text, expected) = case(n);
            (visited(&text, &expected), text.chars().count() as u64)
        });
        // A few looks a character, so four times the characters is about
        // four times the work. Rescanning the word for each sigma would be
        // sixteen times.
        assert!(n1.0 > 0 && n4.0 * 10 <= n1.0 * 41, "{what}: work grew faster than the word: N={}, 4N={}", n1.0, n4.0);
        assert!(n4.0 <= 4 * n4.1, "{what}: {} looks for {} characters", n4.0, n4.1);
    }
}
