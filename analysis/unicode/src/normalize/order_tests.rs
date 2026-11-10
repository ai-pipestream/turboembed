use super::{CLASS_LOOKUPS, Form, aligned, hf, normalize};

/// One run of combining marks whose classes alternate (U+0301 is class
/// 230, U+0323 class 220).
fn run(marks: usize) -> String {
    (0..marks).map(|i| if i % 2 == 0 { '\u{301}' } else { '\u{323}' }).collect()
}

/// Combining classes looked up while `f` orders a run of `marks` marks,
/// after checking that every lower mark comes first and the higher marks
/// keep their place after them.
fn lookups(what: &str, marks: usize, f: &dyn Fn(&str) -> Vec<char>) -> u64 {
    let text = run(marks);
    CLASS_LOOKUPS.with(|n| n.set(0));
    let out = f(&text);
    let looked = CLASS_LOOKUPS.with(std::cell::Cell::get);
    assert_eq!(out.len(), marks, "{what}");
    let above = out.iter().position(|&c| c == '\u{301}').expect("the higher class is kept");
    assert_eq!(above, marks / 2, "{what}: every lower mark first");
    assert!(out[above..].iter().all(|&c| c == '\u{301}'), "{what}: the higher marks after");
    looked
}

/// One way to normalize a text to its characters.
type Normalizer = fn(Form, &str) -> Vec<char>;

/// Ordering by insertion looked at the classes of a run a quadratic number
/// of times, so a text of a few hundred kilobytes of marks alone held a
/// thread for seconds. Each normalizer now sorts the run: four times the
/// marks costs well under eight times the lookups (insertion would be
/// sixteen), and a run of n marks takes no more than about 2 n log2 n.
#[test]
fn a_long_run_of_combining_marks_is_ordered_in_n_log_n_lookups() {
    let forms: [(&str, Normalizer); 3] = [
        ("normalize", |form, t| {
            let mut out = String::new();
            normalize(form, t, &mut out);
            out.chars().collect()
        }),
        ("aligned", |form, t| aligned(form, t).map(|(c, _)| c).collect()),
        ("hf::aligned", |form, t| hf::aligned(form, t).map(|(c, _)| c).collect()),
    ];
    for form in [Form::Nfd, Form::Nfc] {
        for (name, f) in forms {
            let what = format!("{name} {form:?}");
            let [n1, n4] = [4_000usize, 16_000].map(|marks| lookups(&what, marks, &|t| f(form, t)));
            assert!(n4 < n1 * 6, "{what}: 4N/N class lookups {n4} against {n1}");
            assert!(n4 <= 2 * 16_000 * 14, "{what}: {n4} class lookups for 16000 marks");
        }
    }
}
